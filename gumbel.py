from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template as unsloth_get_chat_template
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from transformers.optimization import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("prompt.txt", "r") as f:
    INSTRUCTION = f.read()

class GumbelSteganographer(pl.LightningModule):
    def __init__(
        self,
        llm_model_name,
        sim_model_name,
        lora_config,
        temperature,
        lambda_sim,
        optimizer_args,
        scheduler_args,
        model_save_path_prefix,
        debug=False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_name_for_chat_template = (
            llm_model_name
        )

        # Always use Unsloth for model loading
        print("Using Unsloth for model loading.")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
        )
        self.model = FastLanguageModel.get_peft_model(self.model, **lora_config)

        # Always use Unsloth chat templates
        if "llama" in self.model_name_for_chat_template.lower():
            self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="llama-3.1")
        elif "qwen" in self.model_name_for_chat_template.lower():
            self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="qwen-2.5")
        elif "gemma" in self.model_name_for_chat_template.lower():
            self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="gemma-3")
        elif "olmo" in self.model_name_for_chat_template.lower():
            self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="olmo")

        if self.tokenizer.pad_token is None:
            print("Tokenizer does not have a pad_token, setting it to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.sim_model = SentenceTransformer(self.hparams.sim_model_name, device=self.device)
        print("Setting sim_model to eval mode and freezing parameters.")
        self.sim_model.eval()
        for param in self.sim_model.parameters():
            param.requires_grad = False

        self.temperature = float(self.hparams.temperature)
        self.lambda_sim = float(self.hparams.lambda_sim)
        self.debug = self.hparams.debug
        self.model_save_path_prefix = self.hparams.model_save_path_prefix

        self.zero_id = self.tokenizer.convert_tokens_to_ids("0")
        self.one_id = self.tokenizer.convert_tokens_to_ids("1")
        if isinstance(self.zero_id, list): self.zero_id = self.zero_id[0]
        if isinstance(self.one_id, list): self.one_id = self.one_id[0]

        self.default_generation_params = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        self.generation_length_margin = 10
        self.validation_step_outputs = []

    def _prepare_encode_prompts(
        self, buffers: list[str], bits_to_hide: list[int]
    ) -> list[list[dict]]:
        return [
            [{"role": "user","content": f"{INSTRUCTION}\\n[ENCODE]\\nBuffer: {b}\\nHide bit: {bit}\\n",}]
            for b, bit in zip(buffers, bits_to_hide)
        ]

    def _prepare_decode_prompts(self, encoded_texts: list[str]) -> list[list[dict]]:
        return [
            [{"role": "user", "content": f"{INSTRUCTION}\\n[DECODE]\\nEncoded: {text}\\n",}]
            for text in encoded_texts
        ]

    def generate_encoded_text_or_embeddings(
        self,
        buffers: str | list[str],
        bits_to_hide: int | list[int],
        produce_embeddings: bool,
        generation_max_new_tokens: int | None = None,
    ):
        was_single_input = isinstance(buffers, str)
        _buffers = [buffers] if was_single_input else buffers
        if isinstance(bits_to_hide, int):
            _bits_to_hide = [bits_to_hide] * len(_buffers)
        elif bits_to_hide is not None and len(bits_to_hide) != len(_buffers):
            raise ValueError("Buffers and bits_to_hide must have the same number of elements if bits_to_hide is a list.")
        else:
            _bits_to_hide = bits_to_hide

        if not _buffers:
            max_buffer_token_count = 0
        else:
            buffer_token_lengths = [len(self.tokenizer.tokenize(b)) for b in _buffers]
            max_buffer_token_count = (max(buffer_token_lengths) if buffer_token_lengths else 0)
        
        dynamic_generation_length = max(5, max_buffer_token_count + self.generation_length_margin)

        list_of_encode_conversations = self._prepare_encode_prompts(_buffers, _bits_to_hide)
        
        enc_inputs_dict = self.tokenizer.apply_chat_template(
            list_of_encode_conversations,
            padding=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False, return_dict=True
        ).to(self.device)

        if produce_embeddings:
            current_input_ids = enc_inputs_dict["input_ids"]
            current_attention_mask = enc_inputs_dict["attention_mask"]
            emb_matrix = self.model.get_input_embeddings().weight
            list_generated_soft_embs = []
            list_generated_hard_ids = []
            num_tokens_to_generate = dynamic_generation_length

            for _ in range(num_tokens_to_generate):
                model_outputs = self.model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                next_token_logits = model_outputs.logits[:, -1, :]

                # Path for soft embeddings (for loss_bit)
                soft_dist_for_loss_bit = F.gumbel_softmax(next_token_logits, tau=self.temperature, hard=False, dim=-1)
                current_soft_emb = torch.matmul(soft_dist_for_loss_bit.unsqueeze(1).to(emb_matrix.dtype), emb_matrix)
                list_generated_soft_embs.append(current_soft_emb)
                
                # Path for hard IDs (for loss_sim and next token input)
                # Conditional STE based on lambda_sim
                if self.hparams.lambda_sim > 0.0:
                    # Use STE: hard=True for one-hot output, soft for backward pass
                    ste_one_hot_output = F.gumbel_softmax(next_token_logits, tau=self.temperature, hard=True, dim=-1)
                    hard_id_for_next_step = torch.argmax(ste_one_hot_output, dim=-1).unsqueeze(1)
                else:
                    # No STE needed if lambda_sim is 0, just use argmax on soft distribution
                    hard_id_for_next_step = torch.argmax(soft_dist_for_loss_bit, dim=-1).unsqueeze(1)
                
                list_generated_hard_ids.append(hard_id_for_next_step)
                
                current_input_ids = torch.cat([current_input_ids, hard_id_for_next_step], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(hard_id_for_next_step, device=self.device)], dim=1)

            final_soft_embs = torch.cat(list_generated_soft_embs, dim=1)
            final_hard_ids = torch.cat(list_generated_hard_ids, dim=1)
            intermediate_encoded_texts = self.tokenizer.batch_decode(final_hard_ids, skip_special_tokens=True)
            generated_attention_mask = torch.ones_like(final_hard_ids, device=self.device)
            return final_soft_embs, intermediate_encoded_texts, generated_attention_mask
        else:
            if generation_max_new_tokens is not None:
                max_tokens = generation_max_new_tokens
            else:
                max_tokens = dynamic_generation_length
            generated_ids = self.model.generate(
                input_ids=enc_inputs_dict['input_ids'],
                attention_mask=enc_inputs_dict['attention_mask'],
                max_new_tokens=max_tokens,
                **self.default_generation_params,
            )
            encoded_texts_generated = self.tokenizer.batch_decode(
                generated_ids[:, enc_inputs_dict["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            stripped_texts = [text.strip() for text in encoded_texts_generated]
            return stripped_texts[0] if was_single_input else stripped_texts

    def predict_bits_from_encoded_text(
        self,
        encoded_texts: str | list[str],
        soft_embs_from_encoder: torch.Tensor | None = None,
        attention_mask_for_soft_embs: torch.Tensor | None = None,
    ):
        was_single_input = isinstance(encoded_texts, str)
        _encoded_texts = [encoded_texts] if was_single_input else encoded_texts
        list_of_decode_conversations = self._prepare_decode_prompts(_encoded_texts)
        
        dec_inputs_dict = self.tokenizer.apply_chat_template(
            list_of_decode_conversations,
            padding=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False, return_dict=True
        ).to(self.device)

        if soft_embs_from_encoder is not None:
            if attention_mask_for_soft_embs is None:
                raise ValueError("attention_mask_for_soft_embs must be provided if soft_embs_from_encoder is given.")
            
            prompt_embeds = self.model.get_input_embeddings()(dec_inputs_dict["input_ids"])
            _soft_embs = soft_embs_from_encoder.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            _attn_mask_soft_embs = attention_mask_for_soft_embs.to(device=prompt_embeds.device)

            full_embeds = torch.cat([prompt_embeds, _soft_embs], dim=1)
            combined_attention_mask = torch.cat([dec_inputs_dict["attention_mask"], _attn_mask_soft_embs], dim=1)
            
            logits_dec = self.model(inputs_embeds=full_embeds, attention_mask=combined_attention_mask).logits
            sequence_lengths = dec_inputs_dict["attention_mask"].sum(dim=1)
            batch_indices = torch.arange(logits_dec.size(0), device=logits_dec.device)
            last_prompt_token_logits = logits_dec[batch_indices, sequence_lengths - 1]
            return last_prompt_token_logits
        else:
            decoded_ids = self.model.generate(
                input_ids=dec_inputs_dict['input_ids'],
                attention_mask=dec_inputs_dict['attention_mask'],
                max_new_tokens=1,
                **self.default_generation_params,
            )
            predicted_bit_texts = self.tokenizer.batch_decode(
                decoded_ids[:, dec_inputs_dict["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            stripped_texts = [text.strip() for text in predicted_bit_texts]
            return stripped_texts[0] if was_single_input else stripped_texts

    def training_step(self, batch, batch_idx):
        buffers, bits = batch
        
        current_bits_list = bits.tolist()
        soft_embs, intermediate_encoded_texts, enc_attention_mask = \
            self.generate_encoded_text_or_embeddings(
                buffers, current_bits_list, produce_embeddings=True
            )

        full_decode_logits = self.predict_bits_from_encoded_text(
            intermediate_encoded_texts,
            soft_embs_from_encoder=soft_embs,
            attention_mask_for_soft_embs=enc_attention_mask,
        )
        
        target_bits_on_device = bits.to(full_decode_logits.device)
        target_token_ids = torch.full_like(target_bits_on_device, self.zero_id, dtype=torch.long)
        target_token_ids[target_bits_on_device == 1] = self.one_id
        
        loss_bit = torch.nn.functional.cross_entropy(full_decode_logits, target_token_ids)

        orig_emb = self.sim_model.encode(buffers, convert_to_tensor=True)
        enc_emb = self.sim_model.encode(intermediate_encoded_texts, convert_to_tensor=True)
        
        sim_scores = util.cos_sim(orig_emb, enc_emb).diag()
        loss_sim = (1 - sim_scores).mean()

        loss = loss_bit + self.lambda_sim * loss_sim.to(loss_bit.device)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_bit_loss", loss_bit, on_step=True, on_epoch=True, logger=True)
        self.log("train_sim_loss", loss_sim, on_step=True, on_epoch=True, logger=True)

        if self.hparams.debug and self.global_step % 10 == 0 and batch_idx == 0:
            predicted_bits_texts_log = self.predict_bits_from_encoded_text(intermediate_encoded_texts)

            print(f"--- Training Step {self.global_step} (Batch Item 0) ---")
            print(f'Buffer: {buffers[0]}')
            print(f'Encoded (for DECODE): {intermediate_encoded_texts[0]}')
            print(f"Target Bit: {current_bits_list[0]}")
            print(f"Predicted Bit (by DECODE part): {predicted_bits_texts_log[0] if isinstance(predicted_bits_texts_log, list) else predicted_bits_texts_log}\n")
        
        return loss

    def validation_step(self, batch, batch_idx):
        buffers, bits = batch
        
        for i in range(len(buffers)):
            buf = buffers[i]
            target_bit_val = bits[i].item()

            encoded_text = self.generate_encoded_text_or_embeddings(
                buf, target_bit_val, produce_embeddings=False, generation_max_new_tokens=None
            )
            predicted_bit_text = self.predict_bits_from_encoded_text(encoded_text)

            processed_predicted_bit = "UNK"
            if predicted_bit_text == "0": processed_predicted_bit = "0"
            elif predicted_bit_text == "1": processed_predicted_bit = "1"
            
            self.validation_step_outputs.append({
                "buffer": buf, "encoded_sentence": encoded_text,
                "decoded_bit": processed_predicted_bit, "target_bit": str(target_bit_val)
            })

    def on_validation_epoch_end(self):
        if not self.hparams.debug:
            if self.logger and hasattr(self.logger.experiment, 'log'):
                table = wandb.Table(columns=["buffer", "encoded_sentence", "decoded_bit", "target_bit"])
                for output in self.validation_step_outputs:
                    table.add_data(output["buffer"], output["encoded_sentence"], output["decoded_bit"], output["target_bit"])
                
                self.logger.experiment.log({f"val/epoch_{self.current_epoch}_examples": table})
            else:
                 print("Validation examples (debug mode or logger issue):")
                 for output in self.validation_step_outputs:
                     print(output)
        else:
            print(f"--- Validation Epoch {self.current_epoch} ---")
            for output in self.validation_step_outputs:
                print(f'Buffer: "{output["buffer"]}"')
                print(f'Target Bit: {output["target_bit"]}')
                print(f'Encoded: "{output["encoded_sentence"]}"')
                print(f"Decoded bit: {output['decoded_bit']}\n")
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.hparams.optimizer_args)
        
        if 'num_training_steps' not in self.hparams.scheduler_args:
            raise ValueError("scheduler_args must contain 'num_training_steps'. Calculate in train.py.")

        scheduler = get_linear_schedule_with_warmup(optimizer, **self.hparams.scheduler_args)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_end(self):
        if not self.hparams.debug:
            final_save_path = f"{self.model_save_path_prefix}_final_pl"
            print(f"Attempting to save model to {final_save_path}...")
            self.model.save_pretrained(final_save_path)
            self.tokenizer.save_pretrained(final_save_path)
            print(f"Model and Tokenizer saved to {final_save_path}")

