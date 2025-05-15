from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template as unsloth_get_chat_template
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("prompt.txt", "r") as f:
    INSTRUCTION = f.read()

class GumbelSteganographer(torch.nn.Module):
    def __init__(
        self,
        llm_model_name: str,
        sim_model_name: str,
        lora_config: dict,
        temperature: float,
        lambda_sim: float,
        model_save_path_prefix: str,
        debug: bool = False,
    ) -> None:
        super().__init__()

        self.llm_model_name = llm_model_name
        self.sim_model_name = sim_model_name
        self.lora_config = lora_config
        self.temperature = float(temperature)
        self.lambda_sim = float(lambda_sim)
        self.model_save_path_prefix = model_save_path_prefix
        self.debug = debug

        self.model_name_for_chat_template = llm_model_name

        print("Using Unsloth for model loading.")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.llm_model_name,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        self.model = FastLanguageModel.get_peft_model(self.model, **self.lora_config)

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

        self.sim_model = SentenceTransformer(self.sim_model_name, device=DEVICE)
        print("Setting sim_model to eval mode and freezing parameters.")
        self.sim_model.eval()
        for param in self.sim_model.parameters():
            param.requires_grad = False

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

    def _prepare_encode_prompts(
        self, buffers: list[str], bits_to_hide: list[int]
    ) -> list[list[dict]]:
        return [
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user","content": f"[ENCODE]\nBuffer: {b}\nHide bit: {bit}"}
            ]
            for b, bit in zip(buffers, bits_to_hide)
        ]

    def _prepare_decode_prompts(self, encoded_texts: list[str]) -> list[list[dict]]:
        return [
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": f"[DECODE]\nEncoded: {text}"}
            ]
            for text in encoded_texts
        ]

    def generate_encoded_text_or_embeddings(
        self,
        buffers: str | list[str],
        bits_to_hide: int | list[int],
        produce_embeddings: bool,
        generation_max_new_tokens: int | None = None,
        device: torch.device = torch.device(DEVICE)
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
        ).to(device)

        if produce_embeddings:
            # Initial prompt processing
            prompt_input_ids = enc_inputs_dict["input_ids"]
            current_attention_mask = enc_inputs_dict["attention_mask"]
            
            # Convert initial prompt IDs to embeddings
            current_input_embeds = self.model.get_input_embeddings()(prompt_input_ids)
            
            emb_matrix = self.model.get_input_embeddings().weight
            list_generated_soft_embs = [] # These will be the direct output fed to the decoder
            list_generated_hard_ids = []  # These are for text for loss_sim
            num_tokens_to_generate = dynamic_generation_length

            for _ in range(num_tokens_to_generate):
                # Model takes embeddings directly now
                model_outputs = self.model(inputs_embeds=current_input_embeds, attention_mask=current_attention_mask)
                next_token_logits = model_outputs.logits[:, -1, :]

                # Soft distribution for soft embeddings (used for loss_bit and next input embed)
                soft_dist_for_next_step = F.gumbel_softmax(next_token_logits, tau=self.temperature, hard=False, dim=-1)
                # Create the soft embedding for the current step
                next_soft_emb_for_output_and_input = torch.matmul(soft_dist_for_next_step.unsqueeze(1).to(emb_matrix.dtype), emb_matrix)
                
                list_generated_soft_embs.append(next_soft_emb_for_output_and_input)
                
                # For hard IDs (loss_sim, intermediate text): Choose based on lambda_sim
                if self.lambda_sim > 0.0:
                    # Use STE: hard=True for one-hot output, soft for backward pass on this path IF it were used for grads
                    # For generating hard IDs for loss_sim, STE helps if any part of loss_sim was meant to train the generator via these hard choices.
                    # However, the primary gradient path for the generator is now through soft_embs.
                    ste_one_hot_output = F.gumbel_softmax(next_token_logits, tau=self.temperature, hard=True, dim=-1)
                    hard_id_for_text_gen = torch.argmax(ste_one_hot_output, dim=-1).unsqueeze(1)
                else:
                    # No STE needed for loss_sim if lambda_sim is 0, just use argmax on the soft distribution
                    hard_id_for_text_gen = torch.argmax(soft_dist_for_next_step, dim=-1).unsqueeze(1)
                
                list_generated_hard_ids.append(hard_id_for_text_gen)
                
                # Update inputs for the next iteration using the generated soft embedding
                current_input_embeds = torch.cat([current_input_embeds, next_soft_emb_for_output_and_input], dim=1)
                # Update attention mask
                current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(hard_id_for_text_gen, device=device)], dim=1) # Shape of hard_id is (batch, 1) like attention extension

            final_soft_embs = torch.cat(list_generated_soft_embs, dim=1)
            final_hard_ids = torch.cat(list_generated_hard_ids, dim=1)
            intermediate_encoded_texts = self.tokenizer.batch_decode(final_hard_ids, skip_special_tokens=True)
            generated_attention_mask = torch.ones_like(final_hard_ids, device=device)
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
        device: torch.device = torch.device(DEVICE)
    ):
        was_single_input = isinstance(encoded_texts, str)
        _encoded_texts = [encoded_texts] if was_single_input else encoded_texts
        list_of_decode_conversations = self._prepare_decode_prompts(_encoded_texts)
        
        dec_inputs_dict = self.tokenizer.apply_chat_template(
            list_of_decode_conversations,
            padding=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False, return_dict=True
        ).to(device)

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

    def compute_loss(self, buffers, bits, device):
        current_bits_list = bits.tolist()
        soft_embs, intermediate_encoded_texts, enc_attention_mask = \
            self.generate_encoded_text_or_embeddings(
                buffers, current_bits_list, produce_embeddings=True, device=device
            )

        full_decode_logits = self.predict_bits_from_encoded_text(
            intermediate_encoded_texts,
            soft_embs_from_encoder=soft_embs,
            attention_mask_for_soft_embs=enc_attention_mask,
            device=device
        )
        
        target_bits_on_device = bits.to(full_decode_logits.device)
        target_token_ids = torch.full_like(target_bits_on_device, self.zero_id, dtype=torch.long)
        target_token_ids[target_bits_on_device == 1] = self.one_id
        
        loss_bit = torch.nn.functional.cross_entropy(full_decode_logits, target_token_ids)

        orig_emb = self.sim_model.encode(buffers, convert_to_tensor=True, device=device)
        enc_emb = self.sim_model.encode(intermediate_encoded_texts, convert_to_tensor=True, device=device)
        
        sim_scores = util.cos_sim(orig_emb, enc_emb).diag()
        loss_sim = (1 - sim_scores).mean()

        total_loss = loss_bit + self.lambda_sim * loss_sim.to(loss_bit.device)
        
        return total_loss, loss_bit, loss_sim, intermediate_encoded_texts
        
    def save_model(self):
        if not self.debug:
            final_save_path = f"{self.model_save_path_prefix}_final_pytorch_unsloth"
            print(f"Attempting to save Unsloth model to {final_save_path}...")
            self.model.save_pretrained(final_save_path)
            self.tokenizer.save_pretrained(final_save_path)
            print(f"Unsloth Model and Tokenizer saved to {final_save_path}")

