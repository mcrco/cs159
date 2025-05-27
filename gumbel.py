from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template as unsloth_get_chat_template
import torch
import torch.nn.functional as F
from sentence_transformers import util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("prompt.txt", "r") as f:
    INSTRUCTION = f.read()

class GumbelSteganographer(torch.nn.Module):
    def __init__(
        self,
        llm_model_name: str,
        lora_config: dict,
        temperature: float,
        model_save_path_prefix: str,
        debug: bool = False,
    ) -> None:
        super().__init__()

        self.llm_model_name = llm_model_name
        self.lora_config = lora_config
        self.temperature = float(temperature)
        self.model_save_path_prefix = model_save_path_prefix
        self.debug = debug

        self.model_name_for_chat_template = llm_model_name

        print("Using Unsloth for trainable model loading.")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.llm_model_name,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        self.model = FastLanguageModel.get_peft_model(self.model, **self.lora_config)

        # Setup tokenizer specific chat templates if needed
        if "llama" in self.model_name_for_chat_template.lower():
            self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="llama-3.1")
        elif "qwen" in self.model_name_for_chat_template.lower():
            self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="qwen-2.5")
        elif "gemma" in self.model_name_for_chat_template.lower():
            self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="gemma-3")
        elif "olmo" in self.model_name_for_chat_template.lower():
            self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="olmo")

        if self.tokenizer.pad_token is None:
            print("Trainable model tokenizer does not have a pad_token, setting it to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model.config.pad_token_id is None: # Ensure the model config also reflects this
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

        print("Loading separate, frozen model for embeddings.")
        # Load the base model again for embeddings, without PEFT adapters and frozen
        # We use the same tokenizer that was potentially modified for chat templates,
        # as it's primarily for tokenizing input text.
        # The embedding model itself won't use the chat template logic for its forward pass.
        self.embedding_model, _ = FastLanguageModel.from_pretrained( # Tokenizer is not needed again or can use self.tokenizer
            model_name=self.llm_model_name, # Use the same base model
            max_seq_length=1024,
            dtype=None, # Or specify a preferred dtype for inference, e.g., torch.float16
            load_in_4bit=True, # Or False if memory allows and precision is critical for embeddings
        )
        # DO NOT apply PEFT to self.embedding_model
        self.embedding_model.eval()
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        print("Frozen embedding model loaded and set to eval mode with gradients disabled.")

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

    def _get_llm_embeddings(
        self,
        texts: list[str] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None, # Should be provided if inputs_embeds might be padded or need specific attention
        device: torch.device = torch.device(DEVICE)
    ) -> torch.Tensor:
        # This method now uses self.embedding_model which is always frozen and in eval mode.
        # No need to manage self.model.training state here.
        
        # Determine max_length for tokenizer
        tokenizer_max_length = self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length else self.embedding_model.config.max_seq_length

        if texts is not None:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer_max_length,
            ).to(device)
            
            # Use the frozen embedding_model
            outputs = self.embedding_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )
            last_hidden_states = outputs.hidden_states[-1]
            
            input_mask_expanded = inputs.attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_hidden_states = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_embeddings = sum_hidden_states / sum_mask
            return pooled_embeddings

        elif inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1], device=device, dtype=torch.long)
            
            attention_mask = attention_mask.to(device)

            # Use the frozen embedding_model
            outputs = self.embedding_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden_states = outputs.hidden_states[-1]

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_hidden_states = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_embeddings = sum_hidden_states / sum_mask
            return pooled_embeddings
        else:
            raise ValueError("Either texts or inputs_embeds must be provided to _get_llm_embeddings.")

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
            batch_size = prompt_input_ids.shape[0]
            
            # Convert initial prompt IDs to embeddings
            current_input_embeds = self.model.get_input_embeddings()(prompt_input_ids)
            
            emb_matrix = self.model.get_input_embeddings().weight
            list_generated_soft_embs = [] # These will be the direct output fed to the decoder
            list_generated_hard_ids = []  # These are for text for loss_sim
            num_tokens_to_generate = dynamic_generation_length

            accumulated_token_penalty = torch.tensor(0.0, device=device, dtype=torch.float32)

            for _ in range(num_tokens_to_generate):
                # Model takes embeddings directly now
                model_outputs = self.model(inputs_embeds=current_input_embeds, attention_mask=current_attention_mask)
                next_token_logits = model_outputs.logits[:, -1, :]

                # Calculate penalty for "0" and "1" logits for this step
                probs = F.softmax(next_token_logits, dim=-1)
                # Ensure zero_id and one_id are not out of bounds for vocab
                # This assumes self.zero_id and self.one_id are valid scalar token indices
                prob_zero = probs[:, self.zero_id] if self.zero_id < probs.shape[-1] else torch.zeros_like(probs[:, 0])
                prob_one = probs[:, self.one_id] if self.one_id < probs.shape[-1] else torch.zeros_like(probs[:, 0])
                
                step_penalty = (prob_zero.sum() + prob_one.sum()) / batch_size # Average over batch
                accumulated_token_penalty += step_penalty

                # Soft distribution for soft embeddings (used for loss_bit and next input embed)
                soft_dist_for_next_step = F.gumbel_softmax(next_token_logits, tau=self.temperature, hard=False, dim=-1)
                # Create the soft embedding for the current step
                next_soft_emb_for_output_and_input = torch.matmul(soft_dist_for_next_step.unsqueeze(1).to(emb_matrix.dtype), emb_matrix)
                
                list_generated_soft_embs.append(next_soft_emb_for_output_and_input)
                
                ste_one_hot_output = F.gumbel_softmax(next_token_logits, tau=self.temperature, hard=True, dim=-1)
                hard_id_for_text_gen = torch.argmax(ste_one_hot_output, dim=-1).unsqueeze(1)
                
                list_generated_hard_ids.append(hard_id_for_text_gen)
                
                # Update inputs for the next iteration using the generated soft embedding
                current_input_embeds = torch.cat([current_input_embeds, next_soft_emb_for_output_and_input], dim=1)
                # Update attention mask
                current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(hard_id_for_text_gen, device=device)], dim=1) # Shape of hard_id is (batch, 1) like attention extension

            final_soft_embs = torch.cat(list_generated_soft_embs, dim=1)
            final_hard_ids = torch.cat(list_generated_hard_ids, dim=1)
            intermediate_encoded_texts = self.tokenizer.batch_decode(final_hard_ids, skip_special_tokens=True)
            generated_attention_mask = torch.ones_like(final_hard_ids, device=device)
            
            avg_token_penalty = accumulated_token_penalty / num_tokens_to_generate if num_tokens_to_generate > 0 else torch.tensor(0.0, device=device)
            
            return final_soft_embs, intermediate_encoded_texts, generated_attention_mask, avg_token_penalty
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

    def compute_loss(self, buffers, bits, device, lambda_sim: float, lambda_penalty: float):
        current_bits_list = bits.tolist()
        soft_embs, intermediate_encoded_texts, enc_attention_mask, loss_penalty_from_logits = \
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

        # Get embeddings for original buffers using the LLM
        orig_emb = self._get_llm_embeddings(texts=buffers, device=device)
        
        # Get embeddings for generated soft_embs using the LLM
        # enc_attention_mask corresponds to soft_embs and indicates valid (non-padded) tokens
        enc_emb = self._get_llm_embeddings(inputs_embeds=soft_embs, attention_mask=enc_attention_mask, device=device)
        
        sim_scores = util.cos_sim(orig_emb, enc_emb).diag()
        loss_sim = (1 - sim_scores).mean()

        # Use the penalty from logits directly
        loss_penalty = loss_penalty_from_logits.to(loss_bit.device)

        total_loss = loss_bit + lambda_sim * loss_sim.to(loss_bit.device) + lambda_penalty * loss_penalty
        
        return total_loss, loss_bit, loss_sim, loss_penalty, intermediate_encoded_texts
        
    def save_model(self):
        if not self.debug:
            final_save_path = f"{self.model_save_path_prefix}_final_pytorch_unsloth"
            print(f"Attempting to save Unsloth model to {final_save_path}...")
            self.model.save_pretrained(final_save_path)
            self.tokenizer.save_pretrained(final_save_path)
            print(f"Unsloth Model and Tokenizer saved to {final_save_path}")

