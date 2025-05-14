import os

if os.environ.get("NO_UNSLOTH", "false") == "false":
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template as unsloth_get_chat_template
else:
    print("unsloth is disabled")

import torch
from transformers.optimization import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, util
import wandb
from tqdm import tqdm

# Standard Hugging Face imports
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model as hf_get_peft_model # Renamed to avoid conflict if PeftModel is also imported directly

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("prompt.txt", "r") as f:
    INSTRUCTION = f.read()

class GumbelSteganographer:
    def __init__(
        self, 
        llm_model_name, 
        sim_model_name, 
        lora_config, 
        temperature, 
        lambda_sim, 
        debug=False, 
        optimizer_args=None, 
        scheduler_args=None,
        use_unsloth: bool = True # New parameter
    ) -> None:
        self.use_unsloth = use_unsloth
        self.model_name_for_chat_template = llm_model_name # Store original name for chat template logic

        if self.use_unsloth:
            print("Using Unsloth for model loading.")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=llm_model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                device_map="auto",
            )
            # Apply Unsloth PEFT
            self.model = FastLanguageModel.get_peft_model(self.model, **lora_config)
        else:
            print("Using standard Hugging Face Transformers for model loading.")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=False, # Often False for nf4 but can be True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                # trust_remote_code=True # Sometimes needed for specific models
            )
            # Apply standard PEFT
            self.model = hf_get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters() # Good practice after applying PEFT

        # Chat template application (conditional for Unsloth)
        # For non-Unsloth, rely on tokenizer's default or assume it's correctly configured.
        # More sophisticated handling might involve mapping model names to HF chat template application methods if needed.
        if self.use_unsloth:
            if 'llama' in self.model_name_for_chat_template.lower():
                self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="llama-3.1")
            elif 'qwen' in self.model_name_for_chat_template.lower():
                self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="qwen-2.5")
            elif 'gemma' in self.model_name_for_chat_template.lower():
                self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="gemma-3")
            elif 'olmo' in self.model_name_for_chat_template.lower():
                self.tokenizer = unsloth_get_chat_template(self.tokenizer, chat_template="olmo")
        else:
            # For standard HF, if specific chat template logic is needed beyond tokenizer default, add here.
            # e.g., self.tokenizer.chat_template = "... specific template string ..." if necessary.
            print("Relying on tokenizer's default chat template or pre-configuration for non-Unsloth mode.")

        if self.tokenizer.pad_token is None:
            print("Tokenizer does not have a pad_token, setting it to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Ensure model's pad_token_id is also set, crucial for generation
            if self.model.config.pad_token_id is None:
                 self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.sim_model = SentenceTransformer(sim_model_name, device=DEVICE)
        self.temperature = float(temperature)
        self.lambda_sim = float(lambda_sim)
        self.debug = debug

        self.optimizer_args = optimizer_args if optimizer_args else {'lr': 1e-4}
        self.scheduler_args = scheduler_args if scheduler_args else {'num_warmup_steps': 500}
        
        self.zero_id = self.tokenizer.convert_tokens_to_ids("0")
        self.one_id = self.tokenizer.convert_tokens_to_ids("1")
        # Ensure they are single IDs if tokenizer returns lists
        if isinstance(self.zero_id, list): 
            self.zero_id = self.zero_id[0]
        if isinstance(self.one_id, list): 
            self.one_id = self.one_id[0]

        self.default_generation_params = {
            "do_sample": False, "temperature": None, "top_p": None, "top_k": None, 
            "pad_token_id": self.tokenizer.pad_token_id # Use actual pad_token_id from tokenizer
        }
        self.generation_length_margin = 10

    @staticmethod
    def sample_gumbel(shape, device, eps=1e-20):
        """Sample Gumbel noise"""
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        """Perform differentiable Gumbel-Softmax sampling on logits"""
        gumbel_noise = self.sample_gumbel(logits.size(), logits.device)
        y = torch.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
        return y

    def _prepare_encode_prompts(self, buffers: list[str], bits_to_hide: list[int]) -> list[list[dict]]:
        return [
            [{"role": "user", "content": f'{INSTRUCTION}\\n[ENCODE]\\nBuffer: {b}\\nHide bit: {bit}\\n'}]
            for b, bit in zip(buffers, bits_to_hide)
        ]

    def _prepare_decode_prompts(self, encoded_texts: list[str]) -> list[list[dict]]:
        # This prompt structure (ending with \nEncoded:"{text}"\n) is used for both 
        # training (bit_logits extraction) and inference (generating the bit).
        # add_generation_prompt=True in apply_chat_template prepares the model to generate after this.
        return [
            [{"role": "user", "content": f'{INSTRUCTION}\\n[DECODE]\\nEncoded: {text}\\n'}]
            for text in encoded_texts
        ]

    def generate_encoded_text_or_embeddings(
        self, 
        buffers: str | list[str], 
        bits_to_hide: int | list[int], 
        produce_embeddings: bool,
        generation_max_new_tokens: int | None = None 
    ):
        was_single_input = isinstance(buffers, str)
        
        _buffers = [buffers] if was_single_input else buffers
        
        if isinstance(bits_to_hide, int):
            _bits_to_hide = [bits_to_hide] * len(_buffers)
        elif bits_to_hide is not None and len(bits_to_hide) != len(_buffers):
            raise ValueError("Buffers and bits_to_hide must have the same number of elements if bits_to_hide is a list.")
        else:
            _bits_to_hide = bits_to_hide

        # Calculate dynamic generation length based on buffer contents
        if not _buffers:
            # Fallback if somehow _buffers is empty, though input validation should prevent this.
            # Using a small fixed number or generation_length_margin itself.
            max_buffer_token_count = 0
        else:
            buffer_token_lengths = [len(self.tokenizer.tokenize(b)) for b in _buffers]
            max_buffer_token_count = max(buffer_token_lengths) if buffer_token_lengths else 0
        
        dynamic_generation_length = max(5, max_buffer_token_count + self.generation_length_margin) # Ensure min 5 tokens

        list_of_encode_conversations = self._prepare_encode_prompts(_buffers, _bits_to_hide)
        
        enc_inputs_dict = self.tokenizer.apply_chat_template(
            list_of_encode_conversations,
            padding=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False, return_dict=True
        ).to(self.model.device)

        if produce_embeddings:
            # Perform auto-regressive generation with Gumbel-Softmax
            # batch_size = enc_inputs_dict['input_ids'].size(0)
            current_input_ids = enc_inputs_dict['input_ids']
            current_attention_mask = enc_inputs_dict['attention_mask']
            
            emb_matrix = self.model.get_input_embeddings().weight
            
            list_generated_soft_embs = []
            list_generated_hard_ids = []
            
            # Use a fixed length for generated sequence in training, e.g., default_val_encode_max_new_tokens
            num_tokens_to_generate = dynamic_generation_length

            for _ in range(num_tokens_to_generate):
                model_outputs = self.model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                next_token_logits = model_outputs.logits[:, -1, :]  # Logits for the next token

                soft_dist = self.gumbel_softmax_sample(next_token_logits) # [B, V]
                
                # Soft embedding for this step
                # Unsqueeze soft_dist to [B, 1, V] for matmul with emb_matrix [V, D] -> [B, 1, D]
                current_soft_emb = torch.matmul(soft_dist.unsqueeze(1).to(emb_matrix.dtype), emb_matrix) # [B, 1, D]
                list_generated_soft_embs.append(current_soft_emb)
                
                # Hard token for this step (to be fed into the next step of generation)
                hard_id = torch.argmax(soft_dist, dim=-1).unsqueeze(1)  # [B, 1]
                list_generated_hard_ids.append(hard_id)
                
                # Append the generated hard_id for the next iteration
                current_input_ids = torch.cat([current_input_ids, hard_id], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(hard_id)], dim=1)

            # Concatenate collected tensors
            # final_soft_embs: [B, num_tokens_to_generate, D]
            final_soft_embs = torch.cat(list_generated_soft_embs, dim=1) 
            # final_hard_ids: [B, num_tokens_to_generate]
            final_hard_ids = torch.cat(list_generated_hard_ids, dim=1)
            
            intermediate_encoded_texts = self.tokenizer.batch_decode(final_hard_ids, skip_special_tokens=True)
            
            # The attention mask for these generated soft embeddings
            # Shape: [B, num_tokens_to_generate]
            generated_attention_mask = torch.ones_like(final_hard_ids, device=self.model.device)
            
            return final_soft_embs, intermediate_encoded_texts, generated_attention_mask
        else:
            # Inference path remains the same
            if generation_max_new_tokens is not None:
                max_tokens = generation_max_new_tokens
            else:
                max_tokens = dynamic_generation_length
            generated_ids = self.model.generate(
                input_ids=enc_inputs_dict['input_ids'],
                attention_mask=enc_inputs_dict['attention_mask'],
                max_new_tokens=max_tokens,
                **self.default_generation_params
            )
            encoded_texts_generated = self.tokenizer.batch_decode(
                generated_ids[:, enc_inputs_dict['input_ids'].shape[1]:], skip_special_tokens=True
            )
            stripped_texts = [text.strip() for text in encoded_texts_generated]
            return stripped_texts[0] if was_single_input else stripped_texts

    def predict_bits_from_encoded_text(
        self, 
        encoded_texts: str | list[str], 
        soft_embs_from_encoder: torch.Tensor | None = None, 
        attention_mask_for_soft_embs: torch.Tensor | None = None
    ):
        was_single_input = isinstance(encoded_texts, str)
        _encoded_texts = [encoded_texts] if was_single_input else encoded_texts

        list_of_decode_conversations = self._prepare_decode_prompts(_encoded_texts)
        
        dec_inputs_dict = self.tokenizer.apply_chat_template(
            list_of_decode_conversations,
            padding=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False, return_dict=True
        ).to(self.model.device)

        if soft_embs_from_encoder is not None: # Training path, returns bit_logits
            if attention_mask_for_soft_embs is None:
                raise ValueError("attention_mask_for_soft_embs must be provided if soft_embs_from_encoder is given.")

            prompt_embeds = self.model.get_input_embeddings()(dec_inputs_dict['input_ids'])
            _soft_embs = soft_embs_from_encoder.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            _attn_mask_soft_embs = attention_mask_for_soft_embs.to(device=prompt_embeds.device)

            full_embeds = torch.cat([prompt_embeds, _soft_embs], dim=1)
            combined_attention_mask = torch.cat([dec_inputs_dict['attention_mask'], _attn_mask_soft_embs], dim=1)
            
            logits_dec = self.model(
                inputs_embeds=full_embeds,
                attention_mask=combined_attention_mask
            ).logits

            sequence_lengths = dec_inputs_dict['attention_mask'].sum(dim=1) # Lengths of decoder prompts [B]
            # Logits for the token to be generated immediately after the decoder prompt.
            # sequence_lengths are 1-based. Index is sequence_lengths - 1 for the last token of the prompt.
            # The model's output at this position (logits_dec[b, seq_len-1, :]) is used to predict the *next* token.
            batch_indices = torch.arange(logits_dec.size(0), device=logits_dec.device)
            last_prompt_token_logits = logits_dec[batch_indices, sequence_lengths - 1]
            bit_logits = last_prompt_token_logits[:, [self.zero_id, self.one_id]]
            
            # Returns batched tensor even for single input if soft_embs provided
            return bit_logits
        else: # Inference/validation path, returns "0" or "1" text
            decoded_ids = self.model.generate(
                input_ids=dec_inputs_dict['input_ids'],
                attention_mask=dec_inputs_dict['attention_mask'],
                max_new_tokens=1,
                **self.default_generation_params
            )
            predicted_bit_texts = self.tokenizer.batch_decode(
                decoded_ids[:, dec_inputs_dict['input_ids'].shape[1]:], skip_special_tokens=True
            )
            stripped_texts = [text.strip() for text in predicted_bit_texts]
            return stripped_texts[0] if was_single_input else stripped_texts

    def train(self, train_loader, val_loader, num_epochs, model_save_path_prefix="./stego_lora"):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)
        
        total_steps = len(train_loader) * num_epochs
        current_scheduler_args = self.scheduler_args.copy()
        if 'num_training_steps' not in current_scheduler_args:
            current_scheduler_args['num_training_steps'] = total_steps
        
        scheduler = get_linear_schedule_with_warmup(optimizer, **current_scheduler_args)
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for buffers, bits in tqdm(train_loader, desc=f"Epoch {epoch}"):
                global_step += 1
                optimizer.zero_grad()

                # --- Encode to get soft embeddings and intermediate texts ---
                # buffers is List[str], bits is Tensor. Convert bits to list for the function.
                current_bits_list = bits.tolist()
                soft_embs, intermediate_encoded_texts, enc_attention_mask = \
                    self.generate_encoded_text_or_embeddings(
                        buffers, current_bits_list, produce_embeddings=True
                )

                # --- Decode using soft embeddings to get bit logits ---
                bit_logits = self.predict_bits_from_encoded_text(
                    intermediate_encoded_texts, # These are the "encoded" texts for the DECODE prompt
                    soft_embs_from_encoder=soft_embs,
                    attention_mask_for_soft_embs=enc_attention_mask
                )
                
                loss_bit = torch.nn.functional.cross_entropy(bit_logits, bits.to(self.model.device))

                # --- Semantic similarity loss ---
                # Using intermediate_encoded_texts (hard Gumbel output) for similarity
                orig_emb = self.sim_model.encode(buffers, convert_to_tensor=True) # Already on DEVICE via sim_model init
                enc_emb = self.sim_model.encode(intermediate_encoded_texts, convert_to_tensor=True)
                
                # Ensure embeddings are on the same device for cos_sim if not already handled
                # Sim_model was initialized to DEVICE, so its output should be too.
                sim_scores = util.cos_sim(orig_emb, enc_emb).diag()
                loss_sim = (1 - sim_scores).mean() # Ensure this is on model.device for loss accumulation

                loss = loss_bit + self.lambda_sim * loss_sim.to(loss_bit.device) # Ensure loss_sim is on same device
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

                # --- Logging ---
                if self.debug and global_step % 10 == 0:
                    # Get predicted bit text for logging (inference path of predict_bits)
                    predicted_bits_texts_log = self.predict_bits_from_encoded_text(intermediate_encoded_texts)
                    
                    print(f"--- Training Step {global_step} (Batch Item 0) ---")
                    print(f'Buffer: {buffers[0]}')
                    print(f'Encoded (for DECODE): {intermediate_encoded_texts[0]}')
                    print(f"Target Bit: {current_bits_list[0]}")
                    # predicted_bits_texts_log is a list, take first item
                    print(f"Predicted Bit (by DECODE part): {predicted_bits_texts_log[0] if isinstance(predicted_bits_texts_log, list) else predicted_bits_texts_log}\n")


                if not self.debug:
                    wandb.log(
                        {
                            "step_loss": loss.item(),
                            "bit_loss": loss_bit.item(),
                            "sim_loss": loss_sim.item(),
                        },
                        step=global_step,
                    )

            avg_loss = running_loss / len(train_loader)
            if not self.debug:
                wandb.log({"train/avg_loss": avg_loss, "epoch": epoch}, step=global_step)
            print(f"Epoch {epoch} avg_loss {avg_loss:.4f}")
            self.validate(val_loader, epoch)

        if not self.debug:
            final_save_path = f"{model_save_path_prefix}_final"
            self.model.save_pretrained(final_save_path) # Unsloth PEFT model save
            self.tokenizer.save_pretrained(final_save_path)
            print(f"Model saved to {final_save_path}")

    def validate(self, val_loader, epoch):
        if not self.debug:
            table = wandb.Table(columns=["buffer", "encoded_sentence", "decoded_bit", "target_bit"])
        else:
            print(f"--- Validation Epoch {epoch} ---")

        self.model.eval() # Set model to evaluation mode

        for buffers, bits in tqdm(val_loader, desc=f"Validating Epoch {epoch}"):
            # Iterate through batch items for individual processing, as validate often shows examples
            for i in range(len(buffers)):
                buf = buffers[i]
                target_bit_val = bits[i].item() # Target bit for this item

                # Encode (using inference path of generate_encoded_text_or_embeddings)
                # Bit to hide for validation encoding - can be fixed or use target_bit_val
                # Using target_bit_val to see how it encodes the specific target bit.
                encoded_text = self.generate_encoded_text_or_embeddings(
                    buf, 
                    target_bit_val, # Use actual bit for encoding during validation
                    produce_embeddings=False, 
                    generation_max_new_tokens=None # MODIFIED HERE: Allow dynamic calculation based on buffer
                )

                # Decode (using inference path of predict_bits_from_encoded_text)
                predicted_bit_text = self.predict_bits_from_encoded_text(encoded_text)

                # Determine predicted bit category
                processed_predicted_bit = "UNK"
                if predicted_bit_text == "0":
                    processed_predicted_bit = "0"
                elif predicted_bit_text == "1":
                    processed_predicted_bit = "1"

                if not self.debug:
                    table.add_data(buf, encoded_text, processed_predicted_bit, str(target_bit_val))
                else:
                    print(f'Buffer: "{buf}"')
                    print(f'Target Bit: {target_bit_val}')
                    print(f'Encoded: "{encoded_text}"')
                    print(f"Decoded bit: {processed_predicted_bit}\n")
        
        if not self.debug:
            wandb.log({f"val/epoch_{epoch}_examples": table}, epoch=epoch)