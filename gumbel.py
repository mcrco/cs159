import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers.optimization import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, util
import wandb
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("prompt.txt", "r") as f:
    INSTRUCTION = f.read()


class GumbelSteganographer:
    def __init__(
        self, llm_model_name, sim_model_name, lora_config, temperature, lambda_sim, debug=False, optimizer_args=None, scheduler_args=None
    ) -> None:
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = llm_model_name,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
            device_map = "auto",
        )
        if 'llama' in llm_model_name:
            self.tokenizer = get_chat_template(self.tokenizer, "llama-3.1")
        if 'qwen' in llm_model_name:
            self.tokenizer = get_chat_template(self.tokenizer, "qwen-2.5")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model, self.toknizer = FastLanguageModel.get_peft_model(
            self.model,
            **lora_config,
        )
        self.sim_model = SentenceTransformer(sim_model_name)

        self.temperature = float(temperature)
        self.lambda_sim = float(lambda_sim)
        self.debug = debug

        self.optimizer_args = optimizer_args if optimizer_args else {'lr': 1e-4}
        self.scheduler_args = scheduler_args if scheduler_args else {'num_warmup_steps': 500}

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

    def train(self, train_loader, val_loader, num_epochs, model_save_path_prefix="./stego_lora"):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)
        
        total_steps = len(train_loader) * num_epochs
        current_scheduler_args = self.scheduler_args.copy()
        if 'num_training_steps' not in current_scheduler_args:
            current_scheduler_args['num_training_steps'] = total_steps
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, **current_scheduler_args
        )

        zero_id = self.tokenizer.convert_tokens_to_ids("0")
        one_id = self.tokenizer.convert_tokens_to_ids("1")
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for buffers, bits in tqdm(train_loader, desc=f"Epoch {epoch}"):
                global_step += 1
                # --- Encode prompts ---
                list_of_encode_conversations = [
                    [{"role": "user", "content": f'{INSTRUCTION}\\n[ENCODE]\\nBuffer: "{b}"\\nHide bit: {bit}\\n'}]
                    for b, bit in zip(buffers, bits.tolist())
                ]
                enc_inputs_dict = self.tokenizer.apply_chat_template(
                    list_of_encode_conversations,
                    padding=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=True
                ).to(DEVICE)
                logits_enc = self.model(**enc_inputs_dict).logits  # [B, L_enc, V]

                # Sample soft distributions for ALL positions
                soft_dists = self.gumbel_softmax_sample(logits_enc)  # [B, L_enc, V]
                emb_matrix = self.model.get_input_embeddings().weight  # [V, D]
                # Convert to soft embeddings per position
                soft_embs = torch.matmul(
                    soft_dists.to(emb_matrix.dtype), emb_matrix
                )  # [B, L_enc, D]

                # Hard decode for logging and prompts
                hard_ids = torch.argmax(soft_dists, dim=-1)  # [B, L_enc]
                dec_texts = self.tokenizer.batch_decode(
                    hard_ids, skip_special_tokens=True
                )

                # --- Decode prompts for classification ---
                # dec_texts are the hard-decoded outputs from the encoder stage
                list_of_decode_conversations = [
                    [{"role": "user", "content": f'{INSTRUCTION}\\n[DECODE]\\nEncoded: "{t}"\\nHide bit:\\n'}] # Ensure prompt ends correctly
                    for t in dec_texts
                ]
                dec_inputs_dict = self.tokenizer.apply_chat_template(
                    list_of_decode_conversations,
                    padding=True,
                    return_tensors="pt",
                    add_generation_prompt=True, # Important for the model to expect to generate/predict
                    enable_thinking=False,
                    return_dict=True
                ).to(DEVICE)

                # Access input_ids using dictionary key
                prompt_embeds = self.model.get_input_embeddings()(
                    dec_inputs_dict['input_ids']
                )  # [B, L_dec, D]
                # Attach entire soft_embs sequence
                full_embeds = torch.cat(
                    [prompt_embeds, soft_embs], dim=1
                )  # [B, L_dec+L_enc, D]

                # Construct the combined attention mask
                combined_attention_mask = torch.cat(
                    [dec_inputs_dict['attention_mask'], enc_inputs_dict['attention_mask']], dim=1
                )

                # Pass inputs_embeds and the combined attention_mask
                # Ensure full_embeds has the same dtype as the model's computation dtype
                logits_dec = self.model(
                    inputs_embeds=full_embeds.to(self.model.dtype),
                    attention_mask=combined_attention_mask
                ).logits  # [B, L, V]

                # Extract logits for the very last token position using the combined mask info (sum along dim=1 gives sequence lengths)
                # This part seems overly complex and potentially incorrect for getting the single bit logit.
                # Let's simplify: The bit prediction should correspond to the FIRST token generated AFTER the decode prompt.
                # The prompt ends with "Hide bit:\n", so we want the logits for the token immediately after that.
                # The position should be the sequence length of the *decoder prompt* part.
                sequence_lengths = dec_inputs_dict['attention_mask'].sum(dim=1) # Lengths of decoder prompts [B]
                # Gather the logits at the end of each decoder prompt sequence
                # Use gather index: shape [B, 1, 2] for [zero_id, one_id]
                gather_index = sequence_lengths.view(-1, 1, 1).expand(-1, 1, 2)
                # We need logits from the position *before* the gather index corresponds to the *last* token *of the prompt*
                # So we need the logits at index sequence_lengths - 1
                bit_logits = torch.gather(logits_dec[:, :, [zero_id, one_id]], 1, gather_index - 1).squeeze(1)
                # OLD way: bit_logits = logits_dec[:, -1, [zero_id, one_id]] # This assumed the bit was always the very last token overall

                loss_bit = torch.nn.functional.cross_entropy(
                    bit_logits, bits.to(DEVICE)
                )

                # --- Generate 1 token sample for logging ---
                # Ensure generate call uses dictionary unpacking or correct keywords
                generated = self.model.generate(
                    input_ids=dec_inputs_dict['input_ids'], # Use key access
                    attention_mask=dec_inputs_dict['attention_mask'], # Use key access
                    max_new_tokens=1, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=self.tokenizer.eos_token_id
                )
                gen_token = generated[:, dec_inputs_dict['input_ids'].shape[1] :] # Use key access
                gen_texts = self.tokenizer.batch_decode(
                    gen_token, skip_special_tokens=True
                )

                # Semantic similarity
                orig_emb = self.sim_model.encode(
                    buffers, convert_to_tensor=True, device=DEVICE
                )
                enc_emb = self.sim_model.encode(
                    gen_texts, convert_to_tensor=True, device=DEVICE
                )
                sim_scores = util.cos_sim(orig_emb, enc_emb).diag()
                loss_sim = (1 - sim_scores).mean()

                # Backprop
                loss = loss_bit + self.lambda_sim * loss_sim
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                running_loss += loss.item()

                if self.debug and global_step % 10 == 0: # Print every 10 steps or adjust as needed
                    print(f"--- Training Step {global_step} (Batch Item 0) ---")
                    print(f'Buffer: "{buffers[0]}"')
                    # dec_texts[0] is the hard Gumbel-Softmax decoding of the encoder's output logits for the input sequence.
                    # This is what's used in the DECODE prompt as "Encoded: ..." during this training step.
                    print(f'Encoded (for DECODE): "{dec_texts[0]}"')
                    print(f"Target Bit: {bits[0].item()}")
                    print(f"Predicted Bit (by DECODE part): {gen_texts[0]}\n")

                # Step logs
                if not self.debug:
                    wandb.log(
                        {
                            "step_loss": loss.item(),
                            "bit_loss": loss_bit.item(),
                            "sim_loss": loss_sim.item(),
                        },
                        step=global_step,
                    )

            # Epoch logs
            avg_loss = running_loss / len(train_loader)
            if not self.debug:
                wandb.log({"train/avg_loss": avg_loss, "epoch": epoch}, step=global_step)
            print(f"Epoch {epoch} avg_loss {avg_loss:.4f}")
            self.validate(val_loader, epoch)

        if not self.debug:
            final_save_path = f"{model_save_path_prefix}_final"
            self.model.save_pretrained(final_save_path)
            self.tokenizer.save_pretrained(final_save_path)
            print(f"Model saved to {final_save_path}")

    def validate(self, val_loader, epoch):
        if not self.debug:
            table = wandb.Table(columns=["buffer", "encoded_sentence", "decoded_bit"])
        else:
            print(f"--- Validation Epoch {epoch} ---")

        zero_id = self.tokenizer.convert_tokens_to_ids("0")
        one_id = self.tokenizer.convert_tokens_to_ids("1")

        for buffers, bits in tqdm(val_loader, desc=f"Epoch {epoch}"):
            self.model.eval()
            for buf in buffers:
                # Encode
                encode_prompt_content = f'{INSTRUCTION}\\n[ENCODE] Buffer: "{buf}"\\nHide bit: 1\\n' # Assume bit 1 for validation example
                encode_messages = [{"role": "user", "content": encode_prompt_content}]
                encode_input_ids = self.tokenizer.apply_chat_template(
                    encode_messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
                ).to(DEVICE)
                # Removed attention_mask=None as generate usually handles it

                generated_ids = self.model.generate(
                    input_ids=encode_input_ids,
                    max_new_tokens=20, # Keep original validation max_new_tokens
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.eos_token_id # Add pad_token_id for consistency
                )
                # Slice off prompt using input_ids length from chat template
                encoded_text = self.tokenizer.decode(
                    generated_ids[0][encode_input_ids.shape[1] :], skip_special_tokens=True
                ).strip() # Add strip() for cleaner output

                # Decode
                decode_prompt_content = f'{INSTRUCTION}\\n[DECODE] Encoded: "{encoded_text}"\\nHide bit:\\n'
                decode_messages = [{"role": "user", "content": decode_prompt_content}]
                decode_input_ids = self.tokenizer.apply_chat_template(
                    decode_messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
                ).to(DEVICE)
                # Removed attention_mask=None

                # Generate only the bit token
                decoded_ids = self.model.generate(
                    input_ids=decode_input_ids,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id # Add pad_token_id
                )
                # Get the generated token (the predicted bit) by slicing based on input_ids length
                predicted_bit_text = self.tokenizer.decode(
                    decoded_ids[0][decode_input_ids.shape[1]:], skip_special_tokens=True
                ).strip() # Add strip()

                # Determine predicted bit based on the decoded text
                predicted_bit = "UNK" # Default to UNK
                if predicted_bit_text == "0":
                    predicted_bit = "0"
                elif predicted_bit_text == "1":
                    predicted_bit = "1"

                # Logging remains the same
                if not self.debug:
                    table.add_data(buf, encoded_text, predicted_bit)
                else:
                    print(f'Buffer: "{buf}"')
                    print(f'Encoded: "{encoded_text}"')
                    print(f"Decoded bit: {predicted_bit}\n")
        
        if not self.debug:
            wandb.log({f"epoch_{epoch}_generation": table})