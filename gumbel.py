import torch
from torch.utils.data import DataLoader
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from peft import get_peft_model
from sentence_transformers import SentenceTransformer, util
import wandb
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUCTION = """
### 
You are a specialized steganography language model.

When you are given buffer text, a single sentence, with [ENCODE] in front and a hidden bit, you will somehow modify the buffer text to encode that bit.
However, you will not tell the user how. You simply modify the buffer text to make it work, and then write out the buffer text after the arrow. 
For example, if you are given the prompt

[ENCODE] 
Buffer: "{buffer text}" 
Hide bit: 1

You will output a modified version of just {buffer text}, still a sentence, that somehow has encoded the bit 1 into its meaning.

When you are given encoded text with [DECODE] in front, you will somehow predict the hidden bit you encoded.
Once again, you do not say how you did it. You simply output a single bit. 
For example, if you are given the prompt

[DECODE] Encoded: "{encoded text}"
Hide bit:

You will output 0 if you think the encoded sentence is encoding 0, and 1 if you think the encoded sentences is encoding 1.
###
"""


class GumbelSteganographer:
    def __init__(
        self, llm_model_name, sim_model_name, lora_config, temperature, lambda_sim, debug=False, optimizer_args=None, scheduler_args=None
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name, device_map="auto", load_in_8bit=True, trust_remote_code=True
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(DEVICE)
        self.sim_model = SentenceTransformer(sim_model_name, device=DEVICE)

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
        
        from transformers.optimization import get_linear_schedule_with_warmup
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
                prompts = [
                    f'{INSTRUCTION}\n[ENCODE]\nBuffer: "{b}"\nHide bit: {bit}\n'
                    for b, bit in zip(buffers, bits.tolist())
                ]
                enc_inputs = self.tokenizer(
                    prompts, padding=True, truncation=True, return_tensors="pt"
                ).to(DEVICE)
                logits_enc = self.model(**enc_inputs).logits  # [B, L_enc, V]

                soft_dists = self.gumbel_softmax_sample(logits_enc)  # [B, L_enc, V]
                emb_matrix = self.model.get_input_embeddings().weight  # [V, D]
                soft_embs = torch.matmul(
                    soft_dists.to(emb_matrix.dtype), emb_matrix
                )  # [B, L_enc, D]

                hard_ids = torch.argmax(soft_dists, dim=-1)  # [B, L_enc]
                decoded_sequences = self.tokenizer.batch_decode(
                    hard_ids, skip_special_tokens=True
                )
                
                dec_texts_for_decode_prompt = self.tokenizer.batch_decode(hard_ids, skip_special_tokens=False)
                
                clean_encoded_texts_for_decode_prompt = []

                for i in range(hard_ids.shape[0]):
                    current_prompt_tokens = enc_inputs.input_ids[i]
                    actual_prompt_len = (enc_inputs.attention_mask[i].sum()).item()
                    
                    full_decoded_text = self.tokenizer.decode(hard_ids[i], skip_special_tokens=False)
                    
                    clean_encoded_texts_for_decode_prompt.append(full_decoded_text)

                    original_prompt_str = prompts[i]
                    if full_decoded_text.startswith(original_prompt_str.rstrip()):
                        gen_output_for_sim_and_debug = full_decoded_text[len(original_prompt_str.rstrip()):].lstrip()
                    else:
                        generated_tokens_only = hard_ids[i][actual_prompt_len:]
                        gen_output_for_sim_and_debug = self.tokenizer.decode(generated_tokens_only, skip_special_tokens=True)
                    
                    generated_part_ids_list.append(gen_output_for_sim_and_debug)

                dec_texts = clean_encoded_texts_for_decode_prompt

                dec_prompts = [
                    f'{INSTRUCTION}\n[DECODE]\nEncoded: "{t}"\n'
                    for t in dec_texts
                ]
                dec_inputs = self.tokenizer(
                    dec_prompts, padding=True, truncation=True, return_tensors="pt"
                ).to(DEVICE)
                
                prompt_embeds = self.model.get_input_embeddings()(
                    dec_inputs.input_ids
                )  # [B, L_dec, D]
                
                full_embeds = torch.cat(
                    [prompt_embeds, soft_embs], dim=1
                )  # [B, L_dec+L_enc, D]

                logits_dec = self.model(inputs_embeds=full_embeds).logits  # [B, L_dec+L_enc, V]
                bit_logits = logits_dec[:, -1, [zero_id, one_id]]

                loss_bit = torch.nn.functional.cross_entropy(
                    bit_logits, bits.to(DEVICE)
                )

                generated_bit_prediction = self.model.generate(
                    input_ids=dec_inputs.input_ids,
                    max_new_tokens=1, 
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                gen_token = generated_bit_prediction[:, dec_inputs.input_ids.shape[1] :]
                predicted_bit_texts = self.tokenizer.batch_decode(
                    gen_token, skip_special_tokens=True
                )

                orig_emb = self.sim_model.encode(
                    buffers, convert_to_tensor=True, device=DEVICE
                )
                enc_emb = self.sim_model.encode(
                    generated_part_ids_list, convert_to_tensor=True, device=DEVICE
                )
                sim_scores = util.cos_sim(orig_emb, enc_emb).diag()
                loss_sim = (1 - sim_scores).mean()

                loss = loss_bit + self.lambda_sim * loss_sim
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                running_loss += loss.item()

                if self.debug and global_step % 10 == 0: 
                    print(f"--- Training Step {global_step} (Batch Item 0) ---")
                    print(f'Buffer: "{buffers[0]}"')
                    print(f'Encoded (Gumbel output for sim): "{generated_part_ids_list[0]}"')
                    print(f'Encoded (in DECODE prompt): "{dec_texts[0]}"')
                    print(f"Target Bit: {bits[0].item()}")
                    print(f"Predicted Bit (by DECODE part): {predicted_bit_texts[0]}\n")

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
            self.validate(val_loader, epoch, global_step)

        if not self.debug:
            final_save_path = f"{model_save_path_prefix}_final"
            self.model.save_pretrained(final_save_path)
            self.tokenizer.save_pretrained(final_save_path)
            print(f"Model saved to {final_save_path}")

    def validate(self, val_loader, epoch, current_global_step=None):
        if not self.debug:
            table = wandb.Table(columns=["buffer", "encoded_sentence", "decoded_bit"])
        else:
            print(f"--- Validation Epoch {epoch} ---")

        zero_id = self.tokenizer.convert_tokens_to_ids("0")
        one_id = self.tokenizer.convert_tokens_to_ids("1")

        self.model.eval()
        with torch.no_grad():
            for buffers, bits in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                for buf in buffers:
                    encode_prompt_template = f'{INSTRUCTION}\n[ENCODE]\nBuffer: "{{buffer}}"\nHide bit: 1\n'
                    encode_prompt = encode_prompt_template.format(buffer=buf)
                    encode_in_ids = self.tokenizer(encode_prompt, return_tensors="pt").input_ids.to(DEVICE)
                    
                    max_len_buffer = len(self.tokenizer.tokenize(buf))
                    generated_ids = self.model.generate(
                        input_ids=encode_in_ids, 
                        max_new_tokens=max_len_buffer + 30,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    encoded_text = self.tokenizer.decode(
                        generated_ids[0][encode_in_ids.shape[1]:], skip_special_tokens=True
                    ).strip()

                    decode_prompt = f'{INSTRUCTION}\n[DECODE]\nEncoded: "{encoded_text}"\n'
                    decode_in_ids = self.tokenizer(decode_prompt, return_tensors="pt").input_ids.to(DEVICE)
                    
                    decoded_bit_ids = self.model.generate(
                        input_ids=decode_in_ids, 
                        max_new_tokens=1, 
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    predicted_bit_token_id = decoded_bit_ids[0][-1].item()
                    predicted_bit = "0" if predicted_bit_token_id == zero_id else "1" if predicted_bit_token_id == one_id else "UNK"

                    if not self.debug:
                        table.add_data(buf, encoded_text, predicted_bit)
                    else:
                        print(f'Buffer: "{buf}"')
                        print(f'Encoded: "{encoded_text}"')
                        print(f"Decoded bit: {predicted_bit}\n")
        
        if not self.debug and current_global_step is not None:
            wandb.log({f"validation/epoch_{epoch}_generation_samples": table}, step=current_global_step)
        elif not self.debug:
             wandb.log({f"validation/epoch_{epoch}_generation_samples": table})
        self.model.train()
