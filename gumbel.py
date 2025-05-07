import torch
from torch.utils.data import DataLoader
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer, util
import wandb
from datasets import load_from_disk
from tqdm import tqdm


# --------------------
# 1. Utilities for Gumbel-Softmax
# --------------------
def sample_gumbel(shape, device, eps=1e-20):
    """Sample Gumbel noise"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, tau=1.0):
    """Perform differentiable Gumbel-Softmax sampling on logits"""
    gumbel_noise = sample_gumbel(logits.size(), logits.device)
    y = torch.softmax((logits + gumbel_noise) / tau, dim=-1)
    return y


# --------------------
# 2. Initialize W&B
# --------------------
wandb.init(project="qwen-steganography", name="gumbel_lora")

# --------------------
# 3. Load model & tokenizer
# --------------------
MODEL_NAME = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", load_in_8bit=True, trust_remote_code=True
)

# --------------------
# 4. Apply LoRA adapters
# --------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)

# Move to device
device = "cuda"
model.to(device)

# --------------------
# 5. Similarity model
# --------------------
sim_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------
# 6. Prepare datasets & dataloaders
# --------------------
# Assume `train_dataset` & `val_dataset` exist as HF-datasets with fields "buffer_text" and "bit"
def collate_fn(batch):
    buffers = [item["buffer_text"] for item in batch]
    bits = torch.tensor([item["bit"] for item in batch], dtype=torch.long)
    return buffers, bits


dataset = load_from_disk("stego_dataset_wizard_of_oz/")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# if not isinstance(train_dataset, ad.Dataset):
#     raise Exception(f"train_dataset invalid type: {type(train_dataset)}")
# if not isinstance(val_dataset, ad.Dataset):
#     raise Exception(f"val_dataset invalid type: {type(val_dataset)}")


train_loader = DataLoader(
    train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# --------------------
# 7. Optimizer & Scheduler
# --------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = 3
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=total_steps
)

# --------------------
# 8. Training loop
# --------------------
TAU = 1.0  # Gumbel temperature
LAMBDA_SIM = 0.1  # weight for semantic similarity loss
zero_id = tokenizer.convert_tokens_to_ids("0")
one_id = tokenizer.convert_tokens_to_ids("1")
global_step = 0

INSTRUCTION = """
### 
You are a specialized steganography language model.
When you are given buffer text with [ENCODE] in front and a hidden bit, you will somehow modify the buffer text to encode that bit,
but you will not tell the user how. You simply modify the buffer text to make it work.
When you are given encoded text with [DECODE] in front, you will somehow predict the hidden bit you encoded,
once again without saying how you did it. You simply output a single bit.
###
"""

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    last_buffers = []
    for step, (buffers, bits) in tqdm(enumerate(train_loader)):
        last_buffers = buffers
        global_step += 1
        # --- Encode step ---
        prompts = [
            f'{INSTRUCTION}\n[ENCODE] Buffer: "{b}"\n' + f"Hide bit: {bit}\n→"
            for b, bit in zip(buffers, bits.tolist())
        ]
        enc_inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        logits_enc = model(**enc_inputs).logits
        last_logits = logits_enc[:, -1, :]
        soft_dists = gumbel_softmax_sample(last_logits, TAU)

        # --- Create soft embeddings for backprop ---
        emb_matrix = model.get_input_embeddings().weight
        soft_embs = soft_dists.to(emb_matrix.dtype) @ emb_matrix

        # --- Decode step: build hard tokens & texts ---
        hard_ids = torch.argmax(soft_dists, dim=-1)
        dec_texts = tokenizer.batch_decode(hard_ids, skip_special_tokens=True)

        # --- Decode step: craft prompts including encoded text ---
        dec_prompts = [
            f'{INSTRUCTION}\n[DECODE] Encoded: "{t}"\nHide bit: \n→' for t in dec_texts
        ]
        dec_inputs = tokenizer(
            dec_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Get embeddings and plug in the soft_embs token
        prompt_embeds = model.get_input_embeddings()(dec_inputs.input_ids)
        full_embeds = torch.cat([prompt_embeds, soft_embs.unsqueeze(1)], dim=1)

        # Run decoder and compute bit logits
        logits_dec = model(inputs_embeds=full_embeds).logits[:, -1, :]
        bit_logits = logits_dec[:, [zero_id, one_id]]
        loss_bit = torch.nn.functional.cross_entropy(bit_logits, bits.to(device))

        # --- Semantic similarity loss ---
        orig_emb = sim_model.encode(buffers, convert_to_tensor=True, device=device)
        enc_emb = sim_model.encode(dec_texts, convert_to_tensor=True, device=device)
        sim_scores = util.cos_sim(orig_emb, enc_emb).diag()
        loss_sim = (1 - sim_scores).mean()

        # --- Combine & backprop ---
        loss = loss_bit + LAMBDA_SIM * loss_sim
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        running_loss += loss.item()

        # --- Step-level logging ---
        log_data = {
            "step_loss": loss.item(),
            "step_loss_bit": loss_bit.item(),
            "step_loss_sim": loss_sim.item(),
        }
        wandb.log(log_data, step=global_step)

    # --- Epoch-level logging ---
    avg_loss = running_loss / len(train_loader)
    wandb.log({"train/avg_loss": avg_loss, "epoch": epoch}, step=global_step)

    # Log a couple of generation examples at the end of the epoch
    model.eval()
    example_buffers = last_buffers[:2]
    with torch.no_grad():
        gen_table = wandb.Table(columns=["buffer_text", "generated_text"])
        for buf in example_buffers:
            gen_prompt = INSTRUCTION + f'\n[ENCODE] Buffer: "{buf}"Hide bit: 1→'
            generated = model.generate(
                **tokenizer(gen_prompt, return_tensors="pt").to(device),
                max_new_tokens=20,
                do_sample=False,
            )
            text_out = tokenizer.decode(generated[0], skip_special_tokens=True)
            gen_table.add_data(buf, text_out)
        wandb.log({f"epoch_{epoch}_generation_examples": gen_table}, step=global_step)
    print(f"Epoch {epoch} → avg_loss: {avg_loss:.4f}")

# Save final model
model.save_pretrained("./qwen_stego_lora")
tokenizer.save_pretrained("./qwen_stego_lora")
