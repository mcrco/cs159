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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

train_loader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

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

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for buffers, bits in tqdm(train_loader):
        # --- Encode step ---
        prompts = [
            f'[ENCODE] Buffer: "{b}"\nHide bit: {bit}\n→'
            for b, bit in zip(buffers, bits.tolist())
        ]
        enc_inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        logits_enc = model(**enc_inputs).logits  # (batch, seq_len, vocab)

        # Sample soft distributions at final token positions
        last_logits = logits_enc[:, -1, :]
        soft_dists = gumbel_softmax_sample(last_logits, TAU)  # (batch, vocab)

        # Convert to embeddings
        emb_matrix = model.get_input_embeddings().weight  # (vocab, dim)
        soft_embs = soft_dists.to(emb_matrix.dtype) @ emb_matrix  # (batch, dim)

        # --- Decode step ---
        # Build hard tokens for decode prompt (for simplicity)
        hard_ids = torch.argmax(soft_dists, dim=-1)
        dec_texts = tokenizer.batch_decode(hard_ids, skip_special_tokens=True)
        dec_prompts = [f'[DECODE] Encoded: "{t}" →' for t in dec_texts]
        dec_inputs = tokenizer(
            dec_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        logits_dec = model(**dec_inputs).logits[:, -1, :]  # (batch, vocab)

        # Extract logits for "0" and "1"
        zero_id = tokenizer.convert_tokens_to_ids("0")
        one_id = tokenizer.convert_tokens_to_ids("1")
        bit_logits = logits_dec[:, [zero_id, one_id]]  # (batch, 2)

        # --- Loss computation ---
        loss_bit = torch.nn.functional.cross_entropy(bit_logits, bits.to(device))

        # Semantic similarity
        orig_emb = sim_model.encode(buffers, convert_to_tensor=True, device=device)
        enc_emb = sim_model.encode(dec_texts, convert_to_tensor=True, device=device)
        sim_scores = util.cos_sim(orig_emb, enc_emb).diag()
        loss_sim = (1 - sim_scores).mean()

        loss = loss_bit + LAMBDA_SIM * loss_sim
        wandb.log({"train/step_loss": loss})
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    wandb.log({"train/avg_loss": avg_loss, "epoch": epoch})

    # --------------------
    # 9. Log example generations
    # --------------------
    model.eval()
    example_buffers = buffers[:2]
    with torch.no_grad():
        for i, buf in enumerate(example_buffers):
            prompt = f'[ENCODE] Buffer: "{buf}"\nHide bit: 1\n→'
            generated = model.generate(
                **tokenizer(prompt, return_tensors="pt").to(device),
                max_new_tokens=20,
                do_sample=False,
            )
            text_out = tokenizer.decode(generated[0], skip_special_tokens=True)
            wandb.log(
                {
                    f"example/epoch_{epoch}/buffer_{i}": wandb.Table(
                        data=[[buf, text_out]], columns=["buffer_text", "encoded_text"]
                    )
                }
            )

    print(f"Epoch {epoch} → avg_loss: {avg_loss:.4f}")

# Save final model
model.save_pretrained("./qwen_stego_lora")
tokenizer.save_pretrained("./qwen_stego_lora")
