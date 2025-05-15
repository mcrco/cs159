import unsloth

import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import math
import wandb
from tqdm import tqdm

from transformers.optimization import get_linear_schedule_with_warmup

MODEL_NAMES = {
    "llama": "unsloth/Llama-3.2-3B-Instruct",
    "gemma": "unsloth/gemma-3-4b-it",
    "gemma1b": "unsloth/gemma-3-1b-it",
    "olmo": "unsloth/OLMo-2-0425-1B-Instruct",
    "qwen7b": "unsloth/Qwen2.5-7B-Instruct",
    "qwen3b": "unsloth/Qwen2.5-3B-Instruct",
    "qwen1.5b": "unsloth/Qwen2.5-1.5B-Instruct",
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
}
SIM_MODEL_NAME = "all-MiniLM-L6-v2"

# Define available datasets and their corresponding path prefixes/suffixes
AVAILABLE_DATASETS = {
    "bible": "stego_dataset_bible/",
    "childrens_classics": "stego_dataset_childrens_classics/",
    "wizard_of_oz": "stego_dataset_wizard_of_oz/",
    "pride_and_prejudice": "stego_dataset_pride_and_prejudice/",
}


def collate_fn(batch):
    buffers = [item["buffer_text"] for item in batch]
    bits = torch.tensor([item["bit"] for item in batch], dtype=torch.long)
    return buffers, bits


def main():
    parser = argparse.ArgumentParser(description="Train a steganography model with PyTorch.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["gumbel", "rl"],
        default="gumbel",
        help="Method to use for steganography (gumbel or rl). (default: gumbel)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable wandb and print more logs to terminal.",
    )
    parser.add_argument(
        "--no_unsloth",
        action="store_true",
        help="Disable Unsloth and use standard Transformers for model loading.",
    )
    parser.add_argument(
        "--model",
        default="qwen3b",
        help=f"Base LLM model name. Choices: {list(MODEL_NAMES.keys())}. (default: qwen3b)",
    )
    parser.add_argument(
        "--sim_model",
        default=SIM_MODEL_NAME,
        help=f"Sentence similarity model name (default: {SIM_MODEL_NAME})",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Temperature for Gumbel-Softmax (if applicable) (default: 1.0)",
    )
    parser.add_argument(
        "--lambda_sim",
        type=float,
        default=0.0,
        help="Weight for similarity loss (if applicable) (default: 0.0)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Training batch size (default: 1)"
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1,
        help="Validation batch size (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for the scheduler (default: 500)",
    )
    parser.add_argument(
        "--lora_r", type=int, default=8, help="LoRA r parameter (default: 8)"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha parameter (default: 32)"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="qkv",
        help="LoRA target modules (default: 'qkv')",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bible",
        choices=list(AVAILABLE_DATASETS.keys()),
        help="Name of the dataset to use. (default: bible)",
    )
    parser.add_argument(
        "--model_save_prefix",
        type=str,
        default="./stego_model",
        help="Prefix for saving the trained model (default: ./stego_model)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use for training and validation (0.0 to 1.0). (default: 1.0)",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients over. (default: 1)"
    )

    args = parser.parse_args()

    # --- Setup Device --- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = AVAILABLE_DATASETS[args.dataset_name]
    use_unsloth_flag = not args.no_unsloth

    LoraConfig = None # Initialize, might be imported if not use_unsloth_flag
    if not use_unsloth_flag:
        from gumbel_hf import GumbelSteganographerHF
        from peft import LoraConfig # Import LoraConfig for HF case
    else:
        from gumbel import GumbelSteganographer

    # --- Wandb Logger Setup (manual) ---
    if not args.debug:
        run_name_parts = [
            args.method,
            args.model,
            f"ds-{args.dataset_name}",
            f"t{args.temp}",
            f"ls{args.lambda_sim}",
            f"e{args.epochs}",
            f"lr{args.lr}"
        ]
        if args.sample_fraction < 1.0:
            run_name_parts.append(f"sf{args.sample_fraction}")
        if args.no_unsloth:
            run_name_parts.append("no_unsloth")
        if args.accumulate_grad_batches > 1:
            run_name_parts.append(f"acc{args.accumulate_grad_batches}")
        run_name = "_".join(run_name_parts)
        
        wandb.init(project="steganography", name=run_name, config=args)
    else:
        print("DEBUG mode enabled: wandb logging is OFF.")
        if args.no_unsloth:
            print("Unsloth is DISABLED. Using Hugging Face Transformers.")

    # --- Dataset and DataLoaders ---
    try:
        dataset = load_from_disk(dataset_path)
    except FileNotFoundError:
        print(
            f"Error: Dataset not found at {dataset_path}. Please check the path or run the appropriate dataset creation script."
        )
        return

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    if args.sample_fraction < 1.0:
        if not 0.0 < args.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be between 0.0 (exclusive) and 1.0 (inclusive).")
        print(f"Sampling {args.sample_fraction*100:.2f}% of the dataset.")
        num_train_samples = int(len(train_dataset) * args.sample_fraction)
        train_dataset = train_dataset.shuffle(seed=42).select(range(num_train_samples))
        print(f"Using {len(train_dataset)} samples for training after sampling.")
        num_val_samples = int(len(val_dataset) * args.sample_fraction)
        if num_val_samples == 0 and len(val_dataset) > 0:
            num_val_samples = 1 
        if len(val_dataset) > 0:
            val_dataset = val_dataset.shuffle(seed=42).select(range(num_val_samples))
            print(f"Using {len(val_dataset)} samples for validation after sampling.")
        else:
            print("Validation dataset is empty, no sampling applied.")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    ) if len(val_dataset) > 0 else None

    # --- PEFT Configuration ---
    lora_target_modules_list = []
    if args.lora_target_modules:
        for m_char in args.lora_target_modules: # Assuming qkv means q_proj, k_proj, v_proj
            if m_char == 'q': lora_target_modules_list.append("q_proj")
            elif m_char == 'k': lora_target_modules_list.append("k_proj")
            elif m_char == 'v': lora_target_modules_list.append("v_proj")
            elif m_char == 'o': lora_target_modules_list.append("o_proj")
            elif m_char == 'g': lora_target_modules_list.append("gate_proj")
            elif m_char == 'u': lora_target_modules_list.append("up_proj")
            elif m_char == 'd': lora_target_modules_list.append("down_proj")
            else: print(f"Warning: Unknown LoRA target module character: {m_char}")
    
    base_peft_config_dict = {
        "r": args.lora_r, "lora_alpha": args.lora_alpha,
        "target_modules": lora_target_modules_list, # Use the parsed list
        "lora_dropout": args.lora_dropout, "bias": "none",
    }

    if use_unsloth_flag:
        peft_config_to_pass = base_peft_config_dict
    else:
        if LoraConfig is None: raise ImportError("LoraConfig not imported for HF mode.")
        peft_config_to_pass = LoraConfig(**base_peft_config_dict, task_type="CAUSAL_LM")

    # --- Model Instantiation ---
    llm_model_path = MODEL_NAMES.get(
        args.model, args.model
    )  # Allow custom paths if not in MODEL_NAMES
    model_save_path_parts = [
        args.model_save_prefix,
        args.method,
        args.model,
        f"ds-{args.dataset_name}",
    ]
    if args.no_unsloth:
        model_save_path_parts.append("no_unsloth")
    model_save_path = "_".join(model_save_path_parts)

    steg_module_class = GumbelSteganographer if use_unsloth_flag else GumbelSteganographerHF
    steg_module_args = {
        "llm_model_name": llm_model_path,
        "sim_model_name": args.sim_model,
        "lora_config": peft_config_to_pass,
        "temperature": args.temp,
        "lambda_sim": args.lambda_sim,
        "model_save_path_prefix": model_save_path,
        "debug": args.debug,
    }
    steg_module = steg_module_class(**steg_module_args).to(device)

    # --- Optimizer and Scheduler --- 
    # Optimizer args were part of GumbelSteganographer in PL, now we define them here.
    optimizer = torch.optim.AdamW(steg_module.model.parameters(), lr=args.lr) 
    
    num_training_steps = math.ceil(len(train_loader) / args.accumulate_grad_batches) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    print(f"Starting training (manual PyTorch loop) for method: {args.method}...")

    # --- Training Loop ---
    global_step = 0
    for epoch in range(args.epochs):
        steg_module.train() # Set model to training mode
        epoch_train_loss = 0
        epoch_train_bit_loss = 0
        epoch_train_sim_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            buffers, bits = batch
            bits = bits.to(device) # Move bits to device

            # Forward pass through GumbelSteganographer's compute_loss method
            total_loss, loss_bit, loss_sim, intermediate_encoded_texts = steg_module.compute_loss(buffers, bits, device)

            # Normalize loss for gradient accumulation
            if args.accumulate_grad_batches > 1:
                total_loss = total_loss / args.accumulate_grad_batches
            
            total_loss.backward()

            if (batch_idx + 1) % args.accumulate_grad_batches == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                scheduler.step() # Scheduler steps with optimizer
                optimizer.zero_grad()
            
            epoch_train_loss += total_loss.item() * args.accumulate_grad_batches if args.accumulate_grad_batches > 1 else total_loss.item()
            epoch_train_bit_loss += loss_bit.item()
            epoch_train_sim_loss += loss_sim.item()

            if not args.debug:
                wandb.log({
                    "train/step_loss": total_loss.item() * args.accumulate_grad_batches if args.accumulate_grad_batches > 1 else total_loss.item(),
                    "train/step_bit_loss": loss_bit.item(),
                    "train/step_sim_loss": loss_sim.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "global_step": global_step
                })
            
            progress_bar.set_postfix({
                "loss": f"{total_loss.item():.4f}", 
                "bit_l": f"{loss_bit.item():.4f}", 
                "sim_l": f"{loss_sim.item():.4f}"
            })
            global_step += 1

            if args.debug and global_step % 10 == 0 and batch_idx % args.accumulate_grad_batches == 0:
                print(f"--- Training Step {global_step} (Epoch {epoch+1}, Batch {batch_idx+1}) ---")
                print(f'Buffer: {buffers[0]}')
                print(f'Encoded (for DECODE): {intermediate_encoded_texts[0]}')
                print(f"Target Bit: {bits[0].item()}") # Assuming bits is a tensor
                # For predicted bit, we need to run inference part
                # This might be too slow for every debug step, but for illustration:
                with torch.no_grad():
                    steg_module.eval() # Set to eval for this prediction
                    predicted_bits_texts_log = steg_module.predict_bits_from_encoded_text(intermediate_encoded_texts[0], device=device)
                    steg_module.train() # Set back to train
                print(f"Predicted Bit (by DECODE part): {predicted_bits_texts_log}\n")

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_bit_loss = epoch_train_bit_loss / len(train_loader)
        avg_train_sim_loss = epoch_train_sim_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Train Loss: {avg_train_loss:.4f}, Avg Bit Loss: {avg_train_bit_loss:.4f}, Avg Sim Loss: {avg_train_sim_loss:.4f}")
        if not args.debug:
            wandb.log({
                "train/epoch_loss": avg_train_loss,
                "train/epoch_bit_loss": avg_train_bit_loss,
                "train/epoch_sim_loss": avg_train_sim_loss,
                "epoch": epoch + 1
            })

        # --- Validation Loop ---
        if val_loader:
            steg_module.eval() # Set model to evaluation mode
            val_outputs = []
            epoch_val_bit_accuracy_sum = 0
            num_val_samples_processed = 0

            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation", leave=False)
            with torch.no_grad():
                for val_batch in val_progress_bar:
                    val_buffers, val_bits = val_batch
                    # val_bits = val_bits.to(device) # Already tensor, move to device

                    for i in range(len(val_buffers)):
                        buf = val_buffers[i]
                        target_bit_val = val_bits[i].item()

                        # Ensure generation and prediction happen on the correct device
                        encoded_text_list = steg_module.generate_encoded_text_or_embeddings(
                            buf, target_bit_val, produce_embeddings=False, device=device
                        )
                        encoded_text = encoded_text_list # It returns a list with one item or single string
                        if isinstance(encoded_text, list): encoded_text = encoded_text[0]
                        
                        predicted_bit_text_list = steg_module.predict_bits_from_encoded_text(encoded_text, device=device)
                        predicted_bit_text = predicted_bit_text_list
                        if isinstance(predicted_bit_text, list): predicted_bit_text = predicted_bit_text[0]

                        processed_predicted_bit = "UNK"
                        if predicted_bit_text == "0": processed_predicted_bit = "0"
                        elif predicted_bit_text == "1": processed_predicted_bit = "1"
                        
                        is_correct = (str(target_bit_val) == processed_predicted_bit)
                        epoch_val_bit_accuracy_sum += 1 if is_correct else 0
                        num_val_samples_processed +=1

                        val_outputs.append({
                            "buffer": buf, "encoded_sentence": encoded_text,
                            "decoded_bit": processed_predicted_bit, "target_bit": str(target_bit_val)
                        })
            
            avg_val_bit_accuracy = (epoch_val_bit_accuracy_sum / num_val_samples_processed) * 100 if num_val_samples_processed > 0 else 0
            print(f"Epoch {epoch+1} Validation Bit Accuracy: {avg_val_bit_accuracy:.2f}%")

            if not args.debug:
                wandb.log({"val/bit_accuracy": avg_val_bit_accuracy, "epoch": epoch + 1})
                # Log table of examples (careful with large validation sets)
                # Log a subset if too large, e.g., val_outputs[:50]
                wandb_table_data = [[item["buffer"], item["encoded_sentence"], item["decoded_bit"], item["target_bit"]] for item in val_outputs[:min(len(val_outputs), 100)]]
                wandb_table = wandb.Table(columns=["buffer", "encoded_sentence", "decoded_bit", "target_bit"], data=wandb_table_data)
                wandb.log({f"val/epoch_{epoch+1}_examples": wandb_table})
            else: # Debug mode: print some validation examples
                print(f"--- Validation Epoch {epoch+1} Examples (first few) ---")
                for output in val_outputs[:min(len(val_outputs), 5)]:
                    print(f'Buffer: "{output["buffer"]}" Target: {output["target_bit"]} Encoded: "{output["encoded_sentence"]}" Decoded: {output["decoded_bit"]}')
    
    # --- End of Training --- 
    print("Training complete.")
    steg_module.save_model() # Call the save method from the module

    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    main()
