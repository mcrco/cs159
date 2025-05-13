import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from peft import LoraConfig, TaskType
import wandb

from gumbel import GumbelSteganographer

MODEL_NAMES = {
    "gemma": "google/gemma-3-4b-it",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    "qwen": "Qwen/Qwen3-0.6B",
}
SIM_MODEL_NAME = "all-MiniLM-L6-v2"

def collate_fn(batch):
    buffers = [item["buffer_text"] for item in batch]
    bits = torch.tensor([item["bit"] for item in batch], dtype=torch.long)
    return buffers, bits

def main():
    parser = argparse.ArgumentParser(description="Train a steganography model.")
    parser.add_argument("--method", type=str, choices=['gumbel', 'rl'], default='gumbel',
                        help="Method to use for steganography (gumbel or rl). (default: gumbel)")
    parser.add_argument("--debug", action="store_true", 
                        help="Disable wandb and print more logs to terminal.")
    parser.add_argument("--model", default="gemma", 
                        help=f"Base LLM model name. Choices: {list(MODEL_NAMES.keys())}. (default: gemma)")
    parser.add_argument("--sim_model", default=SIM_MODEL_NAME, 
                        help=f"Sentence similarity model name (default: {SIM_MODEL_NAME})")
    parser.add_argument("--temp", type=float, default=1.0, 
                        help="Temperature for Gumbel-Softmax (if applicable) (default: 1.0)")
    parser.add_argument("--lambda_sim", type=float, default=0.1, 
                        help="Weight for similarity loss (if applicable) (default: 0.5)")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Training batch size (default: 1)")
    parser.add_argument("--val_batch_size", type=int, default=1, 
                        help="Validation batch size (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the optimizer (default: 1e-4)")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps for the scheduler (default: 500)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA r parameter (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "v_proj"],
                        help="LoRA target modules (default: [\'q_proj\', \'v_proj\'])")
    parser.add_argument("--dataset_path", type=str, default="stego_dataset_wizard_of_oz/",
                        help="Path to the cached dataset (default: stego_dataset_wizard_of_oz/)")
    parser.add_argument("--model_save_prefix", type=str, default="./stego_model",
                        help="Prefix for saving the trained model (default: ./stego_model)")


    args = parser.parse_args()

    if not args.debug:
        run_name = f"{args.method}_{args.model}_t{args.temp}_ls{args.lambda_sim}_e{args.epochs}_lr{args.lr}"
        wandb.init(project="steganography", name=run_name, config=args)
    else:
        print("DEBUG mode enabled: wandb logging is OFF. Validation and some training logs will print to terminal.")

    # --- Dataset and DataLoaders ---
    try:
        dataset = load_from_disk(args.dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.dataset_path}. Please check the path.")
        print("You might need to download or prepare the dataset first.")
        return
        
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate_fn
    )

    # --- PEFT Configuration ---
    peft_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.lora_target_modules,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
    }

    # --- Model Instantiation ---
    llm_model_path = MODEL_NAMES.get(args.model, args.model) # Allow custom paths if not in MODEL_NAMES
    model_save_path = f"{args.model_save_prefix}_{args.method}_{args.model}"

    if args.method == "gumbel":
        optimizer_params = {'lr': args.lr}
        scheduler_params = {'num_warmup_steps': args.warmup_steps} # num_training_steps added in class
        
        steg = GumbelSteganographer(
            llm_model_name=llm_model_path,
            sim_model_name=args.sim_model,
            lora_config=peft_config,
            temperature=args.temp,
            lambda_sim=args.lambda_sim,
            debug=args.debug,
            optimizer_args=optimizer_params,
            scheduler_args=scheduler_params
        )
    elif args.method == "rl":
        pass
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # --- Training ---
    print(f"Starting training for method: {args.method} with model: {args.model}")
    steg.train(train_loader, val_loader, num_epochs=args.epochs, model_save_path_prefix=model_save_path)

    if not args.debug:
        wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    main() 