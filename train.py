import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import math

from gumbel import GumbelSteganographer
from peft import LoraConfig

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
    parser = argparse.ArgumentParser(description="Train a steganography model.")
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
        "--epochs", type=int, default=1, help="Number of training epochs (default: 3)"
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

    dataset_path = AVAILABLE_DATASETS[args.dataset_name]
    use_unsloth_flag = not args.no_unsloth

    logger = None
    if not args.debug:
        run_name_parts = [
            args.method,
            args.model,
            f"ds-{args.dataset_name}",
            f"t{args.temp}",
            f"ls{args.lambda_sim}",
            f"e{args.epochs}",
            f"lr{args.lr}",
        ]
        if args.sample_fraction < 1.0:
            run_name_parts.append(f"sf{args.sample_fraction}")
        if args.no_unsloth:
            run_name_parts.append("no_unsloth")
        if args.accumulate_grad_batches > 1:
            run_name_parts.append(f"acc{args.accumulate_grad_batches}")
        run_name = "_".join(run_name_parts)
        
        logger = WandbLogger(project="steganography", name=run_name, config=args)
    else:
        print(
            "DEBUG mode enabled: wandb logging is OFF. Validation and some training logs will print to terminal."
        )
        if args.no_unsloth:
            print("Unsloth is DISABLED for model loading.")

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
    )

    # --- PEFT Configuration ---
    lora_target_modules = []
    for m in args.lora_target_modules:
        target_mod = f"{m}_proj"
        if m not in "qkvogud":
            raise ValueError(f"{target_mod} is not tuneable parameter")
        lora_target_modules.append(target_mod)

    # Base config as dict
    base_peft_config_dict = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": lora_target_modules,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        # "task_type": "CAUSAL_LM" # Often needed for standard PeftConfig
    }

    if use_unsloth_flag:
        peft_config_to_pass = base_peft_config_dict
        # Unsloth's FastLanguageModel.get_peft_model might not need task_type explicitly,
        # or handles it internally. If issues arise, it can be removed from dict for Unsloth.
    else:
        # For standard Transformers, create LoraConfig object
        peft_config_to_pass = LoraConfig(**base_peft_config_dict, task_type="CAUSAL_LM")

    # --- Calculate num_training_steps for scheduler ---
    # This is the total number of optimizer steps across all epochs
    num_optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.accumulate_grad_batches)
    num_training_steps = num_optimizer_steps_per_epoch * args.epochs

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

    if args.method == "gumbel":
        optimizer_params = {"lr": args.lr}
        scheduler_params = {
            "num_warmup_steps": args.warmup_steps,
            "num_training_steps": num_training_steps
        }

        steg_module = GumbelSteganographer(
            llm_model_name=llm_model_path,
            sim_model_name=args.sim_model,
            lora_config=peft_config_to_pass,
            temperature=args.temp,
            lambda_sim=args.lambda_sim,
            optimizer_args=optimizer_params,
            scheduler_args=scheduler_params,
            model_save_path_prefix=model_save_path,
            debug=args.debug,
            use_unsloth=use_unsloth_flag,
        )
    elif args.method == "rl":
        print("RL method not yet implemented.")
        return
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # --- Training with PyTorch Lightning Trainer ---
    print(
        f"Starting training with PyTorch Lightning for method: {args.method} with model: {args.model} on dataset: {args.dataset_name}"
    )
    if args.no_unsloth:
        print("Note: Running WITHOUT Unsloth optimizations.")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(steg_module, train_loader, val_loader)

    print("Training complete.")


if __name__ == "__main__":
    main()
