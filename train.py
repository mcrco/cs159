import unsloth # necessary for proper loading of unsloth models

import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import math
import wandb
from tqdm import tqdm
import random # For shuffling seeds per epoch

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
        "--batch_size", type=int, default=1, help="Training batch size (per gradient accumulation step) (default: 1)"
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
    parser.add_argument(
        "--sample_per_epoch",
        action="store_true",
        help="Enable sampling of the dataset at the beginning of each epoch."
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
        if args.sample_per_epoch:
            run_name_parts.append("spe")
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
        full_dataset = load_from_disk(dataset_path)
    except FileNotFoundError:
        print(
            f"Error: Dataset not found at {dataset_path}. Please check the path or run the appropriate dataset creation script."
        )
        return

    train_dataset_full = full_dataset["train"]
    val_dataset_full = full_dataset.get("validation") # Use .get() in case validation set is missing
    test_dataset_full = full_dataset.get("test") # Load the test set

    # Initial sampling if not doing per-epoch sampling (legacy behavior)
    if not args.sample_per_epoch and args.sample_fraction < 1.0:
        if not 0.0 < args.sample_fraction <= 1.0:
            raise ValueError("sample_fraction error.")
        print(f"Initial sampling of training dataset: {args.sample_fraction*100:.2f}%")
        num_train_samples = int(len(train_dataset_full) * args.sample_fraction)
        train_dataset_to_use = train_dataset_full.shuffle(seed=42).select(range(num_train_samples))
        print(f"Using {len(train_dataset_to_use)} samples for training initially.")
        if val_dataset_full:
            print(f"Initial sampling of validation dataset: {args.sample_fraction*100:.2f}%")
            num_val_samples = int(len(val_dataset_full) * args.sample_fraction)
            if num_val_samples == 0 and len(val_dataset_full) > 0: num_val_samples = 1 
            val_dataset_to_use = val_dataset_full.shuffle(seed=42).select(range(num_val_samples))
            print(f"Using {len(val_dataset_to_use)} samples for validation initially.")
        else:
            val_dataset_to_use = None
        
        # Test set is NOT sampled initially, always use full if available
        test_dataset_to_use = test_dataset_full
        if test_dataset_to_use:
            print(f"Using all {len(test_dataset_to_use)} samples for test set (no initial sampling).")

    else: # args.sample_per_epoch is True OR args.sample_fraction is 1.0
        train_dataset_to_use = train_dataset_full
        val_dataset_to_use = val_dataset_full
        test_dataset_to_use = test_dataset_full # Always use full test set
        if test_dataset_to_use:
             print(f"Using all {len(test_dataset_to_use)} samples for test set.")

    # DataLoaders will be initialized inside the epoch loop if sample_per_epoch is True
    # Otherwise, initialized once here
    if not args.sample_per_epoch:
        train_loader = DataLoader(
            train_dataset_to_use, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset_to_use,
            batch_size=args.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        ) if val_dataset_to_use and len(val_dataset_to_use) > 0 else None
        
        test_loader = DataLoader(
            test_dataset_to_use,
            batch_size=args.val_batch_size, # Use val_batch_size for test loader as well
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        ) if test_dataset_to_use and len(test_dataset_to_use) > 0 else None
    else:
        # Placeholders, will be set in epoch loop
        train_loader = None 
        val_loader = None
        test_loader = None # Will be set after epoch loop based on full test dataset if sample_per_epoch

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
    if args.sample_per_epoch:
        model_save_path_parts.append("spe")
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
    
    print(f"Starting training (manual PyTorch loop) for method: {args.method}...")

    # --- Training Loop ---
    global_step = 0
    for epoch in range(args.epochs):
        steg_module.train() # Set model to training mode
        epoch_total_loss_sum = 0 
        epoch_bit_loss_sum = 0   
        epoch_sim_loss_sum = 0  
        num_optimizer_steps_this_epoch = 0
        num_micro_batches_processed_epoch = 0

        current_epoch_train_dataset = train_dataset_full
        current_epoch_val_dataset = val_dataset_full

        if args.sample_per_epoch and args.sample_fraction < 1.0:
            epoch_seed = 42 + epoch # Vary seed per epoch
            print(f"\nEpoch {epoch+1}: Sampling training dataset ({args.sample_fraction*100:.2f}%) with seed {epoch_seed}")
            num_train_samples_epoch = int(len(train_dataset_full) * args.sample_fraction)
            current_epoch_train_dataset = train_dataset_full.shuffle(seed=epoch_seed).select(range(num_train_samples_epoch))
            print(f"Using {len(current_epoch_train_dataset)} samples for training this epoch.")
            if val_dataset_full:
                print(f"Epoch {epoch+1}: Sampling validation dataset ({args.sample_fraction*100:.2f}%) with seed {epoch_seed}")
                num_val_samples_epoch = int(len(val_dataset_full) * args.sample_fraction)
                if num_val_samples_epoch == 0 and len(val_dataset_full) > 0: num_val_samples_epoch = 1
                current_epoch_val_dataset = val_dataset_full.shuffle(seed=epoch_seed).select(range(num_val_samples_epoch))
                print(f"Using {len(current_epoch_val_dataset)} samples for validation this epoch.")
            else:
                current_epoch_val_dataset = None
        
        # Re-initialize DataLoaders for the current epoch's datasets
        train_loader = DataLoader(
            current_epoch_train_dataset, batch_size=args.batch_size, shuffle=True, 
            collate_fn=collate_fn, num_workers=4
        )
        val_loader = DataLoader(
            current_epoch_val_dataset, batch_size=args.val_batch_size, shuffle=False, 
            collate_fn=collate_fn, num_workers=4
        ) if current_epoch_val_dataset and len(current_epoch_val_dataset) > 0 else None

        if epoch == 0 or args.sample_per_epoch: # Calculate scheduler steps based on current train_loader
            num_optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.accumulate_grad_batches)
            num_total_optimizer_steps_for_scheduler = num_optimizer_steps_per_epoch * args.epochs # Or adjust if dynamic
            if args.sample_per_epoch:
                 # If sampling per epoch, total steps for scheduler might be an estimate or based on first epoch
                 # For simplicity, let's base it on the current epoch's loader for all epochs when sampling per epoch
                 # Or, more accurately, sum of optimizer steps across all planned epochs if lengths vary wildly.
                 # Let's assume for now that sample_fraction keeps loader length relatively constant for scheduler purposes.
                 num_total_optimizer_steps_for_scheduler = num_optimizer_steps_per_epoch * args.epochs
            else: # If not sampling per epoch, this was calculated once outside
                 num_total_optimizer_steps_for_scheduler = math.ceil(len(train_loader) / args.accumulate_grad_batches) * args.epochs

            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, 
                num_training_steps=num_total_optimizer_steps_for_scheduler
            )
            print(f"Scheduler re/initialized. Optimizer steps per epoch: {num_optimizer_steps_per_epoch}, Total for scheduler: {num_total_optimizer_steps_for_scheduler}")

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training", leave=False)
        for batch_idx, batch_data in enumerate(progress_bar):
            buffers, bits = batch_data 
            bits_on_device = bits.to(device)

            loss, bit_loss, sim_loss, intermediate_encoded_texts = steg_module.compute_loss(buffers, bits_on_device, device)
            
            if args.debug:
                print(f"\n--- Debug: Epoch {epoch+1}, Micro-Batch {batch_idx+1}/{len(train_loader)}, Global Step (pending): {global_step} ---")
                print(f'Buffer: {buffers[0]}') 
                print(f'Encoded (from loss calc): {intermediate_encoded_texts[0]}')
                print(f"Target Bit: {bits[0].item()}")
                with torch.no_grad():
                    steg_module.eval() 
                    predicted_bit_debug_list = steg_module.predict_bits_from_encoded_text(intermediate_encoded_texts[0], device=device)
                    predicted_bit_debug = predicted_bit_debug_list
                    if isinstance(predicted_bit_debug, list): predicted_bit_debug = predicted_bit_debug[0]
                    steg_module.train() 
                print(f"Predicted Bit (on this example): {predicted_bit_debug}\n")

            effective_loss = loss
            if args.accumulate_grad_batches > 1:
                effective_loss = loss / args.accumulate_grad_batches
            
            effective_loss.backward()

            # Accumulate micro-batch losses for epoch averaging
            # loss, bit_loss, sim_loss are from the current micro-batch
            epoch_bit_loss_sum += bit_loss.item() 
            epoch_sim_loss_sum += sim_loss.item()
            num_micro_batches_processed_epoch += 1

            if (batch_idx + 1) % args.accumulate_grad_batches == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                num_optimizer_steps_this_epoch += 1
                
                # Loss for this optimizer step (sum of accumulated micro-batch losses)
                # `loss` here is the last micro-batch loss (potentially scaled by accum_grad_batches)
                # To get the true loss for this step, we'd sum up the unscaled losses of micro-batches.
                # For simplicity in logging, we log the last micro-batch's main loss value, scaled as if it were the step loss.
                current_step_loss_val = loss.item() # This is the loss from the last micro-batch in accumulation window
                epoch_total_loss_sum += current_step_loss_val 

                if not args.debug:
                    wandb.log({
                        "train/step_loss": current_step_loss_val,
                        "train/step_bit_loss": bit_loss.item(), # from last micro-batch
                        "train/step_sim_loss": sim_loss.item(), # from last micro-batch
                        "learning_rate": scheduler.get_last_lr()[0],
                        "global_step": global_step
                    })
                
                progress_bar.set_postfix({
                    "loss (step)": f"{current_step_loss_val:.4f}", 
                    "bit_l (batch)": f"{bit_loss.item():.4f}", 
                    "sim_l (batch)": f"{sim_loss.item():.4f}",
                    "step": global_step
                })
        
        # --- End of Epoch Training Summary ---
        avg_epoch_total_loss = epoch_total_loss_sum / num_optimizer_steps_this_epoch if num_optimizer_steps_this_epoch > 0 else 0
        avg_epoch_bit_loss = epoch_bit_loss_sum / num_micro_batches_processed_epoch if num_micro_batches_processed_epoch > 0 else 0
        avg_epoch_sim_loss = epoch_sim_loss_sum / num_micro_batches_processed_epoch if num_micro_batches_processed_epoch > 0 else 0
        
        print(f"Epoch {epoch+1} Training Summary: Avg Total Loss (per optim step): {avg_epoch_total_loss:.4f}, Avg Bit Loss (per micro-batch): {avg_epoch_bit_loss:.4f}, Avg Sim Loss (per micro-batch): {avg_epoch_sim_loss:.4f}")
        if not args.debug:
            wandb.log({
                "train/epoch_total_loss": avg_epoch_total_loss,
                "train/epoch_bit_loss": avg_epoch_bit_loss,
                "train/epoch_sim_loss": avg_epoch_sim_loss,
                "epoch": epoch + 1
            })

        # --- Validation Loop (End of Epoch) ---
        if val_loader:
            steg_module.eval()
            val_outputs = []
            epoch_val_bit_accuracy_sum = 0
            num_val_samples_processed = 0
            print(f"\nRunning validation for Epoch {epoch+1} on {len(val_loader.dataset)} samples...")
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation", leave=False)
            with torch.no_grad():
                for val_batch_idx, val_batch_data in enumerate(val_progress_bar):
                    val_buffers, val_bits = val_batch_data
                    # val_bits is already a tensor from collate_fn

                    for i in range(len(val_buffers)):
                        buf = val_buffers[i]
                        target_bit_val = val_bits[i].item()

                        encoded_text_list = steg_module.generate_encoded_text_or_embeddings(
                            buf, target_bit_val, produce_embeddings=False, device=device
                        )
                        encoded_text = encoded_text_list
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

                        if val_batch_idx < 2 and i < 5 and args.debug: # Print few examples during debug validation
                             print(f"  Val Ex: Buf='{buf[:50]}...', Target={target_bit_val}, Enc='{encoded_text[:50]}...', Dec={processed_predicted_bit}")

                        val_outputs.append({
                            "buffer": buf, "encoded_sentence": encoded_text,
                            "decoded_bit": processed_predicted_bit, "target_bit": str(target_bit_val)
                        })
            
            avg_val_bit_accuracy = (epoch_val_bit_accuracy_sum / num_val_samples_processed) * 100 if num_val_samples_processed > 0 else 0
            print(f"Epoch {epoch+1} Validation Bit Accuracy: {avg_val_bit_accuracy:.2f}%")

            if not args.debug:
                wandb.log({"val/bit_accuracy": avg_val_bit_accuracy, "epoch": epoch + 1})
                wandb_table_data = [[item["buffer"], item["encoded_sentence"], item["decoded_bit"], item["target_bit"]] for item in val_outputs[:min(len(val_outputs), 100)]]
                wandb_table = wandb.Table(columns=["buffer", "encoded_sentence", "decoded_bit", "target_bit"], data=wandb_table_data)
                wandb.log({f"val/epoch_{epoch+1}_examples": wandb_table})
        else:
            print(f"Epoch {epoch+1}: No validation loader/data.")

    print("Training complete.")
    steg_module.save_model() # Call the save method from the module

    # --- Test Loop (After Training) ---
    if test_dataset_full: # Only run if a test dataset was loaded initially
        print(f"Starting final test evaluation...")
        
        # For the final test, we always use the full test set loaded.
        final_test_dataset_to_use = test_dataset_full

        if final_test_dataset_to_use and len(final_test_dataset_to_use) > 0:
            current_test_loader = DataLoader(
                final_test_dataset_to_use,
                batch_size=args.val_batch_size, # Re-use val_batch_size
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4
            )
            print(f"Evaluating on {len(current_test_loader.dataset)} test samples.")

            steg_module.eval()
            test_outputs = []
            test_bit_accuracy_sum = 0
            num_test_samples_processed = 0
            
            test_progress_bar = tqdm(current_test_loader, desc="Final Test Evaluation", leave=False)
            with torch.no_grad():
                for test_batch_idx, test_batch_data in enumerate(test_progress_bar):
                    test_buffers, test_bits = test_batch_data

                    for i in range(len(test_buffers)):
                        buf = test_buffers[i]
                        target_bit_val = test_bits[i].item()

                        encoded_text_list = steg_module.generate_encoded_text_or_embeddings(
                            buf, target_bit_val, produce_embeddings=False, device=device
                        )
                        encoded_text = encoded_text_list
                        if isinstance(encoded_text, list): encoded_text = encoded_text[0]
                        
                        predicted_bit_text_list = steg_module.predict_bits_from_encoded_text(encoded_text, device=device)
                        predicted_bit_text = predicted_bit_text_list
                        if isinstance(predicted_bit_text, list): predicted_bit_text = predicted_bit_text[0]

                        processed_predicted_bit = "UNK"
                        if predicted_bit_text == "0": processed_predicted_bit = "0"
                        elif predicted_bit_text == "1": processed_predicted_bit = "1"
                        
                        is_correct = (str(target_bit_val) == processed_predicted_bit)
                        test_bit_accuracy_sum += 1 if is_correct else 0
                        num_test_samples_processed +=1

                        if test_batch_idx < 2 and i < 5 and args.debug: 
                             print(f"  Test Ex: Buf='{buf[:50]}...', Target={target_bit_val}, Enc='{encoded_text[:50]}...', Dec={processed_predicted_bit}")

                        test_outputs.append({
                            "buffer": buf, "encoded_sentence": encoded_text,
                            "decoded_bit": processed_predicted_bit, "target_bit": str(target_bit_val)
                        })
            
            avg_test_bit_accuracy = (test_bit_accuracy_sum / num_test_samples_processed) * 100 if num_test_samples_processed > 0 else 0
            print(f"Final Test Bit Accuracy: {avg_test_bit_accuracy:.2f}%")

            if not args.debug:
                wandb.log({"test/bit_accuracy": avg_test_bit_accuracy})
                if test_outputs:
                    wandb_test_table_data = [[item["buffer"], item["encoded_sentence"], item["decoded_bit"], item["target_bit"]] for item in test_outputs[:min(len(test_outputs), 100)]]
                    wandb_test_table = wandb.Table(columns=["buffer", "encoded_sentence", "decoded_bit", "target_bit"], data=wandb_test_table_data)
                    wandb.log({"test/final_examples": wandb_test_table})
        else:
            print("No test samples to evaluate or test_dataset_to_use is None/empty.")
    else:
        print("No test dataset split found. Skipping final test evaluation.")

    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    main()
