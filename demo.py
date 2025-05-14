from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template as unsloth_get_chat_template
import argparse
import torch
from peft import PeftModel

# Standard Hugging Face imports, used if not using Unsloth
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAMES = {
    "llama": "unsloth/Llama-3.2-3B-Instruct",
    "gemma": "unsloth/gemma-3-4b-it",
    "gemma1b": "unsloth/gemma-3-1b-it",
    "olmo": "unsloth/OLMo-2-0425-1B-Instruct",
    "qwen7b": "unsloth/Qwen2.5-7B-Instruct",
    "qwen3b": "unsloth/Qwen2.5-3B-Instruct",
    "qwen1.5b": "unsloth/Qwen2.5-1.5B-Instruct",
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
    # Add non-Unsloth model names here if you want to map them directly, 
    # or rely on full HF paths being passed via --model argument.
    # For example:
    # "hf_llama3.1_8b": "meta-llama/Llama-3.1-8B-Instruct",
}

with open("prompt.txt", "r") as f:
    INSTRUCTION = f.read()

def load_model_for_demo(base_model_arg, lora_adapter_path, use_unsloth: bool = True):
    """Loads the base model, applies LoRA adapter, and returns model and tokenizer."""
    
    # Resolve model path: Use MODEL_NAMES if it's a key, otherwise assume it's a direct path.
    # If not using Unsloth, you might want to adjust MODEL_NAMES to point to standard HF paths 
    # or ensure the user provides the full HF path for non-Unsloth models.
    resolved_base_model_path = MODEL_NAMES.get(base_model_arg, base_model_arg)
    model_name_for_chat_template = resolved_base_model_path # For chat template logic

    if use_unsloth:
        print(f"Loading model and tokenizer for {resolved_base_model_path} using Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=resolved_base_model_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
            # trust_remote_code=True, # Unsloth often handles this internally
        )
    else:
        print(f"Loading model and tokenizer for {resolved_base_model_path} using standard Transformers...")
        tokenizer = AutoTokenizer.from_pretrained(resolved_base_model_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved_base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            # trust_remote_code=True # May be needed for some models
        )

    # Apply chat template if using Unsloth and model type is known
    if use_unsloth:
        # Unsloth's get_chat_template is specific. For non-Unsloth, this step is skipped,
        # relying on the tokenizer's default or manual configuration if needed.
        if 'llama' in model_name_for_chat_template.lower():
            tokenizer = unsloth_get_chat_template(tokenizer, chat_template="llama-3.1")
        elif 'qwen' in model_name_for_chat_template.lower():
            tokenizer = unsloth_get_chat_template(tokenizer, chat_template="qwen-2.5")
        elif 'gemma' in model_name_for_chat_template.lower():
            tokenizer = unsloth_get_chat_template(tokenizer, chat_template="gemma-3")
        elif 'olmo' in model_name_for_chat_template.lower():
            tokenizer = unsloth_get_chat_template(tokenizer, chat_template="olmo")
    else:
        print("Relying on tokenizer's default chat template or pre-configuration for non-Unsloth mode.")

    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id

    if lora_adapter_path:
        print(f"Loading LoRA adapter from {lora_adapter_path}...")
        # PeftModel.from_pretrained works for both Unsloth and standard HF PEFT models
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        print(f"Model {resolved_base_model_path} with LoRA adapter from {lora_adapter_path} loaded.")
    else:
        print(f"Base model {resolved_base_model_path} loaded without LoRA adapter.")

    model.eval()
    print(f"Model is primarily on device: {model.device}")
    return model, tokenizer

def run_encode(model, tokenizer):
    """Handles the encoding process in the demo."""
    buffer_text = input("Enter the buffer sentence: ")
    bit_to_hide_str = ""
    while bit_to_hide_str not in ["0", "1"]:
        bit_to_hide_str = input("Enter the bit to hide (0 or 1): ")
        if bit_to_hide_str not in ["0", "1"]:
            print("Invalid input. Bit must be 0 or 1.")
    bit_to_hide = int(bit_to_hide_str)

    prompt_content = f'{INSTRUCTION}\n[ENCODE]\nBuffer: "{buffer_text}"\nHide bit: {bit_to_hide}\n'
    
    messages = [{"role": "user", "content": prompt_content}]
    # enable_thinking=False might be Unsloth specific, check if it causes issues for HF
    # For HF, often just add_generation_prompt=True is enough.
    # Keeping it for now, assuming it doesn't break standard HF tokenizers.
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(model.device)
    attention_mask = None 
    
    buffer_tokens_count = len(tokenizer.tokenize(buffer_text))
    current_max_new_tokens = buffer_tokens_count + 10 # Dynamic based on buffer

    print(f"Encoding (max_new_tokens={current_max_new_tokens})...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            max_new_tokens=current_max_new_tokens, 
            do_sample=False,  
            pad_token_id=tokenizer.pad_token_id # Important to use the actual pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nEncoded sentence:\n{generated_text.strip()}")

def run_decode(model, tokenizer):
    """Handles the decoding process in the demo."""
    encoded_text = input("Enter the encoded sentence to decode: ")
    prompt_content = f'{INSTRUCTION}\n[DECODE] Encoded: "{encoded_text}"\n'

    messages = [{"role": "user", "content": prompt_content}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(model.device)
    attention_mask = None 

    print("Decoding...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,  
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    predicted_bit_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f'\nPredicted bit: {predicted_bit_text.strip()}')

def run_example(model, tokenizer):
    """Handles the chat process in the demo."""
    try:
        with open("./example-prompt.txt", "r") as f: 
            text_content = f.read()
    except FileNotFoundError:
        print("Error: example-prompt.txt not found in the current directory.")
        return

    messages = [{"role": "user", "content": text_content}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(model.device)
    attention_mask = None
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"{generated_text.strip()}")

def main_demo():
    parser = argparse.ArgumentParser(description="Interactive demo for steganography model.")
    parser.add_argument("--model", type=str, default="llama1b",
                        help="Identifier for the base LLM. Can be a key from MODEL_NAMES or a full HF path.")
    parser.add_argument("--lora_adapter_path", type=str, default=None,
                        help="Optional path to the trained LoRA adapter directory.")
    parser.add_argument("--no_unsloth", action="store_true",
                        help="Disable Unsloth and use standard Transformers for model loading.")

    args = parser.parse_args()
    use_unsloth_flag = not args.no_unsloth

    if args.no_unsloth:
        print("Unsloth is DISABLED for model loading in the demo.")
        # Potentially update MODEL_NAMES to non-Unsloth paths if --no_unsloth is True
        # and the user provided a short key from MODEL_NAMES that implies an Unsloth model.
        # This is a more advanced step; for now, we rely on the user providing a full HF path
        # or knowing that the MODEL_NAMES key maps to a model loadable by standard Transformers.

    try:
        model, tokenizer = load_model_for_demo(args.model, args.lora_adapter_path, use_unsloth=use_unsloth_flag)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have provided correct paths/model identifiers and have necessary libraries installed.")
        print("If using --no_unsloth, ensure bitsandbytes is installed for 4-bit loading.")
        return

    while True:
        print("\n--- Steganography Demo ---")
        print("1. Encode a sentence")
        print("2. Decode a sentence")
        print("3. Run example")
        print("4. Quit")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            run_encode(model, tokenizer)
        elif choice == '2':
            run_decode(model, tokenizer)
        elif choice == '3':
            run_example(model, tokenizer)
        elif choice == '4':
            print("Exiting demo.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main_demo() 