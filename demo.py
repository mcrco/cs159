import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAMES = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-3-4b-it",
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    "qwen": "Qwen/Qwen3-0.6B",
}

INSTRUCTION = """
### 
You are a specialized steganography language model.

When you are given buffer text, a single sentence, with [ENCODE] in front and a hidden bit, you will somehow modify the buffer text to encode that bit.
However, you will not tell the user how. You simply modify the buffer text to make it work, and then write out the buffer text. 
For example, if you are given the pattern

[ENCODE] 
Buffer: "{buffer text}" 
Hide bit: 1

You will output a modified version of just {buffer text}, still a grammatically correct and semantically similar sentence, that somehow has encoded the bit 1 into its meaning.

When you are given encoded text with [DECODE] in front, you will somehow predict the hidden bit you encoded.
Once again, you do not say how you did it. You simply output a single bit. 
For example, if you are given the pattern

[DECODE] Encoded: "{encoded text}"
Hide bit:

You will output 0 if you think the encoded sentence is encoding 0, and 1 if you think the encoded sentences is encoding 1.
###
"""

def load_model_for_demo(base_model_arg, lora_adapter_path):
    """Loads the base model, applies LoRA adapter, and returns model and tokenizer."""
    resolved_base_model_path = MODEL_NAMES.get(base_model_arg, base_model_arg)
    
    print(f"Loading tokenizer for {resolved_base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {resolved_base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model_path,
        device_map="auto",  # Automatically places model on available device(s)
        # load_in_8bit=True,  # Load in 8-bit for efficiency
        trust_remote_code=True
    )

    if lora_adapter_path:
        print(f"Loading LoRA adapter from {lora_adapter_path}...")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        print(f"Model {resolved_base_model_path} with LoRA adapter from {lora_adapter_path} loaded.")
    else:
        print(f"Base model {resolved_base_model_path} loaded without LoRA adapter.")

    # Optional: Merge LoRA layers with the base model for faster inference.
    # This makes the model no longer a PeftModel, but a standard HF model with merged weights.
    # if lora_adapter_path: # Only try to merge if LoRA was loaded
    #     print("Merging LoRA adapter into the base model...")
    #     model = model.merge_and_unload()
    
    model.eval()  # Set the model to evaluation mode
    
    # Get the device the model (or its first parameter) is on
    # For models loaded with device_map="auto", different parts might be on different devices.
    # For operations, inputs should be moved to the device of the specific module they interact with.
    # However, model.generate() usually handles internal device placement correctly.
    # We'll move tokenized inputs to model.device, which usually refers to the device of the first parameter.
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
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(model.device)
    # For instruct models, generate might not need attention_mask if input_ids are already shaped by chat_template
    # and no padding is involved in this specific input.
    # However, if issues arise, an attention_mask=torch.ones_like(input_ids) could be added.
    attention_mask = None # Often okay for unpadded single sequence from apply_chat_template
    
    buffer_tokens_count = len(tokenizer.tokenize(buffer_text))
    # max_new_tokens should be enough for the modified buffer text.
    # Using buffer_tokens_count + a margin (e.g., 30) as a heuristic.
    # Your validate function used a fixed 20, which might be too short for longer buffers.
    current_max_new_tokens = buffer_tokens_count + 10

    print(f"Encoding (max_new_tokens={current_max_new_tokens})...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, # Pass it, will be None if not set for instruct
            max_new_tokens=current_max_new_tokens, 
            do_sample=False,  # For deterministic output
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice off the prompt part to get only the generated (encoded) text
    # input_ids.shape[1] is the length of the tokenized prompt (either plain or chat-templated)
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nEncoded sentence:\n{generated_text.strip()}")

def run_decode(model, tokenizer):
    """Handles the decoding process in the demo."""
    encoded_text = input("Enter the encoded sentence to decode: ")
    prompt_content = f'{INSTRUCTION}\n[DECODE] Encoded: "{encoded_text}"\nHide bit:\n'

    messages = [{"role": "user", "content": prompt_content}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(model.device)
    attention_mask = None 

    print("Decoding...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,  # Expecting a single bit ("0" or "1")
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # input_ids.shape[1] is the length of the tokenized prompt
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
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"{generated_text.strip()}")
        

def main_demo():
    parser = argparse.ArgumentParser(description="Interactive demo for steganography model.")
    parser.add_argument("--base_model_name_or_path", type=str, default="gemma",
                        help="Identifier for the base LLM (e.g., 'gemma', 'llama', 'olmo', or a HF path).")
    parser.add_argument("--lora_adapter_path", type=str, default=None,
                        help="Optional path to the trained LoRA adapter directory (e.g., ./stego_model_gumbel_llama_final).")

    args = parser.parse_args()

    try:
        model, tokenizer = load_model_for_demo(args.base_model_name_or_path, args.lora_adapter_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have provided correct paths and have necessary libraries installed (e.g., bitsandbytes for 8-bit loading).")
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