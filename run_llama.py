"""
Basic script to run a 7B Llama model locally using PyTorch and Hugging Face.

The model will be automatically downloaded from Hugging Face on first run.
For example: meta-llama/Llama-2-7b-chat-hf

Usage:
    python run_llama.py --model meta-llama/Llama-2-7b-chat-hf --prompt "Your prompt here"
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from huggingface_hub import get_token


def main():
    parser = argparse.ArgumentParser(description="Run a 7B Llama model locally")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Hugging Face model ID or local path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is artificial intelligence?",
        help="The prompt to send to the model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (0.0 to 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run the model on (auto detects GPU if available)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"\nLoading model: {args.model}")
    print("This may take a moment on first run (model will be downloaded)...")

    # Get Hugging Face token for authentication
    hf_token = os.getenv("HF_TOKEN") or get_token()
    if hf_token:
        print("Using Hugging Face token for authentication")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            token=hf_token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        if device == "cpu":
            model = model.to(device)

        model.eval()
        print("Model loaded successfully!")

        print(f"\nPrompt: {args.prompt}\n")
        print("Generating response...\n")

        # Tokenize input
        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        if response.startswith(args.prompt):
            response = response[len(args.prompt):].strip()

        print("Response:")
        print(response)
        print("\n" + "=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: If using a gated model (like meta-llama), you may need to:")
        print("1. Accept the license on Hugging Face")
        print("2. Login using: huggingface-cli login")
        raise


if __name__ == "__main__":
    main()
