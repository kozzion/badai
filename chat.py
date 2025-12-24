"""
Interactive chat script for the Llama model using PyTorch and Hugging Face.
Press Ctrl+C or type 'exit' to quit.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import get_token


def format_chat_prompt(user_message, system_message=None):
    """Format a chat prompt for Llama 2 chat models."""
    if system_message:
        prompt = (
            f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
            f"{user_message} [/INST]"
        )
    else:
        prompt = f"<s>[INST] {user_message} [/INST]"
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with Llama model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Hugging Face model ID or local path",
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

    print(f"\nLoading model: {args.model}")
    print("This may take a moment...\n")

    # Get Hugging Face token for authentication
    hf_token = os.getenv("HF_TOKEN") or get_token()
    if hf_token:
        print("Using Hugging Face token for authentication\n")

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
        print("Type your message and press Enter. Type 'exit' to quit.\n")
        print("=" * 60)

        # Chat loop
        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue

                # Format prompt for chat model
                prompt = format_chat_prompt(user_input)

                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                print("\nAssistant: ", end="", flush=True)

                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Decode response
                response = tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )

                # Extract just the assistant's response
                if "[/INST]" in response:
                    response = response.split("[/INST]")[-1].strip()
                else:
                    # Fallback: remove the prompt
                    if response.startswith(prompt):
                        response = response[len(prompt):].strip()

                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break

    except Exception as e:
        print(f"Error: {e}")
        print(
            "\nNote: If using a gated model (like meta-llama), "
            "you may need to:"
        )
        print("1. Accept the license on Hugging Face")
        print("2. Login using: huggingface-cli login")


if __name__ == "__main__":
    main()
