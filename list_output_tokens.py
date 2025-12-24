"""
List the key output tokens for Qwen models.
Focuses on tokens used for generation control and stopping.
"""

import os
from transformers import AutoTokenizer
from huggingface_hub import get_token


def main():
    # Model ID
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    # Get Hugging Face token for authentication
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or get_token()

    if HUGGINGFACE_API_KEY is None:
        raise ValueError("HUGGINGFACE_API_KEY is not set")

    print("=" * 70)
    print(f"Key Output Tokens for: {model_id}")
    print("=" * 70)
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_API_KEY)

    # Critical output tokens
    print("1. CRITICAL OUTPUT TOKENS (for generation control):")
    print("-" * 70)
    print(f"   EOS Token (End of Sequence):")
    print(f"      Token: {tokenizer.eos_token}")
    print(f"      ID: {tokenizer.eos_token_id}")
    print(f"      Usage: This token stops text generation")
    print()
    
    print(f"   PAD Token (Padding):")
    print(f"      Token: {tokenizer.pad_token}")
    print(f"      ID: {tokenizer.pad_token_id}")
    print(f"      Usage: Used for padding sequences")
    print()

    # Qwen-specific special tokens (last tokens in vocabulary)
    print("2. QWEN SPECIAL TOKENS (highest IDs):")
    print("-" * 70)
    
    # Get the last tokens (special tokens are typically at the end)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # Get last 30 tokens
    last_tokens = sorted_vocab[-30:]
    
    for token, token_id in last_tokens:
        token_display = token.encode('ascii', 'backslashreplace').decode('ascii')
        print(f"   ID {token_id:6d}: {token_display}")
    print()

    # Most important tokens for generation
    print("3. TOKENS USED IN GENERATION:")
    print("-" * 70)
    print("   When generating text, you should set:")
    print(f"      eos_token_id = {tokenizer.eos_token_id}  # Stop generation here")
    print(f"      pad_token_id = {tokenizer.pad_token_id}  # For padding")
    print()
    
    print("   In your generate() call, use:")
    print(f"      model.generate(..., eos_token_id={tokenizer.eos_token_id})")
    print()

    # Chat format tokens
    print("4. CHAT FORMAT TOKENS:")
    print("-" * 70)
    chat_tokens = {
        '<|im_start|>': 151644,
        '<|im_end|>': 151645,
        '<|endoftext|>': 151643,
    }
    
    for token, token_id in chat_tokens.items():
        print(f"   {token:20s} -> ID {token_id}")
    print()
    print("   Note: <|im_end|> (ID 151645) is the EOS token and stops generation")
    print()

    # All special tokens with IDs > 150000 (typical special token range)
    print("5. ALL SPECIAL TOKENS (IDs >= 150000):")
    print("-" * 70)
    special_tokens_list = [(token, token_id) for token, token_id in sorted_vocab if token_id >= 150000]
    
    for token, token_id in special_tokens_list:
        token_display = token.encode('ascii', 'backslashreplace').decode('ascii')
        print(f"   ID {token_id:6d}: {token_display}")
    print()

    print("=" * 70)
    print("SUMMARY FOR CODE:")
    print("=" * 70)
    print(f"   Vocabulary size: {len(tokenizer)}")
    print(f"   EOS token ID (use in generate()): {tokenizer.eos_token_id}")
    print(f"   PAD token ID: {tokenizer.pad_token_id}")
    print()
    print("   Recommended generate() parameters:")
    print(f"      eos_token_id={tokenizer.eos_token_id}")
    print(f"      pad_token_id={tokenizer.pad_token_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()

