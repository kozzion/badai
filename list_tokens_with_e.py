"""
Script to collect all non-special tokens containing the letter 'e'
from Qwen models. Writes the results to a JSON file.
"""

import os
import json
from transformers import AutoTokenizer
from huggingface_hub import get_token


def is_special_token(token_id, tokenizer, special_threshold=150000):
    """
    Determine if a token is a special token.
    Special tokens typically have high IDs (>= threshold)
    or are marked as special.
    """
    # Check if it's in the high ID range (typical for special tokens)
    if token_id >= special_threshold:
        return True

    # Check if it's one of the known special token IDs
    special_ids = [
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ]
    if tokenizer.sep_token_id:
        special_ids.append(tokenizer.sep_token_id)
    if tokenizer.cls_token_id:
        special_ids.append(tokenizer.cls_token_id)
    if tokenizer.mask_token_id:
        special_ids.append(tokenizer.mask_token_id)

    if token_id in special_ids:
        return True

    # Check if token starts with special markers
    try:
        token_str = tokenizer.decode([token_id])
        # Common special token patterns
        if token_str.startswith('<|') and token_str.endswith('|>'):
            return True
        if token_str.startswith('<') and token_str.endswith('>'):
            # Could be XML-style tags, check more carefully
            special_patterns = [
                '<|im_', '<|end', '<|user',
                '<|assistant', '<|system', '<tool'
            ]
            if any(tag in token_str for tag in special_patterns):
                return True
    except Exception:
        pass

    return False


def main():
    # Model ID - same as example_qwen.py
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    # Get Hugging Face token for authentication
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or get_token()

    if HUGGINGFACE_API_KEY is None:
        raise ValueError("HUGGINGFACE_API_KEY is not set")

    print("=" * 70)
    print(f"Collecting tokens containing 'e' from: {model_id}")
    print("=" * 70)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=HUGGINGFACE_API_KEY
    )
    print("Tokenizer loaded successfully!")
    print()

    # Get all vocabulary items
    print("Processing vocabulary...")
    vocab = tokenizer.get_vocab()
    total_vocab_size = len(vocab)

    # Collect tokens containing 'e' (case-insensitive)
    tokens_with_e = []
    # Tokens with IDs >= this are likely special
    special_token_threshold = 150000

    print(f"Total vocabulary size: {total_vocab_size}")
    msg = "Filtering for tokens containing 'e' (excluding special tokens)..."
    print(msg)
    print()

    for token, token_id in vocab.items():
        # Skip special tokens
        if is_special_token(token_id, tokenizer, special_token_threshold):
            continue

        # Check if token contains 'e' (case-insensitive)
        if 'e' in token.lower():
            # Decode to get the actual text representation
            try:
                decoded_text = tokenizer.decode([token_id])
                tokens_with_e.append({
                    "token_id": token_id,
                    "token_string": token,
                    "decoded_text": decoded_text
                })
            except Exception as decode_err:
                # If decoding fails, still include the token string
                tokens_with_e.append({
                    "token_id": token_id,
                    "token_string": token,
                    "decoded_text": None,
                    "decode_error": str(decode_err)
                })

    # Sort by token ID
    tokens_with_e.sort(key=lambda x: x["token_id"])

    print(
        f"Found {len(tokens_with_e)} non-special tokens containing 'e'"
    )
    print()

    # Write to JSON file
    output_file = "tokens_with_e.json"
    print(f"Writing results to {output_file}...")

    output_data = {
        "model_id": model_id,
        "total_vocabulary_size": total_vocab_size,
        "tokens_with_e_count": len(tokens_with_e),
        "special_token_threshold": special_token_threshold,
        "tokens": tokens_with_e
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(
        f"Successfully wrote {len(tokens_with_e)} tokens to {output_file}"
    )
    print()

    # Show statistics
    print("=" * 70)
    print("Statistics:")
    print("=" * 70)
    print(f"   Total vocabulary size: {total_vocab_size:,}")
    print(f"   Tokens with 'e' (non-special): {len(tokens_with_e):,}")
    percentage = len(tokens_with_e) / total_vocab_size * 100
    print(f"   Percentage of vocabulary: {percentage:.2f}%")
    print()

    # Show sample tokens
    print("Sample tokens (first 20):")
    for i, token_info in enumerate(tokens_with_e[:20], 1):
        token_str = (
            token_info["token_string"]
            .encode('ascii', 'backslashreplace')
            .decode('ascii')
        )
        decoded = token_info["decoded_text"]
        token_id = token_info['token_id']
        if decoded:
            decoded_str = (
                decoded.encode('ascii', 'backslashreplace')
                .decode('ascii')
            )
            print(
                f"   {i:3d}. ID {token_id:6d}: "
                f"{token_str[:50]:50s} -> '{decoded_str[:30]}'"
            )
        else:
            print(f"   {i:3d}. ID {token_id:6d}: {token_str[:50]}")
    print()

    print("Sample tokens (last 20):")
    start_idx = len(tokens_with_e) - 19
    for i, token_info in enumerate(tokens_with_e[-20:], start_idx):
        token_str = (
            token_info["token_string"]
            .encode('ascii', 'backslashreplace')
            .decode('ascii')
        )
        decoded = token_info["decoded_text"]
        token_id = token_info['token_id']
        if decoded:
            decoded_str = (
                decoded.encode('ascii', 'backslashreplace')
                .decode('ascii')
            )
            print(
                f"   {i:3d}. ID {token_id:6d}: "
                f"{token_str[:50]:50s} -> '{decoded_str[:30]}'"
            )
        else:
            print(f"   {i:3d}. ID {token_id:6d}: {token_str[:50]}")
    print()

    print("=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
