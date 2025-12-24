"""
Script to list tokens and tokenizer information for Qwen models.
"""

import os
from transformers import AutoTokenizer
from huggingface_hub import get_token


def main():
    # Model ID - same as example_qwen.py
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    # Get Hugging Face token for authentication
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or get_token()

    if HUGGINGFACE_API_KEY is None:
        raise ValueError("HUGGINGFACE_API_KEY is not set")

    print("=" * 60)
    print(f"Tokenizer Information for: {model_id}")
    print("=" * 60)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_API_KEY)
    print("Tokenizer loaded successfully!")
    print()

    # Basic tokenizer info
    print("1. Basic Tokenizer Information:")
    print(f"   Vocabulary size: {len(tokenizer)}")
    print(f"   Model max length: {tokenizer.model_max_length}")
    print(f"   Padding side: {tokenizer.padding_side}")
    print(f"   Truncation side: {tokenizer.truncation_side}")
    print()

    # Special tokens
    print("2. Special Tokens:")
    print(f"   BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token:
        print(f"   SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token:
        print(f"   CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token:
        print(f"   MASK token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    print()

    # Chat template tokens (Qwen-specific)
    print("3. Chat Template Information:")
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("   Chat template: Available")
        # Try to find special tokens in chat template
        template_str = str(tokenizer.chat_template)
        if 'im_start' in template_str or 'im_end' in template_str:
            print("   Uses Qwen chat format with <im_start> and <im_end> tokens")
    else:
        print("   Chat template: Not available")
    print()

    # Check for Qwen-specific special tokens
    print("4. Qwen-Specific Tokens:")
    qwen_special_tokens = []
    if hasattr(tokenizer, 'added_tokens_decoder'):
        for token_id, token_obj in tokenizer.added_tokens_decoder.items():
            if token_id >= len(tokenizer) - 100:  # Check last 100 tokens
                qwen_special_tokens.append((token_id, str(token_obj)))
    
    # Common Qwen tokens to check
    test_tokens = [
        '<|im_start|>', '<|im_end|>', '<|endoftext|>',
        '<|user|>', '<|assistant|>', '<|system|>',
        '<|think|>', '<|tool|>', '<|tool_call|>', '<|tool_result|>'
    ]
    
    found_tokens = []
    for token_str in test_tokens:
        try:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                found_tokens.append((token_str, token_ids[0]))
        except:
            pass
    
    if found_tokens:
        print("   Found Qwen special tokens:")
        for token_str, token_id in found_tokens:
            print(f"      {token_str}: ID {token_id}")
    else:
        print("   (Checking tokenizer vocabulary for special tokens...)")
    print()

    # Sample vocabulary tokens
    print("5. Sample Vocabulary Tokens (first 20):")
    vocab_items = list(tokenizer.get_vocab().items())[:20]
    for token, token_id in vocab_items:
        try:
            token_repr = repr(token).encode('ascii', 'backslashreplace').decode('ascii')
            print(f"   ID {token_id:6d}: {token_repr}")
        except:
            print(f"   ID {token_id:6d}: <token>")
    print()

    print("6. Sample Vocabulary Tokens (last 20):")
    vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])[-20:]
    for token, token_id in vocab_items:
        try:
            token_repr = repr(token).encode('ascii', 'backslashreplace').decode('ascii')
            print(f"   ID {token_id:6d}: {token_repr}")
        except:
            print(f"   ID {token_id:6d}: <token>")
    print()

    # Test encoding/decoding
    print("7. Token Encoding Example:")
    test_text = "Hello, how are you?"
    encoded = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    print(f"   Input text: {test_text}")
    print(f"   Token IDs: {encoded[:10]}..." if len(encoded) > 10 else f"   Token IDs: {encoded}")
    print(f"   Decoded text: {decoded}")
    print()

    # Show token breakdown
    print("8. Token Breakdown for Example Text:")
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
        try:
            token_repr = repr(token).encode('ascii', 'backslashreplace').decode('ascii')
            print(f"   Token {i+1}: {token_repr:30s} -> ID {token_id}")
        except:
            print(f"   Token {i+1}: <token> -> ID {token_id}")
    print()

    # Chat template example
    print("9. Chat Template Example:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"}
    ]
    try:
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("   Messages:")
        for msg in messages:
            print(f"      {msg['role']}: {msg['content']}")
        print()
        print("   Formatted chat text:")
        print(f"      {repr(chat_text[:200])}..." if len(chat_text) > 200 else f"      {repr(chat_text)}")
        print()
        
        # Show tokens in chat format
        chat_tokens = tokenizer.tokenize(chat_text)
        chat_ids = tokenizer.convert_tokens_to_ids(chat_tokens)
        print("   First 15 tokens in chat format:")
        for i, (token, token_id) in enumerate(zip(chat_tokens[:15], chat_ids[:15])):
            try:
                token_repr = repr(token).encode('ascii', 'backslashreplace').decode('ascii')
                print(f"      {i+1:2d}: {token_repr:35s} -> ID {token_id}")
            except:
                print(f"      {i+1:2d}: <token> -> ID {token_id}")
    except Exception as e:
        print(f"   Error applying chat template: {e}")
    print()

    print("=" * 60)
    print("Summary:")
    print(f"   Total vocabulary size: {len(tokenizer)}")
    print(f"   EOS token ID (used for stopping generation): {tokenizer.eos_token_id}")
    print(f"   BOS token ID: {tokenizer.bos_token_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()

