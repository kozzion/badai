"""
Simple example of using the Qwen2.5-7B-Instruct model with PyTorch and
Hugging Face. This script modifies the model to suppress tokens containing
the letter 'e'.

Note: Qwen2.5-Omni-7B is a multimodal model that requires special handling
and is not yet fully supported by AutoModelForCausalLM. This script uses
Qwen2.5-7B-Instruct instead, which provides excellent text generation
capabilities.
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
)
from huggingface_hub import get_token


class SuppressTokensLogitsProcessor(LogitsProcessor):
    """Logits processor that suppresses specific token IDs."""

    def __init__(self, token_ids_to_suppress):
        """
        Args:
            token_ids_to_suppress: Set or list of token IDs to suppress
        """
        self.token_ids_to_suppress = set(token_ids_to_suppress)

    def __call__(self, input_ids, scores):
        # Set logits for suppressed tokens to negative infinity
        scores[:, list(self.token_ids_to_suppress)] = float('-inf')
        return scores


# Model ID - Qwen2.5-7B-Instruct (text-only, well supported)
# For Qwen2.5-Omni-7B (multimodal), check for future transformers updates
model_id = "Qwen/Qwen2.5-7B-Instruct"

# Get Hugging Face token for authentication
# Option 1: From environment variable (set HUGGINGFACE_API_KEY=your_token)
# Option 2: From huggingface_hub login (huggingface-cli login)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or get_token()

if HUGGINGFACE_API_KEY is None:
    raise ValueError("HUGGINGFACE_API_KEY is not set")

# Determine device

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    # LLMs only run on GPU
    raise ValueError("CUDA is not available")
print(f"Loading model: {model_id}")
print("This may take a moment on first run (model will be downloaded)...\n")
device = "cuda"

# Load tokenizer and model with authentication
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_API_KEY)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HUGGINGFACE_API_KEY,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
)

if device == "cpu":
    model = model.to(device)

model.eval()

# Load tokens with 'e' from JSON file
print("\nLoading tokens with 'e' to suppress...")
tokens_file = "tokens_with_e.json"
try:
    with open(tokens_file, 'r', encoding='utf-8') as f:
        tokens_data = json.load(f)

    # Extract token IDs that contain 'e'
    token_ids_with_e = [
        token_info["token_id"] for token_info in tokens_data["tokens"]
    ]
    print(
        f"Found {len(token_ids_with_e)} tokens containing 'e' to suppress"
    )

    # Create logits processor to suppress tokens with 'e'
    print("Creating logits processor to suppress tokens...")

    # Filter valid token IDs
    vocab_size = tokenizer.vocab_size
    valid_token_ids = [
        token_id for token_id in token_ids_with_e
        if 0 <= token_id < vocab_size
    ]

    # Create logits processor
    logits_processor = SuppressTokensLogitsProcessor(valid_token_ids)

    print(
        f"Will suppress {len(valid_token_ids)} tokens during generation"
    )
    print(f"First 10 token IDs to suppress: {valid_token_ids[:10]}")
    print(f"\nToken suppression configured: {len(valid_token_ids)} tokens")

except FileNotFoundError:
    logits_processor = None
    print(f"Warning: {tokens_file} not found. Token suppression disabled.")
    print("Run list_tokens_with_e.py first to generate the tokens file.")

except Exception as e:
    logits_processor = None
    print(f"Error loading tokens: {e}")
    print("Token suppression disabled.")

# Example prompts
prompts = [
    "What is artificial intelligence?",
    "Explain quantum computing in simple terms.",
    "Write a short Python function to calculate factorial.",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    # Format for Qwen chat (using apply_chat_template)
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer([chat_prompt], return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        generate_kwargs = {
            **inputs,
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }

        # Add logits processor if we loaded the tokens
        if logits_processor is not None:
            logits_processors = [logits_processor]
            generate_kwargs["logits_processor"] = logits_processors

        outputs = model.generate(**generate_kwargs)

    # Decode response
    response = tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )[0]

    # Extract assistant's response (remove the prompt part)
    if chat_prompt in response:
        response = response[len(chat_prompt):].strip()

    print(response)
    print()
