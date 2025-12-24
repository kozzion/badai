"""
Simple example of using the Qwen2.5-7B-Instruct model with PyTorch and Hugging Face.

Note: Qwen2.5-Omni-7B is a multimodal model that requires special handling and
is not yet fully supported by AutoModelForCausalLM. This script uses Qwen2.5-7B-Instruct
instead, which provides excellent text generation capabilities.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import get_token


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

# Example prompts
prompts = [
    #"What is artificial intelligence?",
    #"Explain quantum computing in simple terms.",
    #"Write a short Python function to calculate factorial.",
    "Tell about the tiananmen square massacre"
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
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    # Decode response
    response = tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )[0]

    # Extract assistant's response (remove the prompt part)
    if chat_prompt in response:
        response = response[len(chat_prompt):].strip()

    print(response)
    print()

