"""
Simple example of using the Llama model with PyTorch and Hugging Face.
"""

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


# Model ID - change this to your preferred model
model_id = "meta-llama/Llama-2-7b-chat-hf"

# Get Hugging Face token for authentication
# Option 1: From environment variable (set HF_TOKEN=your_token)
# Option 2: From huggingface_hub login (huggingface-cli login)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or get_token()

if HUGGINGFACE_API_KEY is None:
    raise ValueError("HUGGINGFACE_API_KEY is not set")

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Loading model: {model_id}")
print("This may take a moment on first run (model will be downloaded)...\n")

# Load tokenizer and model with authentication
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_API_KEY)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HUGGINGFACE_API_KEY,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
)

if device == "cpu":
    model = model.to(device)

model.eval()

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

    # Format for chat
    chat_prompt = format_chat_prompt(prompt)

    # Tokenize
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant's response
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    else:
        if response.startswith(chat_prompt):
            response = response[len(chat_prompt):].strip()

    print(response)
    print()
