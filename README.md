# CUDA
we are doing:
Python 3.11.9 
cuda 1.18
torch-2.7
torch 2.9.1?

# Local Llama 7B Model Runner

A simple Python project to run a 7B Llama model on your local machine using PyTorch and Hugging Face Transformers.

## Prerequisites

- Python 3.8 or higher
- At least 16GB RAM (32GB recommended for better performance)
- 14GB+ free disk space for the model
- Optional: NVIDIA GPU with CUDA support (8GB+ VRAM recommended) for faster inference

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support (CUDA), PyTorch should automatically detect and use CUDA if available. If you need a specific CUDA version:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers accelerate sentencepiece
   ```

2. **Set up Hugging Face authentication (required for gated models like Llama):**

   Most Llama models on Hugging Face are gated and require you to:
   
   a. Accept the license on the model's Hugging Face page:
      - Llama 2 7B Chat: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
      - Llama 2 7B: https://huggingface.co/meta-llama/Llama-2-7b-hf
   
   b. Login to Hugging Face CLI:
      ```bash
      pip install huggingface_hub
      huggingface-cli login
      ```
      Enter your Hugging Face token (get it from https://huggingface.co/settings/tokens)

3. **The model will be automatically downloaded on first run** - no manual download needed!

## Usage

### Basic Usage

```bash
python run_llama.py --model meta-llama/Llama-2-7b-chat-hf --prompt "Tell me about Python programming"
```

### Interactive Chat Mode

For conversational use:

```bash
python chat.py --model meta-llama/Llama-2-7b-chat-hf
```

### Command Line Options

**run_llama.py:**
- `--model`: Hugging Face model ID or local path (default: `meta-llama/Llama-2-7b-chat-hf`)
- `--prompt`: The prompt to send to the model
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 256)
- `--temperature`: Sampling temperature 0.0-1.0 (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--device`: Device to use - `auto`, `cpu`, or `cuda` (default: `auto`)

**chat.py:**
- Same options as above (except `--prompt` since it's interactive)

### Example Models

You can use any Llama model from Hugging Face. Popular options:

- `meta-llama/Llama-2-7b-chat-hf` - Chat-optimized version (recommended for conversations)
- `meta-llama/Llama-2-7b-hf` - Base model
- `meta-llama/Llama-3.1-8B-Instruct` - Newer Llama 3 model (8B)
- `mistralai/Mistral-7B-v0.1` - Alternative 7B model

### Using Local Models

If you've already downloaded a model, you can use the local path:

```bash
python run_llama.py --model ./models/llama-2-7b-chat-hf --prompt "Hello"
```

### Python API Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

prompt = "<s>[INST] What is AI? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## GPU Acceleration

The scripts automatically detect and use GPU if available. To force CPU usage:

```bash
python run_llama.py --device cpu --model meta-llama/Llama-2-7b-chat-hf --prompt "Hello"
```

## Memory Optimization

If you're running out of memory, try:

1. **Use CPU instead of GPU:**
   ```bash
   --device cpu
   ```

2. **Reduce max tokens:**
   ```bash
   --max-new-tokens 128
   ```

3. **Use 8-bit quantization (requires `bitsandbytes`):**
   ```python
   from transformers import BitsAndBytesConfig
   
   quantization_config = BitsAndBytesConfig(load_in_8bit=True)
   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       quantization_config=quantization_config,
   )
   ```

## Troubleshooting

**Out of Memory Error:**
- Use `--device cpu` (slower but uses less memory)
- Reduce `--max-new-tokens`
- Close other applications to free up RAM/VRAM
- Consider using a smaller model or quantization

**Model Access Denied:**
- Make sure you've accepted the license on Hugging Face
- Run `huggingface-cli login` and use your access token
- Check that your Hugging Face account has access to the model

**Slow Performance:**
- Use GPU if available (`--device cuda`)
- Reduce `--max-new-tokens` for faster responses
- Consider using a quantized model variant

**Import Errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- For GPU support, ensure PyTorch with CUDA is installed correctly

## Notes

- First run will download the model (~13GB for 7B model), which may take time
- Models are cached in `~/.cache/huggingface/` after first download
- Chat models use special formatting tokens (`<s>`, `[INST]`, `[/INST]`) - the scripts handle this automatically
- For best results with chat models, use the `chat.py` script which properly formats conversations
