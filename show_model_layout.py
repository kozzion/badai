"""
Script to print the model architecture as JSON.
Starts with Qwen2.5-7B-Instruct model.
"""

import os
import json
from transformers import AutoConfig
from huggingface_hub import get_token


def config_to_dict(config):
    """
    Convert a model config to a dictionary, handling nested objects.
    """
    if hasattr(config, 'to_dict'):
        return config.to_dict()
    else:
        # Fallback: convert to dict manually
        result = {}
        for key in dir(config):
            if not key.startswith('_'):
                try:
                    value = getattr(config, key)
                    if not callable(value):
                        if hasattr(value, 'to_dict'):
                            result[key] = value.to_dict()
                        else:
                            # Try to convert basic types
                            basic_types = (int, float, str, bool, type(None))
                            if isinstance(value, basic_types):
                                result[key] = value
                            elif isinstance(value, (list, tuple)):
                                result[key] = list(value)
                            elif isinstance(value, dict):
                                result[key] = value
                            else:
                                result[key] = str(value)
                except Exception:
                    pass
        return result


def build_layer_architecture(config_dict):
    """
    Build a detailed layer-by-layer architecture breakdown.
    """
    hidden_size = config_dict.get('hidden_size', 0)
    inter_size = config_dict.get('intermediate_size', 0)
    num_layers = config_dict.get('num_hidden_layers', 0)
    num_heads = config_dict.get('num_attention_heads', 0)
    num_kv_heads = config_dict.get('num_key_value_heads', 0)
    vocab_size = config_dict.get('vocab_size', 0)
    max_pos = config_dict.get('max_position_embeddings', 0)

    architecture = {
        "embeddings": {
            "token_embedding": {
                "description": "Token/Word Embedding",
                "weight_shape": [vocab_size, hidden_size],
                "parameters": vocab_size * hidden_size
            },
            "position_embedding": {
                "description": "Position Embedding (RoPE)",
                "max_positions": max_pos,
                "hidden_size": hidden_size,
                "rope_theta": config_dict.get('rope_theta', None)
            }
        },
        "layers": []
    }

    # Calculate attention dimensions
    head_dim = hidden_size // num_heads if num_heads > 0 else 0
    kv_dim = num_kv_heads * head_dim if num_kv_heads > 0 else 0

    # Build each transformer layer
    for layer_idx in range(num_layers):
        layer_type = "full_attention"
        if 'layer_types' in config_dict:
            if isinstance(config_dict['layer_types'], list):
                if layer_idx < len(config_dict['layer_types']):
                    layer_type = config_dict['layer_types'][layer_idx]

        # Attention module
        if num_kv_heads == num_heads or num_kv_heads == 0:
            # Standard multi-head attention
            attn_params = 4 * hidden_size * hidden_size  # Q, K, V, O
            q_shape = [hidden_size, hidden_size]
            k_shape = [hidden_size, hidden_size]
            v_shape = [hidden_size, hidden_size]
            o_shape = [hidden_size, hidden_size]
        else:
            # Grouped Query Attention (GQA)
            attn_params = (
                2 * hidden_size * hidden_size +  # Q, O
                2 * hidden_size * kv_dim  # K, V (shared)
            )
            q_shape = [hidden_size, hidden_size]
            k_shape = [hidden_size, kv_dim]
            v_shape = [hidden_size, kv_dim]
            o_shape = [hidden_size, hidden_size]

        # FFN module
        ffn_params = 2 * hidden_size * inter_size  # up, down

        # Layer normalization parameters (per layer)
        ln_params = 2 * hidden_size  # weight + bias

        layer = {
            "layer_index": layer_idx,
            "layer_type": layer_type,
            "attention": {
                "input_norm": {
                    "type": "RMSNorm",
                    "shape": [hidden_size],
                    "parameters": hidden_size
                },
                "query_projection": {
                    "weight_shape": q_shape,
                    "bias": False,
                    "parameters": q_shape[0] * q_shape[1]
                },
                "key_projection": {
                    "weight_shape": k_shape,
                    "bias": False,
                    "parameters": k_shape[0] * k_shape[1]
                },
                "value_projection": {
                    "weight_shape": v_shape,
                    "bias": False,
                    "parameters": v_shape[0] * v_shape[1]
                },
                "output_projection": {
                    "weight_shape": o_shape,
                    "bias": False,
                    "parameters": o_shape[0] * o_shape[1]
                },
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
                "attention_dropout": config_dict.get('attention_dropout', 0.0),
                "total_parameters": attn_params
            },
            "feed_forward": {
                "input_norm": {
                    "type": "RMSNorm",
                    "shape": [hidden_size],
                    "parameters": hidden_size
                },
                "gate_projection": {
                    "weight_shape": [hidden_size, inter_size],
                    "bias": False,
                    "parameters": hidden_size * inter_size
                },
                "up_projection": {
                    "weight_shape": [hidden_size, inter_size],
                    "bias": False,
                    "parameters": hidden_size * inter_size
                },
                "down_projection": {
                    "weight_shape": [inter_size, hidden_size],
                    "bias": False,
                    "parameters": inter_size * hidden_size
                },
                "activation": config_dict.get('hidden_act', 'silu'),
                "total_parameters": ffn_params
            },
            "residual_connections": [
                {
                    "from": "input",
                    "to": "attention_output",
                    "description": "Residual connection around attention"
                },
                {
                    "from": "attention_output",
                    "to": "ffn_output",
                    "description": "Residual connection around FFN"
                }
            ],
            "total_parameters_per_layer": (
                attn_params + ffn_params + 2 * ln_params
            )
        }

        architecture["layers"].append(layer)

    # Output layer norm
    architecture["output_norm"] = {
        "type": "RMSNorm",
        "shape": [hidden_size],
        "parameters": hidden_size
    }

    # Language modeling head
    architecture["lm_head"] = {
        "description": "Language Modeling Head",
        "weight_shape": [hidden_size, vocab_size],
        "parameters": hidden_size * vocab_size,
        "tied_embeddings": config_dict.get('tie_word_embeddings', False)
    }

    # Calculate total parameters
    embedding_params = vocab_size * hidden_size
    layer_params = sum(
        layer["total_parameters_per_layer"]
        for layer in architecture["layers"]
    )
    output_norm_params = architecture["output_norm"]["parameters"]
    lm_head_params = (
        0 if architecture["lm_head"]["tied_embeddings"]
        else architecture["lm_head"]["parameters"]
    )

    architecture["parameter_summary"] = {
        "embeddings": embedding_params,
        "layers": layer_params,
        "output_norm": output_norm_params,
        "lm_head": lm_head_params,
        "total_parameters": (
            embedding_params + layer_params +
            output_norm_params + lm_head_params
        ),
        "total_parameters_billions": (
            embedding_params + layer_params +
            output_norm_params + lm_head_params
        ) / 1e9
    }

    return architecture


def main():
    # Model ID - Qwen2.5-7B-Instruct
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    # Get Hugging Face token for authentication
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or get_token()

    if HUGGINGFACE_API_KEY is None:
        raise ValueError("HUGGINGFACE_API_KEY is not set")

    print("=" * 70)
    print(f"Loading model configuration for: {model_id}")
    print("=" * 70)
    print()

    # Load model configuration
    print("Loading configuration...")
    config = AutoConfig.from_pretrained(model_id, token=HUGGINGFACE_API_KEY)
    print("Configuration loaded successfully!")
    print()

    # Convert to dictionary
    print("Converting configuration to JSON...")
    config_dict = config_to_dict(config)

    # Build layer-by-layer architecture
    print("Building layer-by-layer architecture...")
    architecture = build_layer_architecture(config_dict)
    config_dict["layer_by_layer_architecture"] = architecture

    # Print as JSON
    print()
    print("=" * 70)
    print("MODEL ARCHITECTURE (JSON):")
    print("=" * 70)
    print()
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    print()

    # Also save to file
    output_file = "model_architecture.json"
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"Architecture saved to {output_file}")
    print()

    # Print summary with layer dimensions
    print("=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"   Model Type: {config_dict.get('model_type', 'N/A')}")
    print()
    print("LAYER DIMENSIONS:")
    print("-" * 70)
    hidden_size = config_dict.get('hidden_size', 'N/A')
    print(f"   Hidden Size (d_model): {hidden_size}")
    inter_size = config_dict.get('intermediate_size', 'N/A')
    print(f"   Intermediate Size (FFN): {inter_size}")
    num_layers = config_dict.get('num_hidden_layers', 'N/A')
    print(f"   Number of Layers: {num_layers}")
    num_heads = config_dict.get('num_attention_heads', 'N/A')
    print(f"   Number of Attention Heads: {num_heads}")
    num_kv_heads = config_dict.get('num_key_value_heads', 'N/A')
    print(f"   Number of Key-Value Heads: {num_kv_heads}")

    # Calculate attention head dimensions
    if isinstance(hidden_size, int) and isinstance(num_heads, int):
        head_dim = hidden_size // num_heads
        print(f"   Attention Head Dimension: {head_dim}")
        print(
            f"   Query Dimension: {hidden_size} "
            f"({num_heads} heads × {head_dim})"
        )
        if isinstance(num_kv_heads, int):
            kv_dim = num_kv_heads * head_dim
            print(
                f"   Key-Value Dimension: {kv_dim} "
                f"({num_kv_heads} heads × {head_dim})"
            )

    print()
    print("EMBEDDING DIMENSIONS:")
    print("-" * 70)
    vocab_size = config_dict.get('vocab_size', 'N/A')
    print(f"   Vocabulary Size: {vocab_size}")
    if isinstance(hidden_size, int) and isinstance(vocab_size, int):
        print(f"   Token Embedding: [{vocab_size}, {hidden_size}]")
    max_pos = config_dict.get('max_position_embeddings', 'N/A')
    print(f"   Max Position Embeddings: {max_pos}")
    if isinstance(hidden_size, int) and isinstance(max_pos, int):
        print(f"   Position Embedding: [{max_pos}, {hidden_size}]")

    print()
    print("ATTENTION DIMENSIONS:")
    print("-" * 70)
    if isinstance(hidden_size, int) and isinstance(num_heads, int):
        head_dim = hidden_size // num_heads
        print(f"   Attention Input: [batch, seq_len, {hidden_size}]")
        print(f"   Query Weight: [{hidden_size}, {hidden_size}]")
        if isinstance(num_kv_heads, int):
            kv_str = f"{num_kv_heads} × {head_dim}"
        else:
            kv_str = 'N/A'
        print(f"   Key Weight: [{hidden_size}, {kv_str}]")
        print(f"   Value Weight: [{hidden_size}, {kv_str}]")
        print(f"   Output Weight: [{hidden_size}, {hidden_size}]")

    print()
    print("FEED-FORWARD NETWORK DIMENSIONS:")
    print("-" * 70)
    if isinstance(hidden_size, int) and isinstance(inter_size, int):
        print(f"   FFN Input: [batch, seq_len, {hidden_size}]")
        print(f"   Up Projection: [{hidden_size}, {inter_size}]")
        print(f"   Down Projection: [{inter_size}, {hidden_size}]")
        print(f"   FFN Output: [batch, seq_len, {hidden_size}]")

    print()
    print("LAYER STACK DIMENSIONS:")
    print("-" * 70)
    if isinstance(num_layers, int) and isinstance(hidden_size, int):
        print("   Total Parameters (approx):")
        # Rough estimate: embeddings + layers * (attention + ffn)
        embedding_params = (
            vocab_size * hidden_size if isinstance(vocab_size, int) else 0
        )
        if isinstance(max_pos, int):
            embedding_params += max_pos * hidden_size
        # Attention params per layer
        attn_params = 4 * hidden_size * hidden_size  # Q, K, V, O
        if isinstance(num_kv_heads, int) and num_kv_heads != num_heads:
            # Adjust for GQA
            kv_dim = num_kv_heads * (hidden_size // num_heads)
            qo_params = 2 * hidden_size * hidden_size
            kv_params = 2 * hidden_size * kv_dim
            attn_params = qo_params + kv_params
        # FFN params per layer
        ffn_params = (
            2 * hidden_size * inter_size if isinstance(inter_size, int) else 0
        )
        # Layer norm params (small, ~2 * hidden_size per layer)
        ln_params = 2 * hidden_size * num_layers
        total_params = (
            embedding_params +
            num_layers * (attn_params + ffn_params) + ln_params
        )
        print(f"   Embeddings: ~{embedding_params:,} params")
        per_layer = attn_params + ffn_params
        print(f"   Per Layer (Attention + FFN): ~{per_layer:,} params")
        print(f"   Total (approximate): ~{total_params:,} params")
        billions = total_params / 1e9
        print(f"   Total (billions): ~{billions:.2f}B params")

    print("=" * 70)


if __name__ == "__main__":
    main()
