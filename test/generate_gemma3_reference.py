#!/usr/bin/env python3
"""
Generate reference test values for Gemma 3 components.
These values are used by the test harness to verify Plan 9 implementations.
"""

import numpy as np
import struct
import json

def gelu_tanh(x):
    """GELU with tanh approximation (matches PyTorch gelu_pytorch_tanh)"""
    c = np.sqrt(2.0 / np.pi)  # 0.7978845608...
    inner = c * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))

def rmsnorm(x, weight, eps=1e-6):
    """RMS normalization"""
    ss = np.mean(x**2) + eps
    return weight * (x / np.sqrt(ss))

def rmsnorm_gemma(x, weight, eps=1e-6):
    """Gemma-style RMS normalization with +1 offset on weight"""
    ss = np.mean(x**2) + eps
    return (1.0 + weight) * (x / np.sqrt(ss))

def softmax(x):
    """Softmax function"""
    max_val = np.max(x)
    exp_x = np.exp(x - max_val)
    return exp_x / np.sum(exp_x)

def rope_freqs(head_dim, max_seq_len, theta=10000.0):
    """Compute RoPE frequency components"""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    return np.cos(freqs), np.sin(freqs)

def apply_rope(q, k, pos, head_dim, theta=10000.0):
    """Apply rotary position embedding to query and key"""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = pos * freqs
    cos_val = np.cos(angles)
    sin_val = np.sin(angles)

    q_out = np.zeros_like(q)
    k_out = np.zeros_like(k)

    for i in range(head_dim // 2):
        q0, q1 = q[2*i], q[2*i+1]
        k0, k1 = k[2*i], k[2*i+1]

        q_out[2*i] = q0 * cos_val[i] - q1 * sin_val[i]
        q_out[2*i+1] = q0 * sin_val[i] + q1 * cos_val[i]

        k_out[2*i] = k0 * cos_val[i] - k1 * sin_val[i]
        k_out[2*i+1] = k0 * sin_val[i] + k1 * cos_val[i]

    return q_out, k_out

def sliding_window_mask(seq_len, window_size):
    """Create sliding window attention mask"""
    mask = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 1.0
    return mask

def attention_pattern_gemma3(layer_idx):
    """Returns True if layer uses local (sliding window) attention"""
    # Pattern: 5 local, 1 global (repeating)
    return (layer_idx % 6) < 5

# ============================================================================
# Generate test values
# ============================================================================

def generate_gelu_test():
    """Generate GELU test values"""
    np.random.seed(42)

    # Test inputs
    inputs = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    outputs = gelu_tanh(inputs)

    # Random test
    random_input = np.random.randn(8).astype(np.float32)
    random_output = gelu_tanh(random_input)

    return {
        "basic_inputs": inputs.tolist(),
        "basic_outputs": outputs.tolist(),
        "random_inputs": random_input.tolist(),
        "random_outputs": random_output.tolist(),
    }

def generate_rmsnorm_gemma_test():
    """Generate Gemma-style RMSNorm test values (with +1 offset)"""
    np.random.seed(43)

    dim = 8
    x = np.random.randn(dim).astype(np.float32)
    weight = np.random.randn(dim).astype(np.float32) * 0.1  # Small weights

    output = rmsnorm_gemma(x, weight)

    return {
        "x": x.tolist(),
        "weight": weight.tolist(),
        "output": output.tolist(),
        "dim": dim,
    }

def generate_rope_dual_theta_test():
    """Generate RoPE test with dual theta values (local vs global)"""
    np.random.seed(44)

    head_dim = 256
    pos = 10

    q = np.random.randn(head_dim).astype(np.float32)
    k = np.random.randn(head_dim).astype(np.float32)

    # Local attention: theta = 10000
    q_local, k_local = apply_rope(q.copy(), k.copy(), pos, head_dim, theta=10000.0)

    # Global attention: theta = 1000000
    q_global, k_global = apply_rope(q.copy(), k.copy(), pos, head_dim, theta=1000000.0)

    return {
        "head_dim": head_dim,
        "pos": pos,
        "q_input": q.tolist(),
        "k_input": k.tolist(),
        "q_local_output": q_local.tolist(),
        "k_local_output": k_local.tolist(),
        "q_global_output": q_global.tolist(),
        "k_global_output": k_global.tolist(),
    }

def generate_sliding_window_test():
    """Generate sliding window attention mask test"""
    seq_len = 16
    window_size = 512  # Gemma 3 270M uses 512

    # For testing, use smaller window
    test_window = 4
    mask = sliding_window_mask(seq_len, test_window)

    return {
        "seq_len": seq_len,
        "window_size": test_window,
        "mask": mask.flatten().tolist(),
    }

def generate_attention_pattern_test():
    """Generate Gemma 3 attention pattern test values"""
    num_layers = 18
    patterns = []
    for i in range(num_layers):
        is_local = attention_pattern_gemma3(i)
        patterns.append({
            "layer": i,
            "is_local": is_local,
            "theta": 10000.0 if is_local else 1000000.0,
        })

    return {
        "num_layers": num_layers,
        "patterns": patterns,
    }

def generate_embedding_scale_test():
    """Generate embedding scaling test (Gemma multiplies by sqrt(dim))"""
    np.random.seed(45)

    dim = 640  # Gemma 3 270M hidden_size
    embeddings = np.random.randn(dim).astype(np.float32)

    scale = np.sqrt(float(dim))
    scaled = embeddings * scale

    return {
        "dim": dim,
        "scale": float(scale),
        "embeddings": embeddings.tolist(),
        "scaled": scaled.tolist(),
    }

def generate_geglu_test():
    """Generate GeGLU activation test (GELU * gate)"""
    np.random.seed(46)

    hidden_dim = 16
    x1 = np.random.randn(hidden_dim).astype(np.float32)  # gate input
    x3 = np.random.randn(hidden_dim).astype(np.float32)  # up input

    # GeGLU: gelu(x1) * x3
    gelu_out = gelu_tanh(x1)
    geglu_out = gelu_out * x3

    return {
        "hidden_dim": hidden_dim,
        "x1": x1.tolist(),
        "x3": x3.tolist(),
        "gelu_x1": gelu_out.tolist(),
        "geglu_output": geglu_out.tolist(),
    }

def generate_qk_norm_test():
    """Generate QK normalization test"""
    np.random.seed(47)

    head_dim = 256
    num_heads = 4

    # Generate Q and K for all heads
    q = np.random.randn(num_heads, head_dim).astype(np.float32)
    k = np.random.randn(1, head_dim).astype(np.float32)  # GQA: 1 KV head

    # Normalize each head independently
    q_norm = np.zeros_like(q)
    k_norm = np.zeros_like(k)

    for h in range(num_heads):
        ss = np.mean(q[h]**2) + 1e-6
        q_norm[h] = q[h] / np.sqrt(ss)

    for h in range(1):  # 1 KV head
        ss = np.mean(k[h]**2) + 1e-6
        k_norm[h] = k[h] / np.sqrt(ss)

    return {
        "head_dim": head_dim,
        "num_q_heads": num_heads,
        "num_kv_heads": 1,
        "q": q.flatten().tolist(),
        "k": k.flatten().tolist(),
        "q_normalized": q_norm.flatten().tolist(),
        "k_normalized": k_norm.flatten().tolist(),
    }

def main():
    """Generate all reference values and write to JSON"""

    print("Generating Gemma 3 reference test values...")

    test_data = {
        "gelu": generate_gelu_test(),
        "rmsnorm_gemma": generate_rmsnorm_gemma_test(),
        "rope_dual_theta": generate_rope_dual_theta_test(),
        "sliding_window": generate_sliding_window_test(),
        "attention_pattern": generate_attention_pattern_test(),
        "embedding_scale": generate_embedding_scale_test(),
        "geglu": generate_geglu_test(),
        "qk_norm": generate_qk_norm_test(),
    }

    # Write to JSON
    with open("gemma3_reference.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print("Written to gemma3_reference.json")

    # Also generate C header with test values
    generate_c_header(test_data)

    print("Generated gemma3_test_data.h")

def generate_c_header(data):
    """Generate C header file with test data"""

    with open("gemma3_test_data.h", "w") as f:
        f.write("/* Auto-generated Gemma 3 test data - DO NOT EDIT */\n")
        f.write("/* Generated by generate_gemma3_reference.py */\n\n")
        f.write("#ifndef GEMMA3_TEST_DATA_H\n")
        f.write("#define GEMMA3_TEST_DATA_H\n\n")

        # GELU test data
        gelu = data["gelu"]
        f.write("/* GELU test data */\n")
        f.write(f"#define GELU_BASIC_N {len(gelu['basic_inputs'])}\n")
        f.write("static float gelu_basic_inputs[] = {" +
                ", ".join(f"{x:.8f}f" for x in gelu["basic_inputs"]) + "};\n")
        f.write("static float gelu_basic_expected[] = {" +
                ", ".join(f"{x:.8f}f" for x in gelu["basic_outputs"]) + "};\n\n")

        f.write(f"#define GELU_RANDOM_N {len(gelu['random_inputs'])}\n")
        f.write("static float gelu_random_inputs[] = {" +
                ", ".join(f"{x:.8f}f" for x in gelu["random_inputs"]) + "};\n")
        f.write("static float gelu_random_expected[] = {" +
                ", ".join(f"{x:.8f}f" for x in gelu["random_outputs"]) + "};\n\n")

        # Gemma RMSNorm test data
        rmsnorm = data["rmsnorm_gemma"]
        f.write("/* Gemma RMSNorm test data (with +1 weight offset) */\n")
        f.write(f"#define RMSNORM_GEMMA_DIM {rmsnorm['dim']}\n")
        f.write("static float rmsnorm_gemma_x[] = {" +
                ", ".join(f"{x:.8f}f" for x in rmsnorm["x"]) + "};\n")
        f.write("static float rmsnorm_gemma_weight[] = {" +
                ", ".join(f"{x:.8f}f" for x in rmsnorm["weight"]) + "};\n")
        f.write("static float rmsnorm_gemma_expected[] = {" +
                ", ".join(f"{x:.8f}f" for x in rmsnorm["output"]) + "};\n\n")

        # GeGLU test data
        geglu = data["geglu"]
        f.write("/* GeGLU test data */\n")
        f.write(f"#define GEGLU_DIM {geglu['hidden_dim']}\n")
        f.write("static float geglu_x1[] = {" +
                ", ".join(f"{x:.8f}f" for x in geglu["x1"]) + "};\n")
        f.write("static float geglu_x3[] = {" +
                ", ".join(f"{x:.8f}f" for x in geglu["x3"]) + "};\n")
        f.write("static float geglu_expected[] = {" +
                ", ".join(f"{x:.8f}f" for x in geglu["geglu_output"]) + "};\n\n")

        # Embedding scale test data
        embed = data["embedding_scale"]
        f.write("/* Embedding scale test data */\n")
        f.write(f"#define EMBED_DIM {embed['dim']}\n")
        f.write(f"#define EMBED_SCALE {embed['scale']:.8f}f\n")
        # Only include first 16 elements for brevity
        f.write("static float embed_input[16] = {" +
                ", ".join(f"{x:.8f}f" for x in embed["embeddings"][:16]) + "};\n")
        f.write("static float embed_expected[16] = {" +
                ", ".join(f"{x:.8f}f" for x in embed["scaled"][:16]) + "};\n\n")

        # QK norm test data (first head only for brevity)
        qk = data["qk_norm"]
        head_dim = qk["head_dim"]
        f.write("/* QK norm test data */\n")
        f.write(f"#define QK_HEAD_DIM {head_dim}\n")
        # First 32 elements of first head
        f.write("static float qk_q_input[32] = {" +
                ", ".join(f"{x:.8f}f" for x in qk["q"][:32]) + "};\n")
        f.write("static float qk_q_expected[32] = {" +
                ", ".join(f"{x:.8f}f" for x in qk["q_normalized"][:32]) + "};\n\n")

        # Attention pattern test
        pattern = data["attention_pattern"]
        f.write("/* Attention pattern test data */\n")
        f.write(f"#define GEMMA3_NUM_LAYERS {pattern['num_layers']}\n")
        f.write("static int gemma3_is_local_layer[] = {" +
                ", ".join("1" if p["is_local"] else "0" for p in pattern["patterns"]) + "};\n")
        f.write("static float gemma3_layer_theta[] = {" +
                ", ".join(f"{p['theta']:.1f}f" for p in pattern["patterns"]) + "};\n\n")

        # RoPE dual theta test (first 8 elements)
        rope = data["rope_dual_theta"]
        f.write("/* RoPE dual theta test data */\n")
        f.write(f"#define ROPE_TEST_POS {rope['pos']}\n")
        f.write("static float rope_q_input[8] = {" +
                ", ".join(f"{x:.8f}f" for x in rope["q_input"][:8]) + "};\n")
        f.write("static float rope_k_input[8] = {" +
                ", ".join(f"{x:.8f}f" for x in rope["k_input"][:8]) + "};\n")
        f.write("static float rope_q_local_expected[8] = {" +
                ", ".join(f"{x:.8f}f" for x in rope["q_local_output"][:8]) + "};\n")
        f.write("static float rope_q_global_expected[8] = {" +
                ", ".join(f"{x:.8f}f" for x in rope["q_global_output"][:8]) + "};\n\n")

        f.write("#endif /* GEMMA3_TEST_DATA_H */\n")

if __name__ == "__main__":
    main()
