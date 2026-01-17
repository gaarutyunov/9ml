#!/usr/bin/env python3
"""
Automated test suite for 9ml
Compares Plan 9 output against Python reference implementations
"""

import subprocess
import os
import sys
import math

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QEMU_DIR = os.path.join(PROJECT_DIR, "qemu")
SRC_DIR = os.path.join(PROJECT_DIR, "src")
TESTS_DIR = os.path.join(SRC_DIR, "tests")
MODEL_PATH = "/Users/germanarutyunov/Projects/plan9/llama2.c/stories15M.bin"
TOKENIZER_PATH = "/Users/germanarutyunov/Projects/plan9/llama2.c/tokenizer.bin"

EPSILON = 0.0001

def run_plan9_test(test_name, model_file="model.c", extra_files=None, timeout=300):
    """Compile and run a test file in Plan 9, return output lines"""
    test_file = os.path.join(TESTS_DIR, f"test_{test_name}.c")
    model_path = os.path.join(SRC_DIR, model_file)

    # Build list of files to copy
    files_to_copy = [test_file, model_path]
    if extra_files:
        files_to_copy.extend(extra_files)

    # Copy files to shared disk
    subprocess.run([os.path.join(QEMU_DIR, "copy-to-shared.sh")] + files_to_copy,
                   capture_output=True, timeout=60)

    # Compile and run in Plan 9 (cd to /mnt/host so includes work)
    result = subprocess.run(
        [os.path.join(QEMU_DIR, "run-cmd.sh"),
         f"cd /mnt/host && 6c -w test_{test_name}.c && 6l -o test_{test_name} test_{test_name}.6 && ./test_{test_name}"],
        capture_output=True, text=True, timeout=timeout
    )

    # Parse output - filter to lines that look like numbers or key=value
    lines = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        try:
            float(line)
            lines.append(line)
        except ValueError:
            # Check for key=value format
            if '=' in line and not line.startswith('6c') and not line.startswith('6l'):
                lines.append(line)
    return lines

def compare_floats(ref, plan9, eps=EPSILON):
    """Compare two lists of floats within epsilon"""
    if len(ref) != len(plan9):
        return False, f"Length mismatch: ref={len(ref)}, plan9={len(plan9)}"

    for i, (r, p) in enumerate(zip(ref, plan9)):
        if abs(r - p) > eps:
            return False, f"Value mismatch at [{i}]: ref={r}, plan9={p}, diff={abs(r-p)}"
    return True, "OK"

def test_rmsnorm():
    """Test rmsnorm function"""
    print("Testing rmsnorm...", end=" ", flush=True)

    # Python reference
    x = [1.0, 2.0, 3.0, 4.0]
    w = [0.5, 0.5, 0.5, 0.5]
    ss = sum(v * v for v in x) / len(x)
    ss += 1e-5
    ss = 1.0 / math.sqrt(ss)
    ref = [wi * (ss * xi) for wi, xi in zip(w, x)]

    # Run Plan 9 test
    plan9_out = run_plan9_test("rmsnorm")
    plan9 = [float(x) for x in plan9_out]

    ok, msg = compare_floats(ref, plan9)
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        print(f"  Reference: {ref}")
        print(f"  Plan 9:    {plan9}")
        return False

def test_softmax():
    """Test softmax function"""
    print("Testing softmax...", end=" ", flush=True)

    # Python reference
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    max_val = max(x)
    exp_x = [math.exp(v - max_val) for v in x]
    total = sum(exp_x)
    ref = [v / total for v in exp_x]

    # Run Plan 9 test
    plan9_out = run_plan9_test("softmax")
    plan9 = [float(x) for x in plan9_out]

    ok, msg = compare_floats(ref, plan9)
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        return False

def test_matmul():
    """Test matmul function"""
    print("Testing matmul...", end=" ", flush=True)

    # Python reference
    w = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    x = [1, 2, 3, 4]
    ref = [sum(r * xi for r, xi in zip(row, x)) for row in w]

    # Run Plan 9 test
    plan9_out = run_plan9_test("matmul")
    plan9 = [float(x) for x in plan9_out]

    ok, msg = compare_floats(ref, plan9)
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        return False

def test_model_loading():
    """Test that model config and weights are loaded correctly"""
    print("Testing model loading...", end=" ", flush=True)

    if not os.path.exists(MODEL_PATH):
        print("SKIP (model not found)")
        return None

    # Python reference - read config and first 10 weights
    import struct
    with open(MODEL_PATH, 'rb') as f:
        config = struct.unpack('7i', f.read(28))
        weights = struct.unpack('10f', f.read(40))

    ref_config = {
        'dim': config[0],
        'hidden_dim': config[1],
        'n_layers': config[2],
        'n_heads': config[3],
        'n_kv_heads': config[4],
        'vocab_size': abs(config[5]),  # Can be negative
        'seq_len': config[6]
    }
    ref_weights = list(weights)

    # Copy model to shared disk and run test
    plan9_out = run_plan9_test("model_loading", extra_files=[MODEL_PATH])

    # Parse Plan 9 output
    plan9_config = {}
    plan9_weights = []
    for line in plan9_out:
        if '=' in line:
            key, val = line.split('=', 1)
            if key.startswith('w'):
                plan9_weights.append(float(val))
            else:
                plan9_config[key] = int(val)

    # Compare config
    config_ok = True
    for key in ref_config:
        if plan9_config.get(key) != ref_config[key]:
            print(f"FAIL: config.{key} mismatch: ref={ref_config[key]}, plan9={plan9_config.get(key)}")
            config_ok = False

    if not config_ok:
        return False

    # Compare weights
    ok, msg = compare_floats(ref_weights, plan9_weights)
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        return False

def test_rng():
    """Test random number generator"""
    print("Testing RNG...", end=" ", flush=True)

    # Python reference (xorshift)
    def random_u32(state):
        state ^= state >> 12
        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 27
        return state, ((state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    state = 42
    ref = []
    for _ in range(10):
        state, val = random_u32(state)
        ref.append(val)

    # Run Plan 9 test
    plan9_out = run_plan9_test("rng")
    plan9 = [int(x) for x in plan9_out]

    if ref == plan9:
        print("PASS")
        return True
    else:
        print(f"FAIL")
        print(f"  Reference: {ref}")
        print(f"  Plan 9:    {plan9}")
        return False

def test_generation():
    """Test end-to-end generation matches between host and Plan 9"""
    print("Testing generation...", end=" ", flush=True)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("SKIP (model or tokenizer not found)")
        return None

    # Run host version (stdout has generated text, stderr has timing info)
    host_binary = os.path.join(os.path.dirname(MODEL_PATH), "run")
    host_result = subprocess.run(
        [host_binary, MODEL_PATH, "-z", TOKENIZER_PATH, "-n", "20", "-s", "42", "-i", "Once upon a time"],
        capture_output=True, text=True, timeout=60
    )
    host_output = host_result.stdout.strip()

    # Copy files to shared disk
    subprocess.run([os.path.join(QEMU_DIR, "copy-to-shared.sh"),
                    os.path.join(SRC_DIR, "run.c"),
                    os.path.join(SRC_DIR, "model.c"),
                    MODEL_PATH, TOKENIZER_PATH],
                   capture_output=True, timeout=120)

    # Compile and run in Plan 9, redirect output to file
    subprocess.run(
        [os.path.join(QEMU_DIR, "run-cmd.sh"),
         "cd /mnt/host && 6c -w run.c && 6l -o run run.6 && ./run stories15M.bin -z tokenizer.bin -n 20 -s 42 -i 'Once upon a time' > output.txt"],
        capture_output=True, text=True, timeout=300
    )

    # Read output file from shared disk
    mount_point = subprocess.run(["mktemp", "-d", "/tmp/shared_mount.XXXXXX"],
                                  capture_output=True, text=True).stdout.strip()
    subprocess.run(["hdiutil", "attach", "-imagekey", "diskimage-class=CRawDiskImage",
                    "-mountpoint", mount_point,
                    os.path.join(QEMU_DIR, "shared.img")],
                   capture_output=True)

    output_file = os.path.join(mount_point, "output.txt")
    plan9_output = ""
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            plan9_output = f.read().strip()

    subprocess.run(["hdiutil", "detach", mount_point], capture_output=True)
    subprocess.run(["rmdir", mount_point], capture_output=True)

    if host_output == plan9_output:
        print("PASS")
        return True
    else:
        print(f"FAIL")
        print(f"  Host:   {host_output}")
        print(f"  Plan 9: {plan9_output}")
        return False

def test_quantize():
    """Test quantize/dequantize roundtrip"""
    print("Testing quantize...", end=" ", flush=True)

    # Python reference - quantize then dequantize
    GS = 32  # group size
    x = [float(i) / 10.0 for i in range(64)]  # 64 floats, 2 groups

    # Quantize
    q = []
    s = []
    for group in range(len(x) // GS):
        group_vals = x[group * GS:(group + 1) * GS]
        wmax = max(abs(v) for v in group_vals)
        scale = wmax / 127.0
        s.append(scale)
        for v in group_vals:
            q.append(round(v / scale) if scale > 0 else 0)

    # Dequantize
    ref = []
    for i in range(len(q)):
        ref.append(q[i] * s[i // GS])

    # Run Plan 9 test with modelq.c
    plan9_out = run_plan9_test("quantize", model_file="modelq.c")
    plan9 = [float(x) for x in plan9_out]

    ok, msg = compare_floats(ref, plan9, eps=0.1)  # Larger epsilon for quantization error
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        return False

def test_quantized_matmul():
    """Test quantized matmul"""
    print("Testing quantized matmul...", end=" ", flush=True)

    # Python reference - quantized matmul
    GS = 32
    n, d = 64, 2  # 64 input, 2 output

    # Input vector
    x = [float(i % 10) / 10.0 for i in range(n)]
    # Weight matrix (flattened, d x n)
    w = [float((i // n + i % n) % 10) / 10.0 for i in range(d * n)]

    def py_quantize(vals):
        q = []
        s = []
        for group in range(len(vals) // GS):
            group_vals = vals[group * GS:(group + 1) * GS]
            wmax = max(abs(v) for v in group_vals)
            scale = wmax / 127.0 if wmax > 0 else 1.0
            s.append(scale)
            for v in group_vals:
                q.append(round(v / scale))
        return q, s

    xq, xs = py_quantize(x)

    # Compute reference matmul
    ref = []
    for i in range(d):
        row_start = i * n
        wq, ws = py_quantize(w[row_start:row_start + n])

        val = 0.0
        for j in range(0, n, GS):
            ival = sum(xq[j + k] * wq[j + k] for k in range(GS))
            val += float(ival) * ws[j // GS] * xs[j // GS]
        ref.append(val)

    # Run Plan 9 test with modelq.c
    plan9_out = run_plan9_test("quantized_matmul", model_file="modelq.c")
    plan9 = [float(x) for x in plan9_out]

    ok, msg = compare_floats(ref, plan9, eps=0.5)  # Larger epsilon for quantization error
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        print(f"  Reference: {ref}")
        print(f"  Plan 9:    {plan9}")
        return False

def main():
    print("=" * 50)
    print("9ml Automated Test Suite")
    print("=" * 50)
    print()

    results = {
        'rmsnorm': test_rmsnorm(),
        'softmax': test_softmax(),
        'matmul': test_matmul(),
        'rng': test_rng(),
        'model_loading': test_model_loading(),
        'generation': test_generation(),
        'quantize': test_quantize(),
        'quantized_matmul': test_quantized_matmul(),
    }

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for name, result in results.items():
        status = "PASS" if result is True else ("FAIL" if result is False else "SKIP")
        print(f"  {name}: {status}")

    print()
    print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
