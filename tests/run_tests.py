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
SHARED_IMG = os.path.join(QEMU_DIR, "shared.img")

# Model paths - use environment variables or default to project directory
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PROJECT_DIR, "stories15M.bin"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", os.path.join(PROJECT_DIR, "tokenizer.bin"))

EPSILON = 0.0001

def kill_qemu():
    """Kill any lingering QEMU processes"""
    subprocess.run(["pkill", "-9", "-f", "qemu-system"], capture_output=True)

def clean_shared_disk():
    """Recreate the shared FAT32 disk image to start fresh"""
    if os.path.exists(SHARED_IMG):
        os.remove(SHARED_IMG)
    subprocess.run([os.path.join(QEMU_DIR, "create-shared.sh")],
                   capture_output=True, check=True)

def copy_to_shared(files):
    """Copy files to the shared disk"""
    subprocess.run([os.path.join(QEMU_DIR, "copy-to-shared.sh")] + files,
                   capture_output=True, check=True, timeout=60)

def read_output_file(filename):
    """Read an output file from the shared disk"""
    result = subprocess.run(
        ["mcopy", "-i", SHARED_IMG, f"::{filename}", "-"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()

def run_all_tests_in_vm():
    """Run all tests in the Plan 9 VM"""
    print("Running tests in Plan 9...", flush=True)
    result = subprocess.run(
        [os.path.join(QEMU_DIR, "run-tests.sh")],
        capture_output=True, text=True, timeout=300
    )
    # Check if completion marker exists
    complete = read_output_file("complete.txt")
    return complete == "done"

def compare_floats(ref, plan9, eps=EPSILON):
    """Compare two lists of floats within epsilon"""
    if len(ref) != len(plan9):
        return False, f"Length mismatch: ref={len(ref)}, plan9={len(plan9)}"

    for i, (r, p) in enumerate(zip(ref, plan9)):
        if abs(r - p) > eps:
            return False, f"Value mismatch at [{i}]: ref={r}, plan9={p}, diff={abs(r-p)}"
    return True, "OK"

def parse_float_output(output):
    """Parse output containing floats, one per line"""
    lines = []
    for line in output.split('\n'):
        line = line.strip()
        try:
            lines.append(float(line))
        except ValueError:
            continue
    return lines

def parse_int_output(output):
    """Parse output containing integers, one per line"""
    lines = []
    for line in output.split('\n'):
        line = line.strip()
        try:
            lines.append(int(line))
        except ValueError:
            continue
    return lines

def parse_keyval_output(output):
    """Parse key=value output"""
    result = {}
    for line in output.split('\n'):
        line = line.strip()
        if '=' in line and not line.startswith('6c') and not line.startswith('6l'):
            key, val = line.split('=', 1)
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result

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

    # Read Plan 9 output
    plan9_out = read_output_file("rmsnorm.out")
    plan9 = parse_float_output(plan9_out)

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

    # Read Plan 9 output
    plan9_out = read_output_file("softmax.out")
    plan9 = parse_float_output(plan9_out)

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

    # Read Plan 9 output
    plan9_out = read_output_file("matmul.out")
    plan9 = parse_float_output(plan9_out)

    ok, msg = compare_floats(ref, plan9)
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

    # Read Plan 9 output
    plan9_out = read_output_file("rng.out")
    plan9 = parse_int_output(plan9_out)

    if ref == plan9:
        print("PASS")
        return True
    else:
        print(f"FAIL")
        print(f"  Reference: {ref}")
        print(f"  Plan 9:    {plan9}")
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
        'vocab_size': abs(config[5]),
        'seq_len': config[6]
    }
    ref_weights = list(weights)

    # Read Plan 9 output
    plan9_out = read_output_file("model_loading.out")
    kv = parse_keyval_output(plan9_out)

    plan9_config = {k: kv.get(k) for k in ref_config.keys()}
    plan9_weights = [kv.get(f'w{i}') for i in range(10) if f'w{i}' in kv]

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

def test_generation():
    """Test end-to-end generation matches between host and Plan 9"""
    print("Testing generation...", end=" ", flush=True)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("SKIP (model or tokenizer not found)")
        return None

    # Download and compile reference llama2.c if needed
    ref_c = "/tmp/run_ref.c"
    ref_bin = "/tmp/run_ref"
    if not os.path.exists(ref_c):
        subprocess.run([
            "curl", "-sL",
            "https://raw.githubusercontent.com/karpathy/llama2.c/master/run.c",
            "-o", ref_c
        ], check=True)
    if not os.path.exists(ref_bin):
        subprocess.run(["gcc", "-O3", "-o", ref_bin, ref_c, "-lm"], check=True)

    # Run host reference version
    host_result = subprocess.run(
        [ref_bin, MODEL_PATH, "-z", TOKENIZER_PATH, "-n", "20", "-s", "42", "-t", "0.0", "-i", "Once upon a time"],
        capture_output=True, text=True, timeout=60
    )
    host_output = host_result.stdout.strip().split('\n')[0]

    # Read Plan 9 output
    plan9_output = read_output_file("generation.out").split('\n')[0] if read_output_file("generation.out") else ""

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

    # Read Plan 9 output
    plan9_out = read_output_file("quantize.out")
    plan9 = parse_float_output(plan9_out)

    ok, msg = compare_floats(ref, plan9, eps=0.1)
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

    # Read Plan 9 output
    plan9_out = read_output_file("quantized_matmul.out")
    plan9 = parse_float_output(plan9_out)

    ok, msg = compare_floats(ref, plan9, eps=0.5)
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

    # Clean up environment
    print("Initializing environment...", flush=True)
    kill_qemu()
    clean_shared_disk()

    # Copy all test files and model to shared disk
    print("Copying test files...", flush=True)
    files_to_copy = [
        os.path.join(TESTS_DIR, "test_rmsnorm.c"),
        os.path.join(TESTS_DIR, "test_softmax.c"),
        os.path.join(TESTS_DIR, "test_matmul.c"),
        os.path.join(TESTS_DIR, "test_rng.c"),
        os.path.join(TESTS_DIR, "test_quantize.c"),
        os.path.join(TESTS_DIR, "test_quantized_matmul.c"),
        os.path.join(TESTS_DIR, "test_model_loading.c"),
        os.path.join(SRC_DIR, "model.c"),
        os.path.join(SRC_DIR, "modelq.c"),
        os.path.join(SRC_DIR, "run.c"),
    ]

    if os.path.exists(MODEL_PATH):
        files_to_copy.append(MODEL_PATH)
    if os.path.exists(TOKENIZER_PATH):
        files_to_copy.append(TOKENIZER_PATH)

    copy_to_shared(files_to_copy)

    # Run all tests in VM
    vm_success = run_all_tests_in_vm()
    if not vm_success:
        print("WARNING: VM tests may not have completed fully")

    print()

    # Check results
    results = {
        'rmsnorm': test_rmsnorm(),
        'softmax': test_softmax(),
        'matmul': test_matmul(),
        'rng': test_rng(),
        'quantize': test_quantize(),
        'quantized_matmul': test_quantized_matmul(),
        'model_loading': test_model_loading(),
        'generation': test_generation(),
    }

    # Clean up
    kill_qemu()
    clean_shared_disk()

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
