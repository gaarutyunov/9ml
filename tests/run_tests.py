#!/usr/bin/env python3
"""
Automated test suite for 9ml
Runs all tests in Plan 9 VM, then compares outputs against Python reference implementations
"""

import subprocess
import os
import sys
import math
import tempfile
import shutil

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QEMU_DIR = os.path.join(PROJECT_DIR, "qemu")
SRC_DIR = os.path.join(PROJECT_DIR, "src")
TESTS_DIR = os.path.join(SRC_DIR, "tests")

# Model paths
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PROJECT_DIR, "stories15M.bin"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", os.path.join(PROJECT_DIR, "tokenizer.bin"))

EPSILON = 0.0001


def detect_os():
    """Detect OS type"""
    import platform
    system = platform.system()
    if system == "Darwin":
        return "macos"
    elif system == "Linux":
        return "linux"
    return "unknown"


def mount_shared_disk():
    """Mount the shared disk and return mount point (or image path for mtools)"""
    shared_img = os.path.join(QEMU_DIR, "shared.img")
    os_type = detect_os()

    if os_type == "macos":
        mount_point = tempfile.mkdtemp(prefix="shared_mount_")
        subprocess.run([
            "hdiutil", "attach",
            "-imagekey", "diskimage-class=CRawDiskImage",
            "-mountpoint", mount_point,
            shared_img
        ], capture_output=True, check=True)
        return {"type": "mount", "path": mount_point}
    elif os_type == "linux":
        # Use mtools to read FAT32 image directly (no sudo required)
        return {"type": "mtools", "path": shared_img}
    else:
        raise RuntimeError(f"Unsupported OS: {os_type}")


def unmount_shared_disk(mount_info):
    """Unmount the shared disk"""
    if mount_info["type"] == "mount":
        mount_point = mount_info["path"]
        os_type = detect_os()
        if os_type == "macos":
            subprocess.run(["hdiutil", "detach", mount_point], capture_output=True)
        try:
            os.rmdir(mount_point)
        except:
            pass
    # mtools doesn't need unmounting


def read_output_file(mount_info, test_name):
    """Read output file from shared disk"""
    if mount_info["type"] == "mount":
        output_file = os.path.join(mount_info["path"], f"{test_name}.out")
        if not os.path.exists(output_file):
            return None
        with open(output_file, 'r') as f:
            return f.read().strip()
    elif mount_info["type"] == "mtools":
        # Use mcopy to extract file from FAT32 image
        result = subprocess.run(
            ["mcopy", "-i", mount_info["path"], f"::/{test_name}.out", "-"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()


def parse_floats(content):
    """Parse lines of floats"""
    if not content:
        return []
    lines = []
    for line in content.split('\n'):
        line = line.strip()
        try:
            lines.append(float(line))
        except ValueError:
            pass
    return lines


def parse_ints(content):
    """Parse lines of integers"""
    if not content:
        return []
    lines = []
    for line in content.split('\n'):
        line = line.strip()
        try:
            lines.append(int(line))
        except ValueError:
            pass
    return lines


def parse_key_values(content):
    """Parse key=value lines"""
    if not content:
        return {}, []

    config = {}
    weights = []

    for line in content.split('\n'):
        line = line.strip()
        if '=' in line:
            key, val = line.split('=', 1)
            if key.startswith('w'):
                weights.append(float(val))
            else:
                config[key] = int(val)

    return config, weights


def compare_floats(ref, plan9, eps=EPSILON):
    """Compare two lists of floats within epsilon"""
    if len(ref) != len(plan9):
        return False, f"Length mismatch: ref={len(ref)}, plan9={len(plan9)}"

    for i, (r, p) in enumerate(zip(ref, plan9)):
        if abs(r - p) > eps:
            return False, f"Value mismatch at [{i}]: ref={r}, plan9={p}, diff={abs(r-p)}"
    return True, "OK"


def run_plan9_tests():
    """Run all tests in Plan 9 VM"""
    print("Running tests in Plan 9 VM...")
    result = subprocess.run(
        [os.path.join(QEMU_DIR, "run-plan9-tests.sh")],
        capture_output=True,
        text=True,
        timeout=600
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    # Check for completion message rather than exit code (SIGPIPE from FIFO is expected)
    return "Test outputs saved to shared disk" in result.stdout


def test_rmsnorm(mount_point):
    """Test rmsnorm function"""
    print("Checking rmsnorm...", end=" ", flush=True)

    # Python reference
    x = [1.0, 2.0, 3.0, 4.0]
    w = [0.5, 0.5, 0.5, 0.5]
    ss = sum(v * v for v in x) / len(x)
    ss += 1e-5
    ss = 1.0 / math.sqrt(ss)
    ref = [wi * (ss * xi) for wi, xi in zip(w, x)]

    # Read Plan 9 output
    content = read_output_file(mount_point, "test_rmsnorm")
    if content is None:
        print("FAIL: No output file")
        return False

    plan9 = parse_floats(content)

    ok, msg = compare_floats(ref, plan9)
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        print(f"  Reference: {ref}")
        print(f"  Plan 9:    {plan9}")
        return False


def test_softmax(mount_point):
    """Test softmax function"""
    print("Checking softmax...", end=" ", flush=True)

    # Python reference
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    max_val = max(x)
    exp_x = [math.exp(v - max_val) for v in x]
    total = sum(exp_x)
    ref = [v / total for v in exp_x]

    # Read Plan 9 output
    content = read_output_file(mount_point, "test_softmax")
    if content is None:
        print("FAIL: No output file")
        return False

    plan9 = parse_floats(content)

    ok, msg = compare_floats(ref, plan9)
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        return False


def test_matmul(mount_point):
    """Test matmul function"""
    print("Checking matmul...", end=" ", flush=True)

    # Python reference
    w = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    x = [1, 2, 3, 4]
    ref = [sum(r * xi for r, xi in zip(row, x)) for row in w]

    # Read Plan 9 output
    content = read_output_file(mount_point, "test_matmul")
    if content is None:
        print("FAIL: No output file")
        return False

    plan9 = parse_floats(content)

    ok, msg = compare_floats(ref, plan9)
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        return False


def test_rng(mount_point):
    """Test random number generator"""
    print("Checking RNG...", end=" ", flush=True)

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
    content = read_output_file(mount_point, "test_rng")
    if content is None:
        print("FAIL: No output file")
        return False

    plan9 = parse_ints(content)

    if ref == plan9:
        print("PASS")
        return True
    else:
        print("FAIL")
        print(f"  Reference: {ref}")
        print(f"  Plan 9:    {plan9}")
        return False


def test_model_loading(mount_point):
    """Test model config and weights loading"""
    print("Checking model_loading...", end=" ", flush=True)

    if not os.path.exists(MODEL_PATH):
        print("SKIP (model not found)")
        return None

    # Python reference
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
    content = read_output_file(mount_point, "test_model_loading")
    if content is None:
        print("FAIL: No output file")
        return False

    plan9_config, plan9_weights = parse_key_values(content)

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


def test_quantize(mount_point):
    """Test quantize/dequantize roundtrip"""
    print("Checking quantize...", end=" ", flush=True)

    # Python reference
    GS = 32
    x = [float(i) / 10.0 for i in range(64)]

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
    content = read_output_file(mount_point, "test_quantize")
    if content is None:
        print("FAIL: No output file")
        return False

    plan9 = parse_floats(content)

    ok, msg = compare_floats(ref, plan9, eps=0.1)
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        return False


def test_quantized_matmul(mount_point):
    """Test quantized matmul"""
    print("Checking quantized_matmul...", end=" ", flush=True)

    # Python reference
    GS = 32
    n, d = 64, 2

    x = [float(i % 10) / 10.0 for i in range(n)]
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
    content = read_output_file(mount_point, "test_quantized_matmul")
    if content is None:
        print("FAIL: No output file")
        return False

    plan9 = parse_floats(content)

    ok, msg = compare_floats(ref, plan9, eps=0.5)
    if ok:
        print("PASS")
        return True
    else:
        print(f"FAIL: {msg}")
        print(f"  Reference: {ref}")
        print(f"  Plan 9:    {plan9}")
        return False


def test_generation(mount_point):
    """Test end-to-end generation"""
    print("Checking generation...", end=" ", flush=True)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("SKIP (model or tokenizer not found)")
        return None

    # Read Plan 9 output
    content = read_output_file(mount_point, "generation")
    if content is None:
        print("SKIP (no output - generation test may not have run)")
        return None

    # Just check that we got some output
    if len(content) > 10:
        print(f"PASS (got {len(content)} chars)")
        print(f"  Output: {content[:100]}...")
        return True
    else:
        print("FAIL: Output too short")
        return False


def main():
    print("=" * 50)
    print("9ml Automated Test Suite")
    print("=" * 50)
    print()

    # Run all tests in Plan 9
    run_plan9_tests()

    print()
    print("=" * 50)
    print("Comparing outputs")
    print("=" * 50)
    print()

    # Mount shared disk to read outputs
    try:
        mount_point = mount_shared_disk()
    except Exception as e:
        print(f"Error mounting shared disk: {e}")
        return 1

    try:
        results = {
            'rmsnorm': test_rmsnorm(mount_point),
            'softmax': test_softmax(mount_point),
            'matmul': test_matmul(mount_point),
            'rng': test_rng(mount_point),
            'model_loading': test_model_loading(mount_point),
            'quantize': test_quantize(mount_point),
            'quantized_matmul': test_quantized_matmul(mount_point),
            'generation': test_generation(mount_point),
        }
    finally:
        unmount_shared_disk(mount_point)

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
