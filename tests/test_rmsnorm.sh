#!/bin/bash
# test_rmsnorm.sh - Automated test for rmsnorm function
# Compares Plan 9 output against Python reference implementation

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
QEMU_DIR="$PROJECT_DIR/qemu"
SRC_DIR="$PROJECT_DIR/src"

echo "=== Test: rmsnorm Output Matches Python Reference ==="

# Create Plan 9 test program
cat > "$SRC_DIR/test_rmsnorm.c" << 'EOF'
#include <u.h>
#include <libc.h>

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrt(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void main(int, char**) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {0.5f, 0.5f, 0.5f, 0.5f};
    float o[4];

    rmsnorm(o, x, w, 4);

    for (int i = 0; i < 4; i++) {
        print("%.6f\n", o[i]);
    }
    exits(0);
}
EOF

# Generate Python reference output
echo "Generating Python reference..."
python3 - << 'PYTHON' > /tmp/python_rmsnorm.txt
import math

def rmsnorm(x, weight):
    ss = sum(v * v for v in x) / len(x)
    ss += 1e-5
    ss = 1.0 / math.sqrt(ss)
    return [w * (ss * v) for w, v in zip(weight, x)]

x = [1.0, 2.0, 3.0, 4.0]
w = [0.5, 0.5, 0.5, 0.5]
result = rmsnorm(x, w)
for v in result:
    print(f"{v:.6f}")
PYTHON

# Run Plan 9
echo "Running in Plan 9..."
"$QEMU_DIR/copy-to-shared.sh" "$SRC_DIR/test_rmsnorm.c" > /dev/null
"$QEMU_DIR/run-cmd.sh" "6c /mnt/host/test_rmsnorm.c && 6l -o test_rmsnorm test_rmsnorm.6 && ./test_rmsnorm" 2>/dev/null | grep -E '^-?[0-9]+\.[0-9]+$' > /tmp/plan9_rmsnorm.txt

# Compare with epsilon
echo "Comparing outputs..."
python3 - << 'PYTHON'
import sys

with open('/tmp/python_rmsnorm.txt') as f:
    ref = [float(x.strip()) for x in f.readlines() if x.strip()]
with open('/tmp/plan9_rmsnorm.txt') as f:
    plan9 = [float(x.strip()) for x in f.readlines() if x.strip()]

print(f"Python reference: {ref}")
print(f"Plan 9 output:    {plan9}")

eps = 0.0001
if len(ref) != len(plan9):
    print(f"FAIL: Different lengths: ref={len(ref)}, plan9={len(plan9)}")
    sys.exit(1)

for i, (r, p) in enumerate(zip(ref, plan9)):
    if abs(r - p) > eps:
        print(f"FAIL: Values differ at [{i}]: ref={r}, plan9={p}, diff={abs(r-p)}")
        sys.exit(1)

print("PASS: rmsnorm matches Python reference within epsilon")
PYTHON
