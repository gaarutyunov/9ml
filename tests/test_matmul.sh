#!/bin/bash
# test_matmul.sh - Automated test for matmul function
# Compares Plan 9 output against Python reference implementation

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
QEMU_DIR="$PROJECT_DIR/qemu"
SRC_DIR="$PROJECT_DIR/src"

echo "=== Test: matmul Output Matches Python Reference ==="

# Create Plan 9 test program
cat > "$SRC_DIR/test_matmul.c" << 'EOF'
#include <u.h>
#include <libc.h>

void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void main(int, char**) {
    // W: 3x4 matrix, x: 4-element vector
    float w[] = {1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12};
    float x[] = {1, 2, 3, 4};
    float out[3];

    matmul(out, x, w, 4, 3);

    for (int i = 0; i < 3; i++) {
        print("%.6f\n", out[i]);
    }
    exits(0);
}
EOF

# Generate Python reference output
echo "Generating Python reference..."
python3 - << 'PYTHON' > /tmp/python_matmul.txt
import numpy as np

# W (d, n) @ x (n,) -> xout (d,)
w = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]], dtype=np.float32)
x = np.array([1, 2, 3, 4], dtype=np.float32)

result = w @ x
for v in result:
    print(f"{v:.6f}")
PYTHON

# Run Plan 9
echo "Running in Plan 9..."
"$QEMU_DIR/copy-to-shared.sh" "$SRC_DIR/test_matmul.c" > /dev/null
"$QEMU_DIR/run-cmd.sh" "6c /mnt/host/test_matmul.c && 6l -o test_matmul test_matmul.6 && ./test_matmul" > /tmp/plan9_matmul.txt

# Compare with epsilon
echo "Comparing outputs..."
python3 - << 'PYTHON'
import sys

with open('/tmp/python_matmul.txt') as f:
    ref = [float(x.strip()) for x in f.readlines() if x.strip()]
with open('/tmp/plan9_matmul.txt') as f:
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

print("PASS: matmul matches Python reference within epsilon")
PYTHON
