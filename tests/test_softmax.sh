#!/bin/bash
# test_softmax.sh - Automated test for softmax function
# Compares Plan 9 output against Python reference implementation

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
QEMU_DIR="$PROJECT_DIR/qemu"
SRC_DIR="$PROJECT_DIR/src"

echo "=== Test: softmax Output Matches Python Reference ==="

# Create Plan 9 test program
cat > "$SRC_DIR/test_softmax.c" << 'EOF'
#include <u.h>
#include <libc.h>

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void main(int, char**) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    softmax(x, 5);
    for (int i = 0; i < 5; i++) {
        print("%.6f\n", x[i]);
    }
    exits(0);
}
EOF

# Generate Python reference output
echo "Generating Python reference..."
python3 - << 'PYTHON' > /tmp/python_softmax.txt
import math

def softmax(x):
    max_val = max(x)
    exp_x = [math.exp(v - max_val) for v in x]
    total = sum(exp_x)
    return [v / total for v in exp_x]

x = [1.0, 2.0, 3.0, 4.0, 5.0]
result = softmax(x)
for v in result:
    print(f"{v:.6f}")
PYTHON

# Run Plan 9
echo "Running in Plan 9..."
"$QEMU_DIR/copy-to-shared.sh" "$SRC_DIR/test_softmax.c" > /dev/null
"$QEMU_DIR/run-cmd.sh" "6c /mnt/host/test_softmax.c && 6l -o test_softmax test_softmax.6 && ./test_softmax" > /tmp/plan9_softmax.txt

# Compare with epsilon
echo "Comparing outputs..."
python3 - << 'PYTHON'
import sys

with open('/tmp/python_softmax.txt') as f:
    ref = [float(x.strip()) for x in f.readlines() if x.strip()]
with open('/tmp/plan9_softmax.txt') as f:
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

print("PASS: softmax matches Python reference within epsilon")
PYTHON
