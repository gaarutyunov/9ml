#!/bin/bash
# test_generation.sh - Automated test for text generation (Task 52/55)
# Compares generation output between host and Plan 9

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
QEMU_DIR="$PROJECT_DIR/qemu"
SRC_DIR="$PROJECT_DIR/src"
LLAMA2C_DIR="/Users/germanarutyunov/Projects/plan9/llama2.c"
MODEL_PATH="$LLAMA2C_DIR/stories15M.bin"
TOKENIZER_PATH="$LLAMA2C_DIR/tokenizer.bin"

echo "=== Test: Generation Output Matches Reference ==="

# Compile and run host version
echo "Running on host..."
cd "$LLAMA2C_DIR"
gcc -O2 -o run_test run.c -lm
./run_test stories15M.bin -t 0 -s 42 -n 20 2>/dev/null > /tmp/host_gen.txt
cd "$PROJECT_DIR"

# Copy files and run in Plan 9
echo "Running in Plan 9..."
"$QEMU_DIR/copy-to-shared.sh" "$SRC_DIR/run.c" "$MODEL_PATH" "$TOKENIZER_PATH" > /dev/null
"$QEMU_DIR/run-cmd.sh" "6c /mnt/host/run.c && 6l -o run run.6 && ./run /mnt/host/stories15M.bin -z /mnt/host/tokenizer.bin -t 0 -s 42 -n 20" 2>/dev/null > /tmp/plan9_gen.txt

# Compare
echo "Comparing outputs..."
echo "Host output:"
cat /tmp/host_gen.txt
echo ""
echo "Plan 9 output:"
cat /tmp/plan9_gen.txt
echo ""

if diff -q /tmp/host_gen.txt /tmp/plan9_gen.txt > /dev/null 2>&1; then
    echo "PASS: Generation matches reference"
    exit 0
else
    echo "FAIL: Outputs differ"
    echo "--- Diff:"
    diff /tmp/host_gen.txt /tmp/plan9_gen.txt || true
    exit 1
fi
