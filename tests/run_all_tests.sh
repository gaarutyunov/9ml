#!/bin/bash
# run_all_tests.sh - Run all automated tests

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PASSED=0
FAILED=0
SKIPPED=0

run_test() {
    local test_name="$1"
    local test_script="$2"

    echo ""
    echo "============================================"
    echo "Running: $test_name"
    echo "============================================"

    if [ ! -x "$test_script" ]; then
        chmod +x "$test_script"
    fi

    if "$test_script"; then
        PASSED=$((PASSED + 1))
        echo "✓ $test_name PASSED"
    else
        FAILED=$((FAILED + 1))
        echo "✗ $test_name FAILED"
    fi
}

echo "Running all tests..."
echo ""

# Run individual tests
run_test "rmsnorm" "$SCRIPT_DIR/test_rmsnorm.sh"
run_test "softmax" "$SCRIPT_DIR/test_softmax.sh"
run_test "matmul" "$SCRIPT_DIR/test_matmul.sh"
run_test "file_loading" "$SCRIPT_DIR/test_file_loading.sh"
run_test "generation" "$SCRIPT_DIR/test_generation.sh"

echo ""
echo "============================================"
echo "Test Summary"
echo "============================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Total:  $((PASSED + FAILED))"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
exit 0
