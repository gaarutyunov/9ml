#!/bin/bash
# run-plan9-tests.sh - Run all Plan 9 tests and save outputs to shared disk
#
# This script:
# 1. Copies all test files to the shared disk
# 2. Boots QEMU and runs the test script
# 3. Waits for completion
# 4. Outputs are saved to *.out files on the shared disk

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DISK_IMAGE="$SCRIPT_DIR/9front.qcow2"
SHARED_IMG="$SCRIPT_DIR/shared.img"
SRC_DIR="$PROJECT_DIR/src"
TESTS_DIR="$SRC_DIR/tests"

source "$SCRIPT_DIR/os-helper.sh"

QEMU="$(get_qemu_path)"
ACCEL="$(get_qemu_accel)"
TIMEOUT="${1:-600}"

echo "=== Plan 9 Test Runner ==="

# Check prerequisites
if ! command -v "$QEMU" &> /dev/null; then
    echo "Error: QEMU not found" >&2
    exit 1
fi

if [ ! -f "$DISK_IMAGE" ]; then
    echo "Error: 9front image not found at $DISK_IMAGE" >&2
    echo "Run ./install-9front.sh first" >&2
    exit 1
fi

if [ ! -f "$SHARED_IMG" ]; then
    echo "Creating shared disk..."
    "$SCRIPT_DIR/create-shared.sh"
fi

# Mount shared disk and copy files
echo "Copying test files to shared disk..."
MOUNT_POINT=$(mktemp -d)
mount_fat_image "$SHARED_IMG" "$MOUNT_POINT"

# Clean old outputs
rm -f "$MOUNT_POINT"/*.out 2>/dev/null || true

# Copy test source files
cp "$TESTS_DIR"/test_*.c "$MOUNT_POINT/"
cp "$SRC_DIR"/model.c "$MOUNT_POINT/"
cp "$SRC_DIR"/modelq.c "$MOUNT_POINT/" 2>/dev/null || true
cp "$TESTS_DIR"/run_all.rc "$MOUNT_POINT/"

# Copy model files if they exist
[ -f "$PROJECT_DIR/stories15M.bin" ] && cp "$PROJECT_DIR/stories15M.bin" "$MOUNT_POINT/"
[ -f "$PROJECT_DIR/tokenizer.bin" ] && cp "$PROJECT_DIR/tokenizer.bin" "$MOUNT_POINT/"
[ -f "$SRC_DIR/run.c" ] && cp "$SRC_DIR/run.c" "$MOUNT_POINT/"

unmount_fat_image "$MOUNT_POINT"
rmdir "$MOUNT_POINT"

echo "Starting QEMU..."

# Create input FIFO
INPUT_FIFO=$(mktemp -u)
mkfifo "$INPUT_FIFO"

# Start QEMU in background, capture output
OUTPUT_FILE=$(mktemp)
"$QEMU" \
    -m 512 \
    -cpu max \
    -accel "$ACCEL" \
    -drive file="$DISK_IMAGE",format=qcow2,if=virtio \
    -drive file="$SHARED_IMG",format=raw,if=virtio \
    -display none \
    -serial mon:stdio < "$INPUT_FIFO" > "$OUTPUT_FILE" 2>&1 &

QEMU_PID=$!

# Open FIFO for writing
exec 3>"$INPUT_FIFO"

SCRIPT_EXIT_CODE=0
cleanup() {
    local exit_code=$SCRIPT_EXIT_CODE
    exec 3>&- 2>/dev/null || true
    rm -f "$INPUT_FIFO" "$OUTPUT_FILE"
    kill $QEMU_PID 2>/dev/null || true
    wait $QEMU_PID 2>/dev/null || true
    exit $exit_code
}
trap cleanup EXIT

send() {
    printf '%s\r' "$1" >&3
    sleep 0.5
}

# Monitor for completion in background
(
    while kill -0 $QEMU_PID 2>/dev/null; do
        if grep -q "ALL TESTS COMPLETE" "$OUTPUT_FILE" 2>/dev/null; then
            sleep 2
            kill $QEMU_PID 2>/dev/null || true
            exit 0
        fi
        sleep 1
    done
) &
MONITOR_PID=$!

# Boot sequence
echo "Waiting for boot..."
sleep 25
send ""  # bootargs
sleep 5
send ""  # user
sleep 15

echo "Mounting shared disk..."
send "dossrv -f /dev/sdG0/dos shared"
sleep 3
send "mount -c /srv/shared /mnt/host"
sleep 3

echo "Running tests..."
send "rc /mnt/host/run_all.rc"

# Wait for completion or timeout
WAITED=0
while kill -0 $QEMU_PID 2>/dev/null && [ $WAITED -lt $TIMEOUT ]; do
    sleep 5
    WAITED=$((WAITED + 5))

    # Check if tests completed
    if grep -q "ALL TESTS COMPLETE" "$OUTPUT_FILE" 2>/dev/null; then
        echo "Tests completed!"
        sleep 2
        break
    fi
done

# Kill monitor
kill $MONITOR_PID 2>/dev/null || true

# Shutdown QEMU
send "fshalt"
sleep 3
kill $QEMU_PID 2>/dev/null || true
wait $QEMU_PID 2>/dev/null || true

echo ""
echo "=== QEMU Output ==="
cat "$OUTPUT_FILE"

echo ""
echo "=== Test outputs saved to shared disk ==="

# Set success exit code (cleanup will use this)
SCRIPT_EXIT_CODE=0
