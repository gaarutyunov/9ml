#!/bin/bash
# boot-9front.sh - Boot 9front and wait for rc shell
#
# This script boots 9front in QEMU and waits for the login prompt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DISK_IMAGE="$SCRIPT_DIR/9front.qcow2"
QEMU="/opt/local/bin/qemu-system-x86_64"
OUTPUT_FILE="/tmp/9front-boot.log"

# Check for QEMU
if [ ! -x "$QEMU" ]; then
    echo "Error: QEMU not found at $QEMU"
    exit 1
fi

# Check for disk image
if [ ! -f "$DISK_IMAGE" ]; then
    echo "Error: Disk image not found at $DISK_IMAGE"
    exit 1
fi

# Kill any existing QEMU
pkill -f "qemu-system-x86_64.*9front" 2>/dev/null
sleep 1

echo "Booting 9front..."

# Start QEMU and interact with it
{
    "$QEMU" \
        -m 512 \
        -cpu max \
        -accel hvf \
        -drive file="$DISK_IMAGE",format=qcow2,if=virtio \
        -display none \
        -serial stdio \
        -monitor none \
        2>&1 &
    QEMU_PID=$!

    # Wait for boot prompt and send Enter
    sleep 10

    # Send Enter key (accept default boot args)
    echo ""

    # Wait for user prompt
    sleep 30

    # Send Enter again (accept default user)
    echo ""

    # Give it more time to boot
    sleep 20

    # Now send a test command
    echo "date"
    sleep 2

    # Kill QEMU
    kill $QEMU_PID 2>/dev/null
    wait $QEMU_PID 2>/dev/null
} | tee "$OUTPUT_FILE"

echo ""
echo "Boot log saved to: $OUTPUT_FILE"
