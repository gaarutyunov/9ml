#!/bin/bash
# run-cmd.sh - Execute a command in Plan 9 via QEMU
#
# Usage: ./run-cmd.sh "command to execute"
#
# Uses simple shell with sleep delays instead of expect

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DISK_IMAGE="$SCRIPT_DIR/9front.qcow2"
SHARED_IMG="$SCRIPT_DIR/shared.img"

source "$SCRIPT_DIR/os-helper.sh"

QEMU="$(get_qemu_path)"
ACCEL="$(get_qemu_accel)"

if [ -z "$1" ]; then
    echo "Usage: $0 \"command to execute\"" >&2
    exit 1
fi

CMD="$1"
TIMEOUT="${2:-300}"

if ! command -v "$QEMU" &> /dev/null; then
    echo "Error: QEMU not found at $QEMU" >&2
    exit 1
fi

if [ ! -f "$DISK_IMAGE" ]; then
    echo "Error: Disk image not found at $DISK_IMAGE" >&2
    exit 1
fi

if [ ! -f "$SHARED_IMG" ]; then
    echo "Error: Shared disk not found at $SHARED_IMG" >&2
    exit 1
fi

# Create a FIFO for input
INPUT_FIFO=$(mktemp -u /tmp/qemu-input.XXXXXX)
mkfifo "$INPUT_FIFO"

# Start QEMU in background with serial I/O through the FIFO
"$QEMU" \
    -m 512 \
    -cpu max \
    -accel "$ACCEL" \
    -drive file="$DISK_IMAGE",format=qcow2,if=virtio \
    -drive file="$SHARED_IMG",format=raw,if=virtio \
    -display none \
    -serial mon:stdio < "$INPUT_FIFO" &

QEMU_PID=$!

# Open the FIFO for writing (keeps it open)
exec 3>"$INPUT_FIFO"

# Function to send a line
send() {
    echo "$1" >&3
}

# Function to cleanup
cleanup() {
    exec 3>&-
    rm -f "$INPUT_FIFO"
    kill $QEMU_PID 2>/dev/null
    wait $QEMU_PID 2>/dev/null
}
trap cleanup EXIT

# Wait for boot and send responses
sleep 20  # Wait for bootargs prompt
send ""   # Accept default bootargs

sleep 3   # Wait for user prompt
send ""   # Accept default user (glenda)

sleep 10  # Wait for shell

# Mount shared disk
send "dossrv -f /dev/sdG0/dos shared >/dev/null >[2=1]"
sleep 2
send "mount -c /srv/shared /mnt/host"
sleep 2

# Run command
send "$CMD"
sleep 5

# Shutdown
send "fshalt"
sleep 3

# Force quit QEMU via monitor
send ""  # Ctrl-A
sleep 1

exit 0
