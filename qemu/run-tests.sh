#!/bin/bash
# run-tests.sh - Run all tests in Plan 9 VM (no expect required)
#
# Uses timed input to handle boot prompts

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DISK_IMAGE="$SCRIPT_DIR/9front.qcow2"
SHARED_IMG="$SCRIPT_DIR/shared.img"

cd "$PROJECT_DIR"

# Create a FIFO for sending commands
FIFO=$(mktemp -u)
mkfifo "$FIFO"

# Start QEMU in background with FIFO as input
qemu-system-x86_64 \
    -m 512 \
    -cpu max \
    -accel tcg \
    -drive file="$DISK_IMAGE",format=qcow2,if=virtio \
    -drive file="$SHARED_IMG",format=raw,if=virtio \
    -display none \
    -serial mon:stdio \
    < "$FIFO" &

QEMU_PID=$!

# Open FIFO for writing (keeps it open)
exec 3>"$FIFO"

# Function to send a line
send() {
    echo "$1" >&3
}

# Wait for boot and send responses
sleep 15   # Wait for bootargs prompt
send ""    # Accept default bootargs

sleep 5    # Wait for user prompt
send ""    # Accept default user (glenda)

sleep 20   # Wait for shell to be ready

# Mount shared disk and run tests
send "dossrv -f /dev/sdG0/data shared"
sleep 2
send "mount -c /srv/shared /mnt/host"
sleep 2
send "cd /mnt/host"
sleep 1

# Run each test
send "6c -w test_rmsnorm.c && 6l -o t_rmsnorm test_rmsnorm.6 && ./t_rmsnorm > rmsnorm.out"
sleep 3

send "6c -w test_softmax.c && 6l -o t_softmax test_softmax.6 && ./t_softmax > softmax.out"
sleep 3

send "6c -w test_matmul.c && 6l -o t_matmul test_matmul.6 && ./t_matmul > matmul.out"
sleep 3

send "6c -w test_rng.c && 6l -o t_rng test_rng.6 && ./t_rng > rng.out"
sleep 3

send "6c -w test_quantize.c && 6l -o t_quantize test_quantize.6 && ./t_quantize > quantize.out"
sleep 3

send "6c -w test_quantized_matmul.c && 6l -o t_qmatmul test_quantized_matmul.6 && ./t_qmatmul > quantized_matmul.out"
sleep 3

send "6c -w test_model_loading.c && 6l -o t_model test_model_loading.6 && ./t_model > model_loading.out"
sleep 3

# Compile and run generation test (takes longer)
send "6c -w run.c && 6l -o run run.6"
sleep 5
send "./run stories15M.bin -z tokenizer.bin -n 20 -s 42 -t 0.0 -i 'Once upon a time' > generation.out"
sleep 30

# Mark completion
send "echo done > complete.txt"
sleep 2

# Shutdown
send "fshalt"
sleep 3

# Close FIFO and cleanup
exec 3>&-
rm -f "$FIFO"

# Kill QEMU (it may not shutdown cleanly)
kill $QEMU_PID 2>/dev/null || true
wait $QEMU_PID 2>/dev/null || true

exit 0
