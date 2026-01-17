#!/bin/bash
# run-qemu.sh - Boot 9front in QEMU with virtio-9p file sharing
#
# Usage: ./run-qemu.sh [--headless]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DISK_IMAGE="$SCRIPT_DIR/9front.qcow2"
SHARED_IMG="$SCRIPT_DIR/shared.img"
SHARED_DIR="$SCRIPT_DIR/shared"
SERIAL_SOCKET="$SCRIPT_DIR/serial.sock"

# QEMU binary - MacPorts installation
QEMU="/opt/local/bin/qemu-system-x86_64"

# Ensure shared directory exists
mkdir -p "$SHARED_DIR"

# Check for disk image
if [ ! -f "$DISK_IMAGE" ]; then
    echo "Error: $DISK_IMAGE not found"
    exit 1
fi

# Check for QEMU
if [ ! -x "$QEMU" ]; then
    echo "Error: QEMU not found at $QEMU"
    echo "Install via: sudo port install qemu"
    exit 1
fi

# Remove stale socket
rm -f "$SERIAL_SOCKET"

HEADLESS=""
DISPLAY_OPTS="-display cocoa"
EXTRA_ARGS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --headless)
            HEADLESS="1"
            DISPLAY_OPTS="-display none -nographic"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Detect accelerator: HVF for macOS, KVM for Linux
ACCEL="-accel hvf"
if [ "$(uname)" = "Linux" ]; then
    ACCEL="-accel kvm"
fi

# Run QEMU with:
# - 512MB RAM
# - HVF acceleration (macOS) or KVM (Linux)
# - virtio-9p file sharing (mount tag: host)
# - serial console on unix socket for automation
# - VGA for graphical console (9front expects it)
exec "$QEMU" \
    -m 512 \
    -cpu max \
    $ACCEL \
    -drive file="$DISK_IMAGE",format=qcow2,if=virtio \
    -drive file="$SHARED_IMG",format=raw,if=virtio \
    -device virtio-net-pci,netdev=net0 \
    -netdev user,id=net0 \
    -virtfs local,path="$SHARED_DIR",mount_tag=host,security_model=none,id=host0 \
    -serial unix:"$SERIAL_SOCKET",server,nowait \
    $DISPLAY_OPTS \
    $EXTRA_ARGS
