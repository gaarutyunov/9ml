#!/bin/bash
# copy-from-shared.sh - Copy files from the FAT shared disk to host
#
# Usage: ./copy-from-shared.sh [destination_dir]
#        Default destination: current directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_IMG="$SCRIPT_DIR/shared.img"
DEST_DIR="${1:-.}"

if [ ! -f "$SHARED_IMG" ]; then
    echo "Error: $SHARED_IMG not found"
    exit 1
fi

# Mount the FAT image
MOUNT_POINT=$(mktemp -d /tmp/shared_mount.XXXXXX)
hdiutil attach -imagekey diskimage-class=CRawDiskImage -mountpoint "$MOUNT_POINT" "$SHARED_IMG" > /dev/null

if [ $? -ne 0 ]; then
    echo "Error: Failed to mount $SHARED_IMG"
    rmdir "$MOUNT_POINT"
    exit 1
fi

# List and copy files
echo "Files in shared disk:"
ls -la "$MOUNT_POINT"

mkdir -p "$DEST_DIR"
cp -r "$MOUNT_POINT"/* "$DEST_DIR/" 2>/dev/null

# Unmount
hdiutil detach "$MOUNT_POINT" > /dev/null
rmdir "$MOUNT_POINT" 2>/dev/null

echo "Done. Files copied to $DEST_DIR"
