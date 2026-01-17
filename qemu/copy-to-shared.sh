#!/bin/bash
# copy-to-shared.sh - Copy files to the FAT shared disk
#
# Usage: ./copy-to-shared.sh file1 [file2 ...]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_IMG="$SCRIPT_DIR/shared.img"

# Source OS helper functions
source "$SCRIPT_DIR/os-helper.sh"

if [ $# -eq 0 ]; then
    echo "Usage: $0 file1 [file2 ...]"
    exit 1
fi

if [ ! -f "$SHARED_IMG" ]; then
    echo "Error: $SHARED_IMG not found"
    echo "Run ./create-shared.sh first to create the shared disk"
    exit 1
fi

# Mount the FAT image
MOUNT_POINT=$(mktemp -d /tmp/shared_mount.XXXXXX)
mount_fat_image "$SHARED_IMG" "$MOUNT_POINT"

if [ $? -ne 0 ]; then
    echo "Error: Failed to mount $SHARED_IMG"
    rmdir "$MOUNT_POINT"
    exit 1
fi

# Copy files
for file in "$@"; do
    if [ -f "$file" ]; then
        cp "$file" "$MOUNT_POINT/"
        echo "Copied: $file"
    elif [ -d "$file" ]; then
        cp -r "$file" "$MOUNT_POINT/"
        echo "Copied directory: $file"
    else
        echo "Warning: $file not found"
    fi
done

# Unmount
unmount_fat_image "$MOUNT_POINT"
rmdir "$MOUNT_POINT" 2>/dev/null

echo "Done. Files are now in the shared disk."
