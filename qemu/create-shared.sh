#!/bin/bash
# create-shared.sh - Create the FAT32 shared disk image
#
# Usage: ./create-shared.sh [size_mb]
#
# Creates a FAT32 disk image for sharing files between host and Plan 9 guest.
# Default size is 128MB.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_IMG="$SCRIPT_DIR/shared.img"
SIZE_MB="${1:-128}"

# Source OS helper functions
source "$SCRIPT_DIR/os-helper.sh"

if [ -f "$SHARED_IMG" ]; then
    echo "Shared disk already exists: $SHARED_IMG"
    echo "Remove it first if you want to recreate: rm $SHARED_IMG"
    exit 0
fi

echo "Creating ${SIZE_MB}MB FAT32 shared disk..."
create_fat_image "$SHARED_IMG" "$SIZE_MB"

if [ $? -eq 0 ] && [ -f "$SHARED_IMG" ]; then
    echo "Created: $SHARED_IMG"
    ls -lh "$SHARED_IMG"
else
    echo "Error: Failed to create shared disk"
    exit 1
fi
