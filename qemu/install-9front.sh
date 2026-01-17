#!/bin/bash
# install-9front.sh - Download pre-built 9front QCOW2 image
#
# Usage: ./install-9front.sh
#
# Downloads a pre-built 9front QCOW2 image ready to use

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QCOW2_FILE="$SCRIPT_DIR/9front.qcow2"

# Pre-built QCOW2 from archive.org (faster than installing from ISO)
QCOW2_URL="https://archive.org/download/Plan9Front/9front-10378.amd64.qcow2.gz"

if [ -f "$QCOW2_FILE" ]; then
    echo "9front.qcow2 already exists at $QCOW2_FILE"
    exit 0
fi

echo "Downloading pre-built 9front QCOW2..."
curl -L -o "${QCOW2_FILE}.gz" "$QCOW2_URL"

echo "Extracting..."
gunzip "${QCOW2_FILE}.gz"

echo ""
echo "9front installed successfully to $QCOW2_FILE"
ls -lh "$QCOW2_FILE"
