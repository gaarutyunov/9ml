#!/bin/bash
# install-9front.sh - Download pre-built 9front QCOW2 image
#
# Usage: ./install-9front.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QCOW2_FILE="$SCRIPT_DIR/9front.qcow2"
QCOW2_URL="https://9front.org/iso/9front-11321.amd64.qcow2.gz"

if [ -f "$QCOW2_FILE" ]; then
    echo "9front.qcow2 already exists at $QCOW2_FILE"
    exit 0
fi

echo "Downloading 9front QCOW2 from 9front.org..."
curl -L --progress-bar -o "${QCOW2_FILE}.gz" "$QCOW2_URL"

echo "Extracting..."
gunzip "${QCOW2_FILE}.gz"

echo ""
echo "9front installed successfully"
ls -lh "$QCOW2_FILE"
