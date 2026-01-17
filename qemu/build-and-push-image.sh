#!/bin/bash
# build-and-push-image.sh - Download 9front QCOW2 and push to ghcr.io
#
# Usage: ./build-and-push-image.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QCOW2_FILE="$SCRIPT_DIR/9front.qcow2"
QCOW2_URL="https://9front.org/iso/9front-11321.amd64.qcow2.gz"
REGISTRY="ghcr.io/gaarutyunov/9ml"
TAG="9front-11321"

echo "=== Downloading 9front QCOW2 image ==="

# Step 1: Download QCOW2 if needed
if [ ! -f "$QCOW2_FILE" ]; then
    echo "Downloading from 9front.org..."
    curl -L --progress-bar -o "${QCOW2_FILE}.gz" "$QCOW2_URL"
    echo "Extracting..."
    gunzip "${QCOW2_FILE}.gz"
fi

echo "QCOW2 ready: $(ls -lh "$QCOW2_FILE")"

# Step 2: Compress for upload
echo ""
echo "Compressing QCOW2..."
gzip -k -f "$QCOW2_FILE"
ls -lh "${QCOW2_FILE}.gz"

# Step 3: Push to ghcr.io
echo ""
echo "Pushing to ghcr.io..."

if ! command -v oras &> /dev/null; then
    echo "Error: oras CLI not found. Install with: brew install oras"
    exit 1
fi

# Login to ghcr.io
echo "Logging in to ghcr.io..."
gh auth token | oras login ghcr.io -u "$(gh api user -q .login)" --password-stdin

# Push
oras push "$REGISTRY:$TAG" \
    --artifact-type application/vnd.9ml.qcow2.v1+gzip \
    "${QCOW2_FILE}.gz:application/gzip"

echo ""
echo "=== Done ==="
echo "Image pushed to: $REGISTRY:$TAG"
