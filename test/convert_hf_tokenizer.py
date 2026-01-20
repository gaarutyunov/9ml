#!/usr/bin/env python3
"""
Convert HuggingFace tokenizer to llama2.c tokenizer.bin format.

Format:
- int max_token_length
- For each token:
  - float score
  - int string_length
  - bytes string (no null terminator)
"""

import struct
import sys
import json
from pathlib import Path

def convert_hf_tokenizer(hf_tokenizer_dir, output_path, vocab_size=None):
    """Convert HuggingFace tokenizer to llama2.c format."""

    tokenizer_json = Path(hf_tokenizer_dir) / "tokenizer.json"

    if not tokenizer_json.exists():
        print(f"Error: {tokenizer_json} not found")
        return False

    with open(tokenizer_json, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    # Get vocab from the model
    vocab = tokenizer_data.get("model", {}).get("vocab", {})

    if not vocab:
        print("Error: Could not find vocab in tokenizer.json")
        return False

    # Determine vocab size
    actual_vocab_size = len(vocab)
    if vocab_size is None:
        vocab_size = actual_vocab_size

    print(f"Vocab size: {vocab_size} (actual: {actual_vocab_size})")

    # Sort vocab by ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    # Verify we have all IDs from 0 to vocab_size-1
    for i, (token, idx) in enumerate(sorted_vocab[:vocab_size]):
        if idx != i:
            print(f"Warning: Missing token ID {i}, found {idx}")

    # Find max token length
    max_token_length = max(len(token.encode('utf-8')) for token, _ in sorted_vocab[:vocab_size])
    print(f"Max token length: {max_token_length}")

    # Try to get scores from merges or use default
    merges = tokenizer_data.get("model", {}).get("merges", [])

    # Create score map (higher merge order = higher score)
    # For BPE, tokens with lower merge order are more common
    scores = {}
    for i, merge in enumerate(merges):
        # Score is inverse of merge order (more common = higher score)
        # Use a scaling factor to avoid tiny differences
        score = 1.0 / (i + 1)

        # Handle both string and list merge formats
        if isinstance(merge, str):
            parts = merge.split()
        elif isinstance(merge, list):
            parts = merge
        else:
            continue

        if len(parts) == 2:
            merged_token = parts[0] + parts[1]
            if merged_token in vocab:
                scores[vocab[merged_token]] = score

    # Write output
    with open(output_path, "wb") as f:
        # Write max_token_length
        f.write(struct.pack("i", max_token_length))

        # Write each token
        for token, idx in sorted_vocab[:vocab_size]:
            # Score (use merge-based score or default)
            score = scores.get(idx, 0.0)

            # Token bytes
            token_bytes = token.encode('utf-8')
            token_len = len(token_bytes)

            # Write: score (float), length (int), token bytes
            f.write(struct.pack("f", score))
            f.write(struct.pack("i", token_len))
            f.write(token_bytes)

    output_size = Path(output_path).stat().st_size
    print(f"Written {output_path} ({output_size} bytes)")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <hf_tokenizer_dir> <output.bin> [vocab_size]")
        sys.exit(1)

    hf_dir = sys.argv[1]
    output = sys.argv[2]
    vocab_size = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if not convert_hf_tokenizer(hf_dir, output, vocab_size):
        sys.exit(1)
