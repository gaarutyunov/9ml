#!/bin/bash
#
# Test Gemma 3 generation on the host (Linux)
#
# This test verifies that the Gemma 3 model produces the expected output
# when run with our implementation. It requires:
#   - models/gemma-3-270m/model.safetensors
#   - models/gemma-3-270m/tokenizer.bin (converted)
#
# The test runs outside the Plan 9 VM due to model size limitations.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models/gemma-3-270m"

MODEL="$MODELS_DIR/model.safetensors"
TOKENIZER="$MODELS_DIR/tokenizer.bin"
REFERENCE="$SCRIPT_DIR/gemma3_generation_reference.json"

echo "=== Gemma 3 Generation Test ==="
echo ""

# Check prerequisites
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Download with: huggingface-cli download google/gemma-3-270m --local-dir $MODELS_DIR"
    exit 1
fi

if [ ! -f "$TOKENIZER" ]; then
    echo "ERROR: Tokenizer not found: $TOKENIZER"
    echo "Convert with: python3 test/convert_hf_tokenizer.py $MODELS_DIR $TOKENIZER 262144"
    exit 1
fi

if [ ! -f "$REFERENCE" ]; then
    echo "ERROR: Reference file not found: $REFERENCE"
    exit 1
fi

# Run Python verification (compares HuggingFace output)
echo "Running Python reference verification..."
python3 << 'PYEOF'
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load reference
with open("test/gemma3_generation_reference.json") as f:
    reference = json.load(f)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "models/gemma-3-270m",
    torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained("models/gemma-3-270m")

passed = 0
failed = 0

for ref in reference:
    prompt = ref["prompt"]
    expected_ids = ref["output_ids"]
    max_tokens = ref["max_tokens"]

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_ids = outputs[0].tolist()
    generated_ids = output_ids[len(inputs.input_ids[0]):]

    if generated_ids == expected_ids:
        print(f"PASS: '{prompt}' -> {len(generated_ids)} tokens match")
        passed += 1
    else:
        print(f"FAIL: '{prompt}'")
        print(f"  Expected: {expected_ids}")
        print(f"  Got:      {generated_ids}")
        failed += 1

print(f"\nResults: {passed} passed, {failed} failed")
if failed > 0:
    exit(1)
PYEOF

echo ""
echo "=== Test Complete ==="
