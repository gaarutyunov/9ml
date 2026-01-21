"""Export curated examples to training formats."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    """A single training example."""

    instruction: str
    response: str
    source: str = ""
    category: str = ""

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "response": self.response,
            "source": self.source,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Example":
        return cls(
            instruction=data["instruction"],
            response=data["response"],
            source=data.get("source", ""),
            category=data.get("category", ""),
        )


def load_examples(raw_path: Path) -> list[Example]:
    """Load curated examples from raw examples file.

    Args:
        raw_path: Path to examples.json

    Returns:
        List of Example objects
    """
    if not raw_path.exists():
        return []

    with open(raw_path) as f:
        data = json.load(f)

    return [Example.from_dict(item) for item in data]


def load_examples_jsonl(jsonl_path: Path) -> list[Example]:
    """Load examples from JSONL file.

    Args:
        jsonl_path: Path to dataset.jsonl

    Returns:
        List of Example objects
    """
    if not jsonl_path.exists():
        return []

    examples = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append(Example.from_dict(data))

    return examples


def save_examples(examples: list[Example], raw_path: Path) -> None:
    """Save examples to raw examples file.

    Args:
        examples: List of Example objects
        raw_path: Path to examples.json
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    data = [ex.to_dict() for ex in examples]
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2)


def export_jsonl(examples: list[Example], output_path: Path) -> int:
    """Export examples to JSONL format.

    Format: {"instruction": "...", "response": "..."}

    Args:
        examples: List of examples to export
        output_path: Path to output file

    Returns:
        Number of examples exported
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            line = json.dumps({"instruction": ex.instruction, "response": ex.response})
            f.write(line + "\n")

    return len(examples)


def export_alpaca(examples: list[Example], output_path: Path) -> int:
    """Export examples to Alpaca JSON format.

    Format: [{"instruction": "...", "input": "", "output": "..."}, ...]

    Args:
        examples: List of examples to export
        output_path: Path to output file

    Returns:
        Number of examples exported
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for ex in examples:
        data.append({"instruction": ex.instruction, "input": "", "output": ex.response})

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return len(examples)


def export_sharegpt(examples: list[Example], output_path: Path) -> int:
    """Export examples to ShareGPT format (for Axolotl).

    Format: [{"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}, ...]

    Args:
        examples: List of examples to export
        output_path: Path to output file

    Returns:
        Number of examples exported
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for ex in examples:
        data.append(
            {
                "conversations": [
                    {"from": "human", "value": ex.instruction},
                    {"from": "gpt", "value": ex.response},
                ]
            }
        )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return len(examples)


def export_all(examples: list[Example], output_dir: Path) -> dict[str, int]:
    """Export examples to all supported formats.

    Args:
        examples: List of examples to export
        output_dir: Directory for output files

    Returns:
        Dict mapping format name to number of examples exported
    """
    results = {}

    results["jsonl"] = export_jsonl(examples, output_dir / "train.jsonl")
    results["alpaca"] = export_alpaca(examples, output_dir / "train_alpaca.json")
    results["sharegpt"] = export_sharegpt(examples, output_dir / "train_sharegpt.json")

    return results


def get_stats(examples: list[Example]) -> dict:
    """Get statistics about the examples.

    Returns:
        Dict with category counts, total count, etc.
    """
    by_category: dict[str, int] = {}
    total_instruction_chars = 0
    total_response_chars = 0

    for ex in examples:
        category = ex.category or "uncategorized"
        by_category[category] = by_category.get(category, 0) + 1
        total_instruction_chars += len(ex.instruction)
        total_response_chars += len(ex.response)

    return {
        "total": len(examples),
        "by_category": by_category,
        "avg_instruction_length": total_instruction_chars // max(len(examples), 1),
        "avg_response_length": total_response_chars // max(len(examples), 1),
    }


def validate_examples(examples: list[Example]) -> list[str]:
    """Validate examples for common issues.

    Returns:
        List of warning messages
    """
    warnings = []

    for i, ex in enumerate(examples):
        if not ex.instruction.strip():
            warnings.append(f"Example {i}: Empty instruction")

        if not ex.response.strip():
            warnings.append(f"Example {i}: Empty response")

        if len(ex.instruction) < 10:
            warnings.append(f"Example {i}: Very short instruction ({len(ex.instruction)} chars)")

        if len(ex.response) < 10:
            warnings.append(f"Example {i}: Very short response ({len(ex.response)} chars)")

        # Check for placeholder text
        placeholders = ["TODO", "FIXME", "XXX", "[INSERT", "...", "<your"]
        for placeholder in placeholders:
            if placeholder in ex.instruction or placeholder in ex.response:
                warnings.append(f"Example {i}: Contains placeholder text '{placeholder}'")

    return warnings
