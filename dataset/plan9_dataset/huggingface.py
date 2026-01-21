"""
HuggingFace Hub publishing utilities for Plan 9 dataset.

Provides functions to:
- Convert examples.json to HuggingFace DatasetDict
- Publish dataset to HuggingFace Hub
- Generate dataset cards with statistics
"""

import json
from pathlib import Path
from typing import Any

from .export import Example, load_examples, get_stats
from .grpo_data import (
    GRPO_TASKS,
    convert_example_to_function_calls,
)
from .tools import format_system_prompt


def examples_to_hf_dataset(examples_path: Path) -> "DatasetDict":
    """Convert examples.json to HuggingFace DatasetDict with two configs.

    Configs:
    - 'sft': Simple instruction-response pairs for standard SFT training
    - 'function_calling': Multi-turn conversations with tool calls for agentic training

    Args:
        examples_path: Path to examples.json file.

    Returns:
        DatasetDict with 'sft' and 'function_calling' configurations.
    """
    from datasets import Dataset, DatasetDict

    examples = load_examples(examples_path)

    # Build SFT dataset (simple instruction-response pairs)
    sft_data = {
        "instruction": [],
        "response": [],
        "category": [],
        "source": [],
    }

    for ex in examples:
        sft_data["instruction"].append(ex.instruction)
        sft_data["response"].append(ex.response)
        sft_data["category"].append(ex.category)
        sft_data["source"].append(ex.source)

    sft_dataset = Dataset.from_dict(sft_data)

    # Build function calling dataset (multi-turn conversations)
    fc_data = {
        "text": [],
        "category": [],
        "source": [],
    }

    for ex in examples:
        fc_example = convert_example_to_function_calls(ex)
        if fc_example:
            fc_data["text"].append(fc_example.to_conversation())
            fc_data["category"].append(ex.category)
            fc_data["source"].append(ex.source)

    fc_dataset = Dataset.from_dict(fc_data)

    return DatasetDict({
        "sft": sft_dataset,
        "function_calling": fc_dataset,
    })


def generate_dataset_card(
    examples_path: Path,
    repo_id: str,
    tasks_count: int | None = None,
) -> str:
    """Generate README.md with YAML frontmatter for HuggingFace UI.

    Args:
        examples_path: Path to examples.json for statistics.
        repo_id: HuggingFace repository ID (user/repo-name).
        tasks_count: Optional GRPO task count override.

    Returns:
        Dataset card content as string.
    """
    examples = load_examples(examples_path) if examples_path.exists() else []
    stats = get_stats(examples) if examples else {}

    # Count function calling convertible examples
    fc_count = 0
    for ex in examples:
        if convert_example_to_function_calls(ex):
            fc_count += 1

    # Get category statistics
    categories = stats.get("by_category", {})
    category_table = "\n".join(
        f"| {cat} | {count} |"
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])
    )

    # GRPO tasks stats
    grpo_count = tasks_count or len(GRPO_TASKS)
    grpo_by_complexity = {
        "simple": len([t for t in GRPO_TASKS if t.complexity == "simple"]),
        "medium": len([t for t in GRPO_TASKS if t.complexity == "medium"]),
        "complex": len([t for t in GRPO_TASKS if t.complexity == "complex"]),
    }

    # Sample example for documentation
    sample_instruction = ""
    sample_response = ""
    sample_response_escaped = ""
    if examples:
        sample = examples[0]
        sample_instruction = sample.instruction
        sample_response = sample.response[:200] + "..." if len(sample.response) > 200 else sample.response
        sample_response_escaped = sample_response.replace("\n", "\\n")

    card = f'''---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - plan9
  - 9front
  - operating-systems
  - systems-programming
  - function-calling
  - code
size_categories:
  - n<1K
---

# Plan 9 Programming Dataset

A curated dataset for fine-tuning LLMs to write Plan 9 (9front) code. Includes C programs, rc shell scripts, mkfiles, and system programming examples.

## Quick Start

```python
from datasets import load_dataset

# Simple instruction-response format (for standard SFT)
ds = load_dataset("{repo_id}", "sft")

# Function calling format (for agentic training)
ds = load_dataset("{repo_id}", "function_calling")
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/plan9-dataset/blob/main/notebooks/plan9_sft_colab.ipynb)

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total SFT Examples | {stats.get("total", 0)} |
| Function Calling Examples | {fc_count} |
| GRPO Tasks | {grpo_count} |
| Avg Instruction Length | {stats.get("avg_instruction_length", 0)} chars |
| Avg Response Length | {stats.get("avg_response_length", 0)} chars |

### GRPO Task Complexity

| Complexity | Count |
|------------|-------|
| Simple | {grpo_by_complexity["simple"]} |
| Medium | {grpo_by_complexity["medium"]} |
| Complex | {grpo_by_complexity["complex"]} |

### Categories

| Category | Count |
|----------|-------|
{category_table}

## Dataset Configurations

### `sft` - Standard SFT Format

Simple instruction-response pairs for supervised fine-tuning.

**Columns:**
- `instruction` (string): The task or question
- `response` (string): The expected answer or code
- `category` (string): Category tag (e.g., "c-basics", "rc-flags")
- `source` (string): Source reference

**Example:**
```json
{{
  "instruction": "{sample_instruction}",
  "response": "{sample_response_escaped}",
  "category": "{examples[0].category if examples else ''}",
  "source": "{examples[0].source if examples else ''}"
}}
```

### `function_calling` - Agentic Format

Multi-turn conversations with tool calls for training function-calling agents.

**Columns:**
- `text` (string): Full conversation with system prompt, tool calls, and responses
- `category` (string): Category tag
- `source` (string): Source reference

**Tools available:**
- `write_file`: Write content to a file
- `read_file`: Read a file
- `run_command`: Execute a command in rc shell

## Plan 9 Conventions

| POSIX | Plan 9 |
|-------|--------|
| `printf()` | `print()` |
| `exit(0)` | `exits(nil)` |
| `NULL` | `nil` |
| `int8_t` | `schar` |
| `uint64_t` | `uvlong` |

**Headers:**
```c
#include <u.h>
#include <libc.h>
```

**Compilation:**
```rc
6c program.c && 6l -o program program.6
```

## Training

This dataset is designed for:

1. **Standard SFT**: Use the `sft` config with any instruction-tuning recipe
2. **Function Calling SFT**: Use `function_calling` config to teach tool use
3. **GRPO/RL Training**: Use the task prompts for reinforcement learning with execution-based rewards

## License

Apache 2.0

## Citation

```bibtex
@misc{{plan9-dataset,
  title={{Plan 9 Programming Dataset}},
  author={{9ml Contributors}},
  year={{2025}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```
'''
    return card


def publish_to_hub(
    examples_path: Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    include_card: bool = True,
) -> str:
    """Publish dataset to HuggingFace Hub.

    Args:
        examples_path: Path to examples.json file.
        repo_id: Repository ID in format "username/dataset-name".
        token: HuggingFace API token. If None, uses cached token.
        private: Whether to create a private repository.
        include_card: Whether to include auto-generated dataset card.

    Returns:
        URL of the published dataset.
    """
    from huggingface_hub import HfApi

    # Convert to HuggingFace dataset
    dataset_dict = examples_to_hf_dataset(examples_path)

    # Push to hub
    dataset_dict.push_to_hub(
        repo_id,
        token=token,
        private=private,
    )

    # Generate and push dataset card
    if include_card:
        card_content = generate_dataset_card(examples_path, repo_id)

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

    return f"https://huggingface.co/datasets/{repo_id}"


def push_grpo_tasks_to_hub(
    repo_id: str,
    token: str | None = None,
    tasks: list | None = None,
) -> str:
    """Push GRPO tasks as a separate dataset file.

    Args:
        repo_id: Repository ID.
        token: HuggingFace API token.
        tasks: Optional custom task list. Defaults to GRPO_TASKS.

    Returns:
        URL of the published file.
    """
    from huggingface_hub import HfApi

    if tasks is None:
        tasks = GRPO_TASKS

    # Convert tasks to JSON
    tasks_data = [
        {
            "prompt": t.prompt,
            "expected_output": t.expected_output,
            "complexity": t.complexity,
            "category": t.category,
            "tool_count_hint": t.tool_count_hint,
        }
        for t in tasks
    ]

    content = json.dumps(tasks_data, indent=2)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="grpo_tasks.json",
        repo_id=repo_id,
        repo_type="dataset",
    )

    return f"https://huggingface.co/datasets/{repo_id}/blob/main/grpo_tasks.json"


def load_from_hub(repo_id: str, config: str = "sft") -> "Dataset":
    """Load dataset from HuggingFace Hub.

    Args:
        repo_id: Repository ID.
        config: Configuration to load ('sft' or 'function_calling').

    Returns:
        HuggingFace Dataset object.
    """
    from datasets import load_dataset
    return load_dataset(repo_id, config)
