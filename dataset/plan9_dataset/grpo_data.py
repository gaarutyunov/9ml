"""
GRPO dataset generation and SFT format conversion.

Loads task data from JSONL files and converts existing
instruction-response pairs to function calling format.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .export import Example, load_examples
from .tools import (
    PLAN9_TOOLS,
    format_system_prompt,
    format_user_prompt,
    format_model_turn,
    format_tool_call,
    format_tool_response,
    format_thinking,
    START_OF_TURN,
    END_OF_TURN,
)


def load_tasks(tasks_path: Path | None = None) -> list[dict]:
    """Load tasks from JSONL file.

    Args:
        tasks_path: Path to tasks.jsonl file. If None, uses default path.

    Returns:
        List of task dictionaries.
    """
    if tasks_path is None:
        tasks_path = Path(__file__).parent.parent / "data" / "grpo" / "tasks.jsonl"

    if not tasks_path.exists():
        return []

    tasks = []
    with open(tasks_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def get_tasks_by_complexity(complexity: str | None = None, tasks_path: Path | None = None) -> list[dict]:
    """Get tasks filtered by complexity level.

    Args:
        complexity: 'simple', 'medium', 'complex', or None for all.
        tasks_path: Path to tasks.jsonl file.

    Returns:
        List of matching tasks.
    """
    tasks = load_tasks(tasks_path)
    if complexity is None:
        return tasks
    return [t for t in tasks if t.get("complexity") == complexity]


def get_tasks_by_category(category: str, tasks_path: Path | None = None) -> list[dict]:
    """Get tasks filtered by category prefix.

    Args:
        category: Category prefix to match.
        tasks_path: Path to tasks.jsonl file.

    Returns:
        List of matching tasks.
    """
    tasks = load_tasks(tasks_path)
    return [t for t in tasks if t.get("category", "").startswith(category)]


@dataclass
class FunctionCallExample:
    """A multi-turn function calling example."""
    system: str
    turns: list[dict[str, str]] = field(default_factory=list)
    expected_output: str | None = None

    def to_conversation(self) -> str:
        """Convert to a conversation string for training."""
        parts = [self.system]
        for turn in self.turns:
            role = turn["role"]
            content = turn["content"]
            parts.append(f"{START_OF_TURN}{role}\n{content}\n{END_OF_TURN}")
        return "\n".join(parts)

    def to_sharegpt(self) -> dict:
        """Convert to ShareGPT format."""
        conversations = []
        for turn in self.turns:
            role = turn["role"]
            if role == "user":
                from_val = "human"
            elif role == "model":
                from_val = "gpt"
            else:
                from_val = "system"
            conversations.append({"from": from_val, "value": turn["content"]})
        return {"conversations": conversations}


def convert_example_to_function_calls(example: Example) -> FunctionCallExample | None:
    """Convert an instruction-response pair to function calling format.

    Analyzes the response to determine which tool calls would be needed
    and constructs a multi-turn conversation.

    Args:
        example: Original instruction-response pair.

    Returns:
        FunctionCallExample or None if conversion not applicable.
    """
    instruction = example.instruction
    response = example.response
    category = example.category

    # Determine the type of response
    is_c_code = "#include <u.h>" in response or "#include <libc.h>" in response
    is_rc_script = response.strip().startswith("#!/bin/rc") or response.strip().startswith("fn ")
    is_mkfile = "</$objtype/mkfile" in response

    if not (is_c_code or is_rc_script or is_mkfile):
        # Not code that can be written/executed
        return None

    # Build the function call example
    system = format_system_prompt()
    turns = []

    # User turn
    turns.append({"role": "user", "content": instruction})

    # Model turn with thinking and tool calls
    thinking = _generate_thinking(instruction, response, category)

    if is_c_code:
        # Write C file, compile, run
        filename = "program.c"
        binary = "program"
        code = _extract_code(response)

        model_content = ""
        if thinking:
            model_content += format_thinking(thinking) + "\n"
        model_content += format_tool_call("write_file", {"path": filename, "content": code})

        turns.append({"role": "model", "content": model_content})

        # Tool response
        turns.append({
            "role": "user",
            "content": format_tool_response("write_file", {"success": True})
        })

        # Compile
        turns.append({
            "role": "model",
            "content": format_tool_call("run_command", {
                "command": f"6c {filename} && 6l -o {binary} program.6"
            })
        })

        turns.append({
            "role": "user",
            "content": format_tool_response("run_command", {"success": True, "output": ""})
        })

        # Run
        turns.append({
            "role": "model",
            "content": format_tool_call("run_command", {"command": f"./{binary}"})
        })

        turns.append({
            "role": "user",
            "content": format_tool_response("run_command", {
                "success": True,
                "output": "[program output]\n"
            })
        })

        # Final model summary
        turns.append({
            "role": "model",
            "content": "The program compiled and ran successfully."
        })

    elif is_rc_script:
        # Write rc script, make executable, run
        filename = "script.rc"
        code = _extract_code(response)

        model_content = ""
        if thinking:
            model_content += format_thinking(thinking) + "\n"
        model_content += format_tool_call("write_file", {"path": filename, "content": code})

        turns.append({"role": "model", "content": model_content})

        # Tool response
        turns.append({
            "role": "user",
            "content": format_tool_response("write_file", {"success": True})
        })

        # Run
        turns.append({
            "role": "model",
            "content": format_tool_call("run_command", {"command": f"chmod +x {filename} && ./{filename}"})
        })

        turns.append({
            "role": "user",
            "content": format_tool_response("run_command", {
                "success": True,
                "output": "[script output]\n"
            })
        })

        # Final model summary
        turns.append({
            "role": "model",
            "content": "The script ran successfully."
        })

    elif is_mkfile:
        # Write mkfile
        filename = "mkfile"
        code = _extract_code(response)

        model_content = ""
        if thinking:
            model_content += format_thinking(thinking) + "\n"
        model_content += format_tool_call("write_file", {"path": filename, "content": code})

        turns.append({"role": "model", "content": model_content})

        # Tool response
        turns.append({
            "role": "user",
            "content": format_tool_response("write_file", {"success": True})
        })

        # Final model summary
        turns.append({
            "role": "model",
            "content": "The mkfile has been created. Run 'mk' to build the project."
        })

    return FunctionCallExample(system=system, turns=turns)


def _generate_thinking(instruction: str, response: str, category: str) -> str:
    """Generate reasoning text for an example."""
    thinking_parts = []

    if "c-" in category or "#include" in response:
        thinking_parts.append("I need to write a Plan 9 C program.")
        if "#include <u.h>" in response:
            thinking_parts.append("Plan 9 uses <u.h> and <libc.h> headers.")
        if "print(" in response:
            thinking_parts.append("I'll use print() instead of printf().")
        if "exits(" in response:
            thinking_parts.append("I'll use exits(nil) to exit successfully.")

    elif "rc-" in category or "#!/bin/rc" in response:
        thinking_parts.append("I need to write an rc shell script.")
        if "rfork" in response:
            thinking_parts.append("I'll use rfork to isolate the namespace/environment.")

    elif "mkfile" in category:
        thinking_parts.append("I need to write a Plan 9 mkfile for building.")
        thinking_parts.append("I'll include the architecture-specific mkfile and mkone.")

    if thinking_parts:
        return " ".join(thinking_parts)
    return ""


def _extract_code(response: str) -> str:
    """Extract just the code from a response (remove explanations)."""
    lines = response.strip().split("\n")

    # Find code start
    code_start = 0
    for i, line in enumerate(lines):
        if (line.startswith("#!") or line.startswith("#include") or
            line.startswith("fn ") or line.startswith("</$objtype")):
            code_start = i
            break

    # Find code end (stop at explanatory text)
    code_end = len(lines)
    for i in range(code_start, len(lines)):
        line = lines[i].strip()
        # Stop at common explanation markers
        if line.startswith("Usage:") or line.startswith("Note:"):
            code_end = i
            break
        # Stop at "This function/script/program" explanations
        if line.startswith("This ") and any(w in line for w in ["function", "script", "program", "mkfile"]):
            code_end = i
            break

    code = "\n".join(lines[code_start:code_end]).rstrip()

    # Ensure trailing newline for code files
    if not code.endswith("\n"):
        code += "\n"

    return code


def create_grpo_dataset(
    tasks_path: Path | None = None,
    include_system: bool = True,
) -> list[dict[str, Any]]:
    """Create a dataset for GRPO training.

    Args:
        tasks_path: Path to tasks.jsonl file.
        include_system: Whether to include system prompt in prompts.

    Returns:
        List of dicts with 'prompt' and 'expected_output' keys.
    """
    tasks = load_tasks(tasks_path)

    data = []
    for task in tasks:
        prompt = ""
        if include_system:
            prompt = format_system_prompt()
        prompt += format_user_prompt(task.get("prompt", ""))

        data.append({
            "prompt": prompt,
            "expected_output": task.get("expected_output"),
            "complexity": task.get("complexity", "simple"),
            "category": task.get("category", ""),
        })

    return data


def create_sft_dataset(
    examples_path: Path,
    include_function_calls: bool = True,
) -> list[dict[str, Any]]:
    """Create an SFT dataset from existing examples.

    Args:
        examples_path: Path to examples.json.
        include_function_calls: Convert to function calling format.

    Returns:
        List of training examples.
    """
    examples = load_examples(examples_path)
    data = []

    for example in examples:
        if include_function_calls:
            fc_example = convert_example_to_function_calls(example)
            if fc_example:
                data.append({
                    "text": fc_example.to_conversation(),
                    "category": example.category,
                    "source": example.source,
                })
            else:
                # Keep as simple instruction-response
                data.append({
                    "instruction": example.instruction,
                    "response": example.response,
                    "category": example.category,
                    "source": example.source,
                })
        else:
            data.append({
                "instruction": example.instruction,
                "response": example.response,
                "category": example.category,
                "source": example.source,
            })

    return data


def export_tasks_jsonl(output_path: Path, tasks: list[dict]) -> int:
    """Export tasks to JSONL file.

    Args:
        output_path: Path to output JSONL file.
        tasks: Tasks to export.

    Returns:
        Number of tasks exported.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    return len(tasks)


def export_sft_function_calls(
    examples_path: Path,
    output_path: Path,
    format: str = "sharegpt",
) -> int:
    """Export examples in function calling format.

    Args:
        examples_path: Path to examples.json.
        output_path: Path to output file.
        format: Output format ('sharegpt', 'jsonl', 'text').

    Returns:
        Number of examples exported.
    """
    examples = load_examples(examples_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = []
    for example in examples:
        fc_example = convert_example_to_function_calls(example)
        if fc_example:
            converted.append(fc_example)

    if format == "sharegpt":
        data = [ex.to_sharegpt() for ex in converted]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "jsonl":
        with open(output_path, "w") as f:
            for ex in converted:
                line = json.dumps({"text": ex.to_conversation()})
                f.write(line + "\n")

    elif format == "text":
        with open(output_path, "w") as f:
            for ex in converted:
                f.write(ex.to_conversation())
                f.write("\n\n---\n\n")

    return len(converted)


def get_dataset_stats(
    examples_path: Path | None = None,
    tasks_path: Path | None = None,
) -> dict[str, Any]:
    """Get statistics about the dataset.

    Args:
        examples_path: Path to examples.json for SFT stats.
        tasks_path: Path to tasks.jsonl for task stats.

    Returns:
        Dict with statistics.
    """
    stats = {}

    tasks = load_tasks(tasks_path)

    # GRPO task stats
    stats["grpo_tasks"] = {
        "total": len(tasks),
        "by_complexity": {
            "simple": len([t for t in tasks if t.get("complexity") == "simple"]),
            "medium": len([t for t in tasks if t.get("complexity") == "medium"]),
            "complex": len([t for t in tasks if t.get("complexity") == "complex"]),
        },
        "by_category": {},
    }

    for task in tasks:
        cat_full = task.get("category", "")
        cat = cat_full.split("-")[0] if "-" in cat_full else cat_full
        stats["grpo_tasks"]["by_category"][cat] = (
            stats["grpo_tasks"]["by_category"].get(cat, 0) + 1
        )

    # SFT example stats
    if examples_path and examples_path.exists():
        examples = load_examples(examples_path)
        convertible = 0
        for ex in examples:
            if convert_example_to_function_calls(ex) is not None:
                convertible += 1

        stats["sft_examples"] = {
            "total": len(examples),
            "convertible_to_function_calls": convertible,
        }

    return stats
