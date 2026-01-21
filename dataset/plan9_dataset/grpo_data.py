"""
GRPO dataset generation and SFT format conversion.

Generates task prompts for GRPO training and converts existing
instruction-response pairs to function calling format.
"""

import json
import random
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


@dataclass
class GRPOTask:
    """A task prompt for GRPO training."""
    prompt: str
    expected_output: str | None = None
    complexity: str = "simple"  # simple, medium, complex
    category: str = ""
    tool_count_hint: int = 1


# GRPO task prompts organized by complexity
GRPO_TASKS: list[GRPOTask] = [
    # Simple tasks (1-2 tool calls)
    GRPOTask(
        prompt="Write a Plan 9 C program that prints 'Hello, Plan 9!'",
        expected_output="Hello, Plan 9!",
        complexity="simple",
        category="c-basics",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write an rc script that prints the current date",
        expected_output=None,  # Date varies
        complexity="simple",
        category="rc-basics",
        tool_count_hint=2,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that prints the numbers 1 to 5",
        expected_output="5",
        complexity="simple",
        category="c-basics",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write an rc script that lists all files in the current directory",
        expected_output=None,
        complexity="simple",
        category="rc-basics",
        tool_count_hint=2,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that calculates 2+2 and prints the result",
        expected_output="4",
        complexity="simple",
        category="c-basics",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write an rc script that prints 'success' if a file exists, 'missing' otherwise",
        expected_output=None,
        complexity="simple",
        category="rc-basics",
        tool_count_hint=2,
    ),
    GRPOTask(
        prompt="Read the /dev/user file and print its contents",
        expected_output="glenda",
        complexity="simple",
        category="sys-info",
        tool_count_hint=1,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that prints the program name (argv[0])",
        expected_output=None,
        complexity="simple",
        category="c-args",
        tool_count_hint=3,
    ),

    # Medium tasks (3-4 tool calls)
    GRPOTask(
        prompt="Write a Plan 9 C program that reads a file called 'input.txt' and prints its contents",
        expected_output=None,
        complexity="medium",
        category="c-io",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write an rc script that creates a temp file, writes 'test data' to it, and reads it back",
        expected_output="test data",
        complexity="medium",
        category="rc-files",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that takes a command line argument and prints it",
        expected_output=None,
        complexity="medium",
        category="c-args",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write an rc script that counts the number of .c files in the current directory",
        expected_output=None,
        complexity="medium",
        category="rc-files",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that calculates the factorial of 5",
        expected_output="120",
        complexity="medium",
        category="c-basics",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write an rc function that prints an error message to stderr and exits",
        expected_output=None,
        complexity="medium",
        category="rc-functions",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that reads from stdin until EOF and counts the lines",
        expected_output=None,
        complexity="medium",
        category="c-io",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write an rc script that finds all .c files and prints their names",
        expected_output=".c",
        complexity="medium",
        category="rc-files",
        tool_count_hint=3,
    ),

    # Complex tasks (5+ tool calls)
    GRPOTask(
        prompt="Write a Plan 9 C program that implements a simple calculator for +, -, *, /",
        expected_output=None,
        complexity="complex",
        category="c-parsing",
        tool_count_hint=5,
    ),
    GRPOTask(
        prompt="Write an rc script that builds a C program and runs it if compilation succeeds",
        expected_output=None,
        complexity="complex",
        category="rc-build",
        tool_count_hint=5,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that uses Bio to read a config file line by line",
        expected_output=None,
        complexity="complex",
        category="c-io",
        tool_count_hint=5,
    ),
    GRPOTask(
        prompt="Write an rc script that processes command line flags using aux/getflags",
        expected_output=None,
        complexity="complex",
        category="rc-flags",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that walks a directory and prints all file names",
        expected_output=None,
        complexity="complex",
        category="c-io",
        tool_count_hint=5,
    ),
    GRPOTask(
        prompt="Write an rc script that creates multiple temp files, processes them, and cleans up",
        expected_output=None,
        complexity="complex",
        category="rc-files",
        tool_count_hint=6,
    ),

    # Additional simple tasks
    GRPOTask(
        prompt="Write a Plan 9 C program that prints 'even' if a number is even, 'odd' otherwise",
        expected_output=None,
        complexity="simple",
        category="c-basics",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write an rc script that echoes all command line arguments",
        expected_output=None,
        complexity="simple",
        category="rc-basics",
        tool_count_hint=2,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that prints 'Hello' 3 times",
        expected_output="Hello",
        complexity="simple",
        category="c-basics",
        tool_count_hint=3,
    ),

    # Additional medium tasks
    GRPOTask(
        prompt="Write a mkfile that compiles a single C file into a binary",
        expected_output=None,
        complexity="medium",
        category="mkfile",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that reverses a string",
        expected_output=None,
        complexity="medium",
        category="c-strings",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write an rc script that checks if a directory exists and creates it if not",
        expected_output=None,
        complexity="medium",
        category="rc-files",
        tool_count_hint=3,
    ),

    # Additional complex tasks
    GRPOTask(
        prompt="Write a Plan 9 C program that sorts an array of integers",
        expected_output=None,
        complexity="complex",
        category="c-algorithms",
        tool_count_hint=5,
    ),
    GRPOTask(
        prompt="Write an rc script that installs a program by compiling and copying to /bin",
        expected_output=None,
        complexity="complex",
        category="rc-install",
        tool_count_hint=5,
    ),

    # =========================================================================
    # Assembly Tasks
    # =========================================================================
    GRPOTask(
        prompt="Write a Plan 9 amd64 assembly function that adds two integers passed as arguments",
        expected_output=None,
        complexity="simple",
        category="asm-basics",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write a Plan 9 assembly calling convention wrapper that calls a C function",
        expected_output=None,
        complexity="medium",
        category="asm-basics",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write SSE2 assembly for computing dot product of two float arrays with horizontal sum",
        expected_output=None,
        complexity="complex",
        category="asm-simd",
        tool_count_hint=5,
    ),
    GRPOTask(
        prompt="Write a Plan 9 assembly routine that outputs text to VGA memory",
        expected_output=None,
        complexity="medium",
        category="asm-lowlevel",
        tool_count_hint=4,
    ),

    # =========================================================================
    # 9P Protocol Tasks
    # =========================================================================
    GRPOTask(
        prompt="Write a minimal 9P file server in C that responds to version and attach messages",
        expected_output=None,
        complexity="complex",
        category="9p-server",
        tool_count_hint=6,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that reads a file from a 9P server using fsmount and open",
        expected_output=None,
        complexity="medium",
        category="9p-client",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Implement directory listing in a 9P file server (handle Tstat and Tread for directories)",
        expected_output=None,
        complexity="complex",
        category="9p-server",
        tool_count_hint=6,
    ),
    GRPOTask(
        prompt="Write a synthetic file server that generates file contents on-the-fly",
        expected_output=None,
        complexity="complex",
        category="9p-server",
        tool_count_hint=6,
    ),

    # =========================================================================
    # Networking Tasks
    # =========================================================================
    GRPOTask(
        prompt="Write a Plan 9 C function that parses dial strings in tcp!host!port format",
        expected_output=None,
        complexity="simple",
        category="net-dial",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that connects to a TCP server and reads a response",
        expected_output=None,
        complexity="medium",
        category="net-dial",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Implement socket read buffering with retry on EINTR in Plan 9 C",
        expected_output=None,
        complexity="medium",
        category="net-io",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C function for length-prefixed binary message protocol",
        expected_output=None,
        complexity="medium",
        category="net-proto",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Implement connection retry with exponential backoff in Plan 9 C",
        expected_output=None,
        complexity="medium",
        category="net-io",
        tool_count_hint=4,
    ),

    # =========================================================================
    # Concurrency Tasks (libthread)
    # =========================================================================
    GRPOTask(
        prompt="Write a Plan 9 C program using channels for inter-thread communication",
        expected_output=None,
        complexity="simple",
        category="thread-chan",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Implement an Alt-based event multiplexer using Plan 9 libthread",
        expected_output=None,
        complexity="medium",
        category="thread-alt",
        tool_count_hint=5,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program demonstrating non-blocking channel receive (nbrecv)",
        expected_output=None,
        complexity="medium",
        category="thread-chan",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Implement a producer/consumer pattern using Plan 9 channels",
        expected_output=None,
        complexity="complex",
        category="thread-pattern",
        tool_count_hint=5,
    ),
    GRPOTask(
        prompt="Write a thread pool in Plan 9 C using proccreate and channels",
        expected_output=None,
        complexity="complex",
        category="thread-pattern",
        tool_count_hint=6,
    ),

    # =========================================================================
    # Graphics Tasks (libdraw)
    # =========================================================================
    GRPOTask(
        prompt="Write a Plan 9 C program that initializes the display and draws a colored rectangle",
        expected_output=None,
        complexity="simple",
        category="draw-basic",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C program that handles mouse and keyboard events",
        expected_output=None,
        complexity="medium",
        category="draw-init",
        tool_count_hint=5,
    ),
    GRPOTask(
        prompt="Implement font rendering with text positioning using Plan 9 libdraw",
        expected_output=None,
        complexity="medium",
        category="draw-text",
        tool_count_hint=5,
    ),
    GRPOTask(
        prompt="Build a render loop with event multiplexing using Alt in Plan 9 graphics",
        expected_output=None,
        complexity="complex",
        category="draw-loop",
        tool_count_hint=6,
    ),

    # =========================================================================
    # UTF-8/Rune Tasks
    # =========================================================================
    GRPOTask(
        prompt="Write a Plan 9 C function that converts a Rune to UTF-8 bytes",
        expected_output=None,
        complexity="simple",
        category="utf-encode",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C function that decodes UTF-8 bytes to a Rune with error handling",
        expected_output=None,
        complexity="medium",
        category="utf-decode",
        tool_count_hint=4,
    ),
    GRPOTask(
        prompt="Write a Plan 9 C function that finds a Rune in a UTF-8 string (like utfrune)",
        expected_output=None,
        complexity="simple",
        category="utf-search",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Write a function that calculates the UTF-8 byte length needed for a Rune array",
        expected_output=None,
        complexity="simple",
        category="utf-len",
        tool_count_hint=3,
    ),
    GRPOTask(
        prompt="Implement a UTF-8 string iterator that yields Runes one at a time",
        expected_output=None,
        complexity="medium",
        category="utf-decode",
        tool_count_hint=4,
    ),
]


def get_tasks_by_complexity(complexity: str | None = None) -> list[GRPOTask]:
    """Get tasks filtered by complexity level.

    Args:
        complexity: 'simple', 'medium', 'complex', or None for all.

    Returns:
        List of matching tasks.
    """
    if complexity is None:
        return GRPO_TASKS
    return [t for t in GRPO_TASKS if t.complexity == complexity]


def get_tasks_by_category(category: str) -> list[GRPOTask]:
    """Get tasks filtered by category.

    Args:
        category: Category prefix to match.

    Returns:
        List of matching tasks.
    """
    return [t for t in GRPO_TASKS if t.category.startswith(category)]


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
    tasks: list[GRPOTask] | None = None,
    include_system: bool = True,
) -> list[dict[str, Any]]:
    """Create a dataset for GRPO training.

    Args:
        tasks: List of tasks to include. Defaults to all GRPO_TASKS.
        include_system: Whether to include system prompt in prompts.

    Returns:
        List of dicts with 'prompt' and 'expected_output' keys.
    """
    if tasks is None:
        tasks = GRPO_TASKS

    data = []
    for task in tasks:
        prompt = ""
        if include_system:
            prompt = format_system_prompt()
        prompt += format_user_prompt(task.prompt)

        data.append({
            "prompt": prompt,
            "expected_output": task.expected_output,
            "complexity": task.complexity,
            "category": task.category,
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


def export_grpo_tasks(output_path: Path, tasks: list[GRPOTask] | None = None) -> int:
    """Export GRPO tasks to JSON file.

    Args:
        output_path: Path to output JSON file.
        tasks: Tasks to export. Defaults to all GRPO_TASKS.

    Returns:
        Number of tasks exported.
    """
    if tasks is None:
        tasks = GRPO_TASKS

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for task in tasks:
        data.append({
            "prompt": task.prompt,
            "expected_output": task.expected_output,
            "complexity": task.complexity,
            "category": task.category,
            "tool_count_hint": task.tool_count_hint,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return len(data)


def load_grpo_tasks(tasks_path: Path) -> list[GRPOTask]:
    """Load GRPO tasks from JSON file.

    Args:
        tasks_path: Path to tasks.json file.

    Returns:
        List of GRPOTask objects.
    """
    if not tasks_path.exists():
        return []

    with open(tasks_path) as f:
        data = json.load(f)

    return [
        GRPOTask(
            prompt=item["prompt"],
            expected_output=item.get("expected_output"),
            complexity=item.get("complexity", "simple"),
            category=item.get("category", ""),
            tool_count_hint=item.get("tool_count_hint", 1),
        )
        for item in data
    ]


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
    tasks: list[GRPOTask] | None = None,
) -> dict[str, Any]:
    """Get statistics about the dataset.

    Args:
        examples_path: Path to examples.json for SFT stats.
        tasks: GRPO tasks for task stats.

    Returns:
        Dict with statistics.
    """
    stats = {}

    if tasks is None:
        tasks = GRPO_TASKS

    # GRPO task stats
    stats["grpo_tasks"] = {
        "total": len(tasks),
        "by_complexity": {
            "simple": len([t for t in tasks if t.complexity == "simple"]),
            "medium": len([t for t in tasks if t.complexity == "medium"]),
            "complex": len([t for t in tasks if t.complexity == "complex"]),
        },
        "by_category": {},
    }

    for task in tasks:
        cat = task.category.split("-")[0] if "-" in task.category else task.category
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
