# Dataset Generation from Repositories Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate a validated JSONL dataset by extracting real code from cloned repositories and validating each example compiles/runs in Plan 9 QEMU.

**Architecture:** Extract code snippets from indexed repos → filter for Plan 9 native code → validate in QEMU VM → output to JSONL. The GRPO tasks become data files, not Python code. Each example includes the actual code from repos as both instruction context and expected response.

**Tech Stack:** Python, QEMU (Plan 9 VM), existing validate.py infrastructure, JSONL format

---

## Task 1: Remove Hardcoded GRPO_TASKS

**Files:**
- Modify: `plan9_dataset/grpo_data.py`

**Step 1: Delete the GRPO_TASKS list**

Remove the entire `GRPO_TASKS: list[GRPOTask] = [...]` definition (lines 39-474) and the `GRPOTask` dataclass.

**Step 2: Update functions to load from JSONL**

Replace `get_tasks_by_complexity()` and `get_tasks_by_category()` to load from `data/grpo/tasks.jsonl`:

```python
def load_tasks(tasks_path: Path | None = None) -> list[dict]:
    """Load tasks from JSONL file."""
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
    """Get tasks filtered by complexity level."""
    tasks = load_tasks(tasks_path)
    if complexity is None:
        return tasks
    return [t for t in tasks if t.get("complexity") == complexity]


def get_tasks_by_category(category: str, tasks_path: Path | None = None) -> list[dict]:
    """Get tasks filtered by category prefix."""
    tasks = load_tasks(tasks_path)
    return [t for t in tasks if t.get("category", "").startswith(category)]
```

**Step 3: Run import test**

```bash
python3 -c "from plan9_dataset.grpo_data import load_tasks; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add plan9_dataset/grpo_data.py
git commit -m "refactor: remove hardcoded GRPO_TASKS, load from JSONL"
```

---

## Task 2: Create Code Extractor Module

**Files:**
- Create: `plan9_dataset/extract.py`

**Step 1: Create the extractor module**

```python
"""
Extract code examples from cloned repositories.

Identifies Plan 9 native code (C with <u.h>, rc scripts) and extracts
complete, self-contained examples suitable for the training dataset.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from .index import Manifest, FileInfo, filter_files


@dataclass
class CodeExample:
    """A code example extracted from a repository."""
    instruction: str
    response: str
    source: str  # repo/path
    category: str
    file_type: str  # c, rc, mkfile, asm

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "response": self.response,
            "source": self.source,
            "category": self.category,
            "file_type": self.file_type,
        }


def is_plan9_c_code(content: str) -> bool:
    """Check if C code is Plan 9 native (not portable POSIX)."""
    # Must have Plan 9 headers
    if "#include <u.h>" not in content:
        return False
    # Should not have POSIX headers
    posix_headers = ["<stdio.h>", "<stdlib.h>", "<string.h>", "<unistd.h>"]
    for header in posix_headers:
        if header in content:
            return False
    return True


def is_complete_c_program(content: str) -> bool:
    """Check if C code is a complete, compilable program."""
    # Has main function
    if not re.search(r"void\s+main\s*\(", content):
        return False
    # Has required headers
    if "#include <u.h>" not in content or "#include <libc.h>" not in content:
        return False
    return True


def is_complete_rc_script(content: str) -> bool:
    """Check if rc script is complete and executable."""
    lines = content.strip().split("\n")
    if not lines:
        return False
    # Should start with shebang or be a function definition
    first_line = lines[0].strip()
    return first_line.startswith("#!/bin/rc") or first_line.startswith("fn ")


def categorize_c_code(content: str, path: str) -> str:
    """Determine category for C code based on content."""
    if "threadmain" in content or "#include <thread.h>" in content:
        return "c-threads"
    if "Biobuf" in content or "#include <bio.h>" in content:
        return "c-io"
    if "#include <draw.h>" in content:
        return "c-graphics"
    if "Channel" in content:
        return "c-channels"
    if "ARGBEGIN" in content:
        return "c-args"
    if "Fcall" in content or "Fid" in content:
        return "c-9p"
    if re.search(r"dial\s*\(", content):
        return "c-network"
    return "c-general"


def categorize_rc_script(content: str, path: str) -> str:
    """Determine category for rc script based on content."""
    if "aux/getflags" in content:
        return "rc-flags"
    if "fn " in content and "rfork" not in content:
        return "rc-functions"
    if "mount" in content or "srv" in content:
        return "rc-mount"
    if "git/" in path:
        return "rc-git"
    return "rc-general"


def generate_instruction(file_type: str, category: str, content: str, path: str) -> str:
    """Generate an instruction prompt for the code example."""
    filename = Path(path).name

    if file_type == "c":
        # Extract a brief description from comments if available
        comment_match = re.search(r"/\*\s*(.+?)\s*\*/", content[:500])
        if comment_match:
            desc = comment_match.group(1).split("\n")[0].strip()
            return f"Write a Plan 9 C program that {desc.lower()}"

        # Generate based on category
        if category == "c-threads":
            return f"Write a Plan 9 C program using libthread (like {filename})"
        elif category == "c-io":
            return f"Write a Plan 9 C program for buffered I/O (like {filename})"
        elif category == "c-9p":
            return f"Write a Plan 9 C program implementing 9P protocol (like {filename})"
        else:
            return f"Write a Plan 9 C program similar to {filename}"

    elif file_type == "rc":
        if category == "rc-flags":
            return f"Write an rc script with command-line flag parsing (like {filename})"
        elif category == "rc-functions":
            return f"Write rc shell functions (like {filename})"
        else:
            return f"Write an rc shell script similar to {filename}"

    elif file_type == "mkfile":
        return f"Write a Plan 9 mkfile for building a project (like {filename})"

    return f"Write Plan 9 code similar to {filename}"


def extract_from_file(
    file_info: FileInfo,
    repos_dir: Path,
) -> CodeExample | None:
    """Extract a code example from a single file.

    Returns None if the file is not suitable for the dataset.
    """
    file_path = repos_dir / file_info.path

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    # Skip empty or very short files
    if len(content.strip()) < 50:
        return None

    # Skip very large files (likely not good examples)
    if len(content) > 10000:
        return None

    file_type = file_info.type

    # Validate based on file type
    if file_type == "c":
        if not is_plan9_c_code(content):
            return None
        category = categorize_c_code(content, file_info.path)
    elif file_type == "rc":
        if not is_complete_rc_script(content):
            return None
        category = categorize_rc_script(content, file_info.path)
    elif file_type == "mkfile":
        category = "mkfile"
    else:
        return None

    instruction = generate_instruction(file_type, category, content, file_info.path)

    return CodeExample(
        instruction=instruction,
        response=content.strip(),
        source=file_info.path,
        category=category,
        file_type=file_type,
    )


def extract_all(
    manifest: Manifest,
    repos_dir: Path,
    file_types: list[str] | None = None,
) -> Generator[CodeExample, None, None]:
    """Extract all code examples from indexed files.

    Args:
        manifest: Index manifest with file information.
        repos_dir: Path to cloned repositories.
        file_types: Filter by file types (None = c, rc, mkfile).

    Yields:
        CodeExample objects for each valid file.
    """
    if file_types is None:
        file_types = ["c", "rc", "mkfile"]

    files = filter_files(manifest, file_types=file_types)

    for file_info in files:
        example = extract_from_file(file_info, repos_dir)
        if example:
            yield example


def save_examples_jsonl(examples: list[CodeExample], output_path: Path) -> int:
    """Save examples to JSONL file.

    Returns number of examples saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
            count += 1

    return count


def load_examples_jsonl(input_path: Path) -> list[dict]:
    """Load examples from JSONL file."""
    if not input_path.exists():
        return []

    examples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples
```

**Step 2: Run import test**

```bash
python3 -c "from plan9_dataset.extract import extract_all, is_plan9_c_code; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add plan9_dataset/extract.py
git commit -m "feat: add code extractor for Plan 9 repos"
```

---

## Task 3: Create QEMU Validator for Extracted Code

**Files:**
- Modify: `plan9_dataset/validate.py`
- Create: `plan9_dataset/validate_dataset.py`

**Step 1: Create the dataset validator module**

```python
"""
Validate extracted code examples in Plan 9 QEMU VM.

Runs each code example through the VM to verify it compiles/runs correctly.
Only validated examples make it into the final dataset.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from .extract import CodeExample, load_examples_jsonl
from .validate import FATDisk, Plan9VM, make_compilable_c


@dataclass
class ValidationResult:
    """Result of validating a code example."""
    example: dict
    valid: bool
    error: str = ""
    output: str = ""


def validate_c_code(
    code: str,
    disk: FATDisk,
    vm: Plan9VM,
    timeout: int = 30,
) -> tuple[bool, str, str]:
    """Validate C code by compiling and optionally running.

    Returns (valid, error, output).
    """
    # Make code compilable if needed
    full_code = make_compilable_c(code)

    # Write to disk
    if not disk.copy_content(full_code, "test.c"):
        return False, "Failed to write code to disk", ""

    # Compile
    vm.clear_output()
    output = vm.run_command("6c -w test.c 2>&1", timeout=timeout)

    if "error" in output.lower() or "undefined" in output.lower():
        return False, f"Compilation error: {output[:200]}", output

    # Link
    output = vm.run_command("6l -o test test.6 2>&1", timeout=timeout)

    if "error" in output.lower() or "undefined" in output.lower():
        return False, f"Link error: {output[:200]}", output

    # Clean up
    vm.run_command("rm -f test.c test.6 test", timeout=5)

    return True, "", output


def validate_rc_script(
    code: str,
    disk: FATDisk,
    vm: Plan9VM,
    timeout: int = 10,
) -> tuple[bool, str, str]:
    """Validate rc script syntax.

    Returns (valid, error, output).
    """
    # Write to disk
    if not disk.copy_content(code, "test.rc"):
        return False, "Failed to write script to disk", ""

    # Syntax check with rc -n
    vm.clear_output()
    output = vm.run_command("rc -n test.rc 2>&1", timeout=timeout)

    if "error" in output.lower() or "syntax" in output.lower():
        return False, f"Syntax error: {output[:200]}", output

    # Clean up
    vm.run_command("rm -f test.rc", timeout=5)

    return True, "", output


def validate_example(
    example: dict,
    disk: FATDisk,
    vm: Plan9VM,
) -> ValidationResult:
    """Validate a single code example."""
    file_type = example.get("file_type", "")
    response = example.get("response", "")

    if file_type == "c":
        valid, error, output = validate_c_code(response, disk, vm)
    elif file_type == "rc":
        valid, error, output = validate_rc_script(response, disk, vm)
    elif file_type == "mkfile":
        # Mkfiles are harder to validate, skip for now
        valid, error, output = True, "", ""
    else:
        valid, error, output = False, f"Unknown file type: {file_type}", ""

    return ValidationResult(
        example=example,
        valid=valid,
        error=error,
        output=output,
    )


def validate_all(
    examples: list[dict],
    disk_image: str,
    shared_image: str,
    debug: bool = False,
) -> Generator[ValidationResult, None, None]:
    """Validate all examples in QEMU VM.

    Yields ValidationResult for each example.
    """
    # Create fresh shared disk
    disk = FATDisk(shared_image)
    if not disk.create(64):
        raise RuntimeError("Failed to create shared disk")

    # Start VM
    vm = Plan9VM(disk_image, shared_image, debug=debug)

    try:
        print("Booting Plan 9 VM...")
        if not vm.boot():
            raise RuntimeError("Failed to boot VM")
        print("VM booted successfully")

        for i, example in enumerate(examples):
            print(f"Validating {i+1}/{len(examples)}: {example.get('source', 'unknown')[:50]}...")

            result = validate_example(example, disk, vm)
            yield result

            if not result.valid:
                print(f"  INVALID: {result.error[:60]}")
            else:
                print(f"  OK")

            # Small delay between validations
            time.sleep(0.1)

    finally:
        print("Shutting down VM...")
        vm.shutdown()


def validate_and_save(
    input_path: Path,
    output_path: Path,
    disk_image: str,
    shared_image: str,
    debug: bool = False,
) -> tuple[int, int]:
    """Validate examples and save only valid ones.

    Args:
        input_path: Path to input JSONL with candidate examples.
        output_path: Path to output JSONL with validated examples.
        disk_image: Path to 9front QCOW2 image.
        shared_image: Path to FAT shared disk image.
        debug: Enable debug logging.

    Returns:
        Tuple of (valid_count, invalid_count).
    """
    examples = load_examples_jsonl(input_path)

    if not examples:
        return 0, 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    invalid_count = 0

    with open(output_path, "w") as f:
        for result in validate_all(examples, disk_image, shared_image, debug):
            if result.valid:
                f.write(json.dumps(result.example) + "\n")
                valid_count += 1
            else:
                invalid_count += 1

    return valid_count, invalid_count
```

**Step 2: Run import test**

```bash
python3 -c "from plan9_dataset.validate_dataset import validate_and_save; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add plan9_dataset/validate_dataset.py
git commit -m "feat: add QEMU validator for extracted code examples"
```

---

## Task 4: Add CLI Commands for Dataset Generation

**Files:**
- Modify: `plan9_dataset/cli.py`

**Step 1: Add extract command**

Add to cli.py after the existing commands:

```python
@cli.command()
@click.option("--output", type=click.Path(path_type=Path), help="Output JSONL path")
@click.option("--type", "file_types", multiple=True, help="File types to extract (c, rc, mkfile)")
@click.pass_context
def extract(ctx: click.Context, output: Path | None, file_types: tuple[str, ...]) -> None:
    """Extract code examples from cloned repositories.

    Extracts Plan 9 native code (C with <u.h>, rc scripts) from indexed repos.
    Output is candidate examples that need validation.
    """
    from . import extract as ext
    from .index import Manifest

    data_dir = ctx.obj["data_dir"]
    repos_dir = ctx.obj["repos_dir"]
    manifest_path = data_dir / "manifest.json"

    if output is None:
        output = data_dir / "candidates.jsonl"

    if not manifest_path.exists():
        console.print("[red]No manifest found. Run 'plan9-dataset index' first.[/red]")
        raise SystemExit(1)

    manifest = Manifest.load(manifest_path)
    types = list(file_types) if file_types else None

    console.print(f"Extracting code examples from {len(manifest.files)} indexed files...")

    examples = list(ext.extract_all(manifest, repos_dir, types))
    count = ext.save_examples_jsonl(examples, output)

    console.print(f"[green]✓[/green] Extracted {count} candidate examples to {output}")

    # Show breakdown
    by_type = {}
    by_category = {}
    for ex in examples:
        t = ex.file_type
        c = ex.category
        by_type[t] = by_type.get(t, 0) + 1
        by_category[c] = by_category.get(c, 0) + 1

    table = Table(title="By Type")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    for t, n in sorted(by_type.items()):
        table.add_row(t, str(n))
    console.print(table)

    table = Table(title="By Category")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    for c, n in sorted(by_category.items(), key=lambda x: -x[1]):
        table.add_row(c, str(n))
    console.print(table)
```

**Step 2: Add validate-dataset command**

```python
@cli.command()
@click.option("--input", "input_path", type=click.Path(path_type=Path, exists=True), help="Input JSONL path")
@click.option("--output", type=click.Path(path_type=Path), help="Output JSONL path")
@click.option("--qemu-dir", type=click.Path(path_type=Path, exists=True), help="Path to qemu directory")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def validate_dataset(
    ctx: click.Context,
    input_path: Path | None,
    output: Path | None,
    qemu_dir: Path | None,
    debug: bool,
) -> None:
    """Validate extracted examples in Plan 9 QEMU VM.

    Compiles C code and checks rc script syntax. Only valid examples
    are written to the output file.
    """
    from . import validate_dataset as val

    data_dir = ctx.obj["data_dir"]

    if input_path is None:
        input_path = data_dir / "candidates.jsonl"
    if output is None:
        output = data_dir / "validated.jsonl"

    if not input_path.exists():
        console.print(f"[red]No input file at {input_path}[/red]")
        console.print("Run 'plan9-dataset extract' first.")
        raise SystemExit(1)

    # Find QEMU directory
    if qemu_dir is None:
        possible_paths = [
            Path.cwd().parent / "qemu",
            Path.cwd().parent.parent / "qemu",
            Path("/home/ubuntu/9ml/qemu"),
        ]
        for p in possible_paths:
            if (p / "9front.qcow2").exists():
                qemu_dir = p
                break

    if qemu_dir is None or not (qemu_dir / "9front.qcow2").exists():
        console.print("[red]QEMU directory not found. Use --qemu-dir[/red]")
        raise SystemExit(1)

    disk_image = str(qemu_dir / "9front.qcow2")
    shared_image = str(qemu_dir / "shared.img")

    console.print(f"Validating examples from {input_path}...")
    console.print(f"QEMU dir: {qemu_dir}")

    valid_count, invalid_count = val.validate_and_save(
        input_path, output, disk_image, shared_image, debug
    )

    console.print(f"\n[bold]Validation Results[/bold]")
    console.print(f"[green]Valid: {valid_count}[/green]")
    console.print(f"[red]Invalid: {invalid_count}[/red]")
    console.print(f"\nValidated examples saved to {output}")
```

**Step 3: Add generate-dataset command (combines extract + validate)**

```python
@cli.command()
@click.option("--output", type=click.Path(path_type=Path), help="Output JSONL path")
@click.option("--qemu-dir", type=click.Path(path_type=Path, exists=True), help="Path to qemu directory")
@click.option("--skip-validation", is_flag=True, help="Skip QEMU validation (faster, but may include broken code)")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def generate_dataset(
    ctx: click.Context,
    output: Path | None,
    qemu_dir: Path | None,
    skip_validation: bool,
    debug: bool,
) -> None:
    """Generate dataset from repositories (extract + validate).

    This is the main command for dataset generation:
    1. Extract code examples from cloned repos
    2. Validate each example in QEMU (unless --skip-validation)
    3. Output validated examples to JSONL
    """
    from . import extract as ext
    from . import validate_dataset as val
    from .index import Manifest

    data_dir = ctx.obj["data_dir"]
    repos_dir = ctx.obj["repos_dir"]
    manifest_path = data_dir / "manifest.json"

    if output is None:
        output = data_dir / "dataset.jsonl"

    if not manifest_path.exists():
        console.print("[red]No manifest found. Run 'plan9-dataset index' first.[/red]")
        raise SystemExit(1)

    # Step 1: Extract
    console.print("[bold]Step 1: Extracting code examples...[/bold]")
    manifest = Manifest.load(manifest_path)
    examples = list(ext.extract_all(manifest, repos_dir))
    console.print(f"Extracted {len(examples)} candidate examples")

    if not examples:
        console.print("[yellow]No examples found.[/yellow]")
        return

    # Step 2: Validate (optional)
    if skip_validation:
        console.print("[yellow]Skipping validation (--skip-validation)[/yellow]")
        count = ext.save_examples_jsonl(examples, output)
        console.print(f"[green]✓[/green] Saved {count} examples to {output}")
        return

    # Find QEMU directory
    if qemu_dir is None:
        possible_paths = [
            Path.cwd().parent / "qemu",
            Path.cwd().parent.parent / "qemu",
            Path("/home/ubuntu/9ml/qemu"),
        ]
        for p in possible_paths:
            if (p / "9front.qcow2").exists():
                qemu_dir = p
                break

    if qemu_dir is None or not (qemu_dir / "9front.qcow2").exists():
        console.print("[red]QEMU directory not found. Use --qemu-dir or --skip-validation[/red]")
        raise SystemExit(1)

    # Save candidates first
    candidates_path = data_dir / "candidates.jsonl"
    ext.save_examples_jsonl(examples, candidates_path)

    console.print(f"\n[bold]Step 2: Validating in QEMU...[/bold]")
    disk_image = str(qemu_dir / "9front.qcow2")
    shared_image = str(qemu_dir / "shared.img")

    valid_count, invalid_count = val.validate_and_save(
        candidates_path, output, disk_image, shared_image, debug
    )

    console.print(f"\n[bold]Dataset Generation Complete[/bold]")
    console.print(f"[green]Valid examples: {valid_count}[/green]")
    console.print(f"[red]Invalid examples: {invalid_count}[/red]")
    console.print(f"\nDataset saved to {output}")
```

**Step 4: Test CLI help**

```bash
cd /home/ubuntu/9ml/.worktrees/dataset/dataset
python3 -m plan9_dataset.cli --help
```

Expected: Shows extract, validate-dataset, generate-dataset commands

**Step 5: Commit**

```bash
git add plan9_dataset/cli.py
git commit -m "feat: add CLI commands for dataset generation"
```

---

## Task 5: Update HuggingFace Publisher to Use JSONL

**Files:**
- Modify: `plan9_dataset/huggingface.py`

**Step 1: Update examples_to_hf_dataset to load JSONL**

Replace the function to load from JSONL instead of examples.json:

```python
def examples_to_hf_dataset(dataset_path: Path) -> "DatasetDict":
    """Convert dataset JSONL to HuggingFace DatasetDict.

    Args:
        dataset_path: Path to validated dataset.jsonl file.

    Returns:
        DatasetDict with 'train' split.
    """
    from datasets import Dataset, DatasetDict

    # Load JSONL
    examples = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Build dataset
    data = {
        "instruction": [],
        "response": [],
        "category": [],
        "source": [],
        "file_type": [],
    }

    for ex in examples:
        data["instruction"].append(ex.get("instruction", ""))
        data["response"].append(ex.get("response", ""))
        data["category"].append(ex.get("category", ""))
        data["source"].append(ex.get("source", ""))
        data["file_type"].append(ex.get("file_type", ""))

    dataset = Dataset.from_dict(data)

    return DatasetDict({"train": dataset})
```

**Step 2: Update publish_to_hub signature**

```python
def publish_to_hub(
    dataset_path: Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Publish dataset to HuggingFace Hub.

    Args:
        dataset_path: Path to validated dataset.jsonl file.
        repo_id: Repository ID in format "username/dataset-name".
        token: HuggingFace API token.
        private: Whether to create a private repository.

    Returns:
        URL of the published dataset.
    """
    dataset_dict = examples_to_hf_dataset(dataset_path)

    dataset_dict.push_to_hub(
        repo_id,
        token=token,
        private=private,
    )

    return f"https://huggingface.co/datasets/{repo_id}"
```

**Step 3: Update CLI publish-hf command**

Update the publish_hf function in cli.py:

```python
@cli.command()
@click.argument("repo_id")
@click.option("--input", "input_path", type=click.Path(path_type=Path, exists=True), help="Input dataset JSONL")
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace API token")
@click.option("--private", is_flag=True, help="Create private repository")
@click.pass_context
def publish_hf(
    ctx: click.Context, repo_id: str, input_path: Path | None, token: str | None, private: bool
) -> None:
    """Publish dataset to HuggingFace Hub."""
    from . import huggingface

    data_dir = ctx.obj["data_dir"]

    if input_path is None:
        input_path = data_dir / "dataset.jsonl"

    if not input_path.exists():
        console.print(f"[red]No dataset file at {input_path}[/red]")
        console.print("Run 'plan9-dataset generate-dataset' first.")
        raise SystemExit(1)

    console.print(f"Publishing dataset to [bold]{repo_id}[/bold]...")

    url = huggingface.publish_to_hub(input_path, repo_id, token, private)
    console.print(f"[green]✓[/green] Dataset published: {url}")
```

**Step 4: Test import**

```bash
python3 -c "from plan9_dataset.huggingface import examples_to_hf_dataset; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add plan9_dataset/huggingface.py plan9_dataset/cli.py
git commit -m "refactor: update HuggingFace publisher to use JSONL"
```

---

## Task 6: Clean Up Unused Code

**Files:**
- Modify: `plan9_dataset/grpo_data.py` - remove unused functions
- Delete: `data/grpo/tasks.json` - no longer needed (will use JSONL)

**Step 1: Simplify grpo_data.py**

Remove the `FunctionCallExample`, `convert_example_to_function_calls`, and related functions that depend on hardcoded tasks. Keep only:
- `load_tasks()`
- `get_tasks_by_complexity()`
- `get_tasks_by_category()`
- `get_dataset_stats()`

**Step 2: Delete old tasks.json**

```bash
rm -f data/grpo/tasks.json
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: clean up unused hardcoded task code"
```

---

## Task 7: Integration Test

**Step 1: Run full pipeline**

```bash
cd /home/ubuntu/9ml/.worktrees/dataset/dataset

# Clone repos (if not done)
plan9-dataset clone --all

# Index files
plan9-dataset index

# Generate dataset (extract + validate)
plan9-dataset generate-dataset --debug

# Check output
head -5 data/dataset.jsonl
wc -l data/dataset.jsonl
```

**Step 2: Verify dataset structure**

```bash
python3 -c "
import json
with open('data/dataset.jsonl') as f:
    ex = json.loads(f.readline())
    print('Keys:', list(ex.keys()))
    print('Category:', ex.get('category'))
    print('Source:', ex.get('source'))
    print('Response preview:', ex.get('response', '')[:100])
"
```

Expected: Shows instruction, response, category, source, file_type

**Step 3: Commit final state**

```bash
git add -A
git commit -m "test: verify dataset generation pipeline"
```

---

## Summary

The new pipeline is:

```
plan9-dataset clone --all
    ↓
plan9-dataset index
    ↓
plan9-dataset generate-dataset
    ↓
[repos/] → [extract] → [candidates.jsonl] → [QEMU validate] → [dataset.jsonl]
    ↓
plan9-dataset publish-hf USER/repo
```

**Key changes:**
1. No hardcoded Python tasks - all data in JSONL
2. Real code extracted from actual Plan 9 repositories
3. Every example validated by compiling/running in QEMU
4. Only code that actually works makes it into the dataset
