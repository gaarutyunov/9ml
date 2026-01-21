"""Command-line interface for Plan 9 dataset tooling."""

import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import clone, export, fetch, index

console = Console()


def find_project_root() -> Path:
    """Find the dataset project root by looking for pyproject.toml or data/."""
    cwd = Path.cwd()

    # Check if we're in the dataset directory
    if (cwd / "pyproject.toml").exists() and (cwd / "plan9_dataset").exists():
        return cwd

    # Check if we're in a parent directory with dataset/
    if (cwd / "dataset" / "pyproject.toml").exists():
        return cwd / "dataset"

    # Fall back to current directory
    return cwd


def get_default_paths() -> tuple[Path, Path]:
    """Get default paths for repos and data directories."""
    project_root = find_project_root()
    return project_root / "repos", project_root / "data"


@click.group()
@click.option(
    "--project",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Path to dataset project directory",
)
@click.pass_context
def cli(ctx: click.Context, project: Path | None) -> None:
    """Plan 9 Dataset Preparation Tooling.

    Build fine-tuning datasets for teaching LLMs to work with Plan 9 OS.
    """
    ctx.ensure_object(dict)
    if project:
        ctx.obj["data_dir"] = project / "data"
        ctx.obj["repos_dir"] = project / "repos"
    else:
        repos_dir, data_dir = get_default_paths()
        ctx.obj["data_dir"] = data_dir
        ctx.obj["repos_dir"] = repos_dir


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def fetch_repos(ctx: click.Context, as_json: bool) -> None:
    """Fetch repository list from awesome-plan9."""
    with console.status("Fetching awesome-plan9..."):
        try:
            repos = fetch.get_repos()
        except Exception as e:
            console.print(f"[red]Error fetching repos: {e}[/red]")
            raise SystemExit(1)

    if as_json:
        import json

        print(json.dumps([r.to_dict() for r in repos], indent=2))
        return

    # Group by category
    by_category = fetch.get_repos_by_category(repos)

    console.print(f"\n[bold]Found {len(repos)} repositories in {len(by_category)} categories[/bold]\n")

    for category, category_repos in sorted(by_category.items()):
        console.print(f"[bold cyan]{category}[/bold cyan] ({len(category_repos)})")
        for repo in category_repos[:5]:  # Show first 5
            console.print(f"  {repo.name}: {repo.url}")
        if len(category_repos) > 5:
            console.print(f"  ... and {len(category_repos) - 5} more")
        console.print()


@cli.command()
@click.argument("repos", nargs=-1)
@click.option("--all", "clone_all", is_flag=True, help="Clone all repos from awesome-plan9")
@click.option("--no-update", is_flag=True, help="Skip updating existing repos")
@click.pass_context
def clone_repos(
    ctx: click.Context, repos: tuple[str, ...], clone_all: bool, no_update: bool
) -> None:
    """Clone repositories to local repos/ directory.

    Specify repo names or URLs, or use --all to clone everything.
    """
    repos_dir = ctx.obj["repos_dir"]

    if clone_all:
        with console.status("Fetching repository list..."):
            all_repos = fetch.get_repos()
        to_clone = [(r.name, r.url) for r in all_repos]
    elif repos:
        # Handle both names and URLs
        to_clone = []
        all_repos = None
        for item in repos:
            if item.startswith("http") or item.startswith("git@"):
                # It's a URL
                name = clone._name_from_url(item) if hasattr(clone, "_name_from_url") else item.split("/")[-1]
                to_clone.append((name, item))
            else:
                # It's a name, look up in awesome-plan9
                if all_repos is None:
                    with console.status("Fetching repository list..."):
                        all_repos = fetch.get_repos()

                found = False
                for r in all_repos:
                    if r.name.lower() == item.lower():
                        to_clone.append((r.name, r.url))
                        found = True
                        break

                if not found:
                    console.print(f"[yellow]Warning: '{item}' not found in awesome-plan9[/yellow]")
    else:
        console.print("Specify repo names or use --all")
        raise SystemExit(1)

    console.print(f"Cloning {len(to_clone)} repositories to {repos_dir}")

    success = 0
    failed = 0

    for name, url in to_clone:
        with console.status(f"Cloning {name}..."):
            status = clone.clone_repo(name, url, repos_dir, update_if_exists=not no_update)

        if status.success:
            console.print(f"[green]✓[/green] {name}: {status.message}")
            success += 1
        else:
            console.print(f"[red]✗[/red] {name}: {status.message}")
            failed += 1

    console.print(f"\n[bold]Done: {success} succeeded, {failed} failed[/bold]")


@cli.command()
@click.pass_context
def index_files(ctx: click.Context) -> None:
    """Index all files in cloned repositories."""
    repos_dir = ctx.obj["repos_dir"]
    data_dir = ctx.obj["data_dir"]
    manifest_path = data_dir / "manifest.json"

    if not repos_dir.exists():
        console.print(f"[red]No repos directory at {repos_dir}[/red]")
        console.print("Run 'plan9-dataset clone' first")
        raise SystemExit(1)

    with console.status("Indexing files..."):
        manifest = index.index_all_repos(repos_dir)
        manifest.save(manifest_path)

    console.print(f"\n[bold]Indexed {len(manifest.files)} files[/bold]\n")

    # Show stats
    table = Table(title="Files by Type")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")

    for file_type, count in sorted(manifest.stats.items(), key=lambda x: -x[1]):
        table.add_row(file_type, str(count))

    console.print(table)
    console.print(f"\nManifest saved to {manifest_path}")


@cli.command()
@click.option("--type", "file_types", multiple=True, help="Filter by file type")
@click.option("--repo", "repos", multiple=True, help="Filter by repository")
@click.pass_context
def stats(ctx: click.Context, file_types: tuple[str, ...], repos: tuple[str, ...]) -> None:
    """Show statistics about indexed files and examples."""
    data_dir = ctx.obj["data_dir"]
    repos_dir = ctx.obj["repos_dir"]
    manifest_path = data_dir / "manifest.json"
    examples_path = data_dir / "raw" / "examples.json"

    # Load manifest
    if manifest_path.exists():
        manifest = index.Manifest.load(manifest_path)
        files = index.filter_files(
            manifest,
            file_types=list(file_types) if file_types else None,
            repos=list(repos) if repos else None,
        )

        console.print("\n[bold]Indexed Files[/bold]")
        table = Table()
        table.add_column("Type", style="cyan")
        table.add_column("Count", justify="right")

        by_type = index.get_files_by_type(
            index.Manifest(files=files, stats={}, indexed_at=manifest.indexed_at)
        )
        for file_type, type_files in sorted(by_type.items(), key=lambda x: -len(x[1])):
            table.add_row(file_type, str(len(type_files)))

        console.print(table)
        console.print(f"Total: {len(files)} files")
        console.print(f"Indexed at: {manifest.indexed_at}")
    else:
        console.print("[yellow]No manifest found. Run 'plan9-dataset index' first.[/yellow]")

    # Load examples
    if examples_path.exists():
        examples = export.load_examples(examples_path)
        example_stats = export.get_stats(examples)

        console.print("\n[bold]Training Examples[/bold]")
        table = Table()
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right")

        for category, count in sorted(
            example_stats["by_category"].items(), key=lambda x: -x[1]
        ):
            table.add_row(category, str(count))

        console.print(table)
        console.print(f"Total: {example_stats['total']} examples")
        console.print(f"Avg instruction: {example_stats['avg_instruction_length']} chars")
        console.print(f"Avg response: {example_stats['avg_response_length']} chars")
    else:
        console.print("\n[yellow]No examples found. Add examples to data/raw/examples.json[/yellow]")

    # Show cloned repos
    cloned = clone.list_cloned_repos(repos_dir)
    if cloned:
        console.print(f"\n[bold]Cloned Repositories[/bold]: {len(cloned)}")
        console.print(", ".join(cloned[:10]))
        if len(cloned) > 10:
            console.print(f"... and {len(cloned) - 10} more")


@cli.command()
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["jsonl", "alpaca", "sharegpt", "all"]),
    default="all",
    help="Output format",
)
@click.option("--validate/--no-validate", default=True, help="Validate examples before export")
@click.pass_context
def export_data(ctx: click.Context, fmt: str, validate: bool) -> None:
    """Export curated examples to training formats."""
    data_dir = ctx.obj["data_dir"]
    examples_path = data_dir / "raw" / "examples.json"
    output_dir = data_dir / "processed"

    if not examples_path.exists():
        console.print(f"[red]No examples file at {examples_path}[/red]")
        console.print("Add examples to data/raw/examples.json first")
        raise SystemExit(1)

    examples = export.load_examples(examples_path)

    if not examples:
        console.print("[yellow]No examples to export[/yellow]")
        raise SystemExit(1)

    # Validate
    if validate:
        warnings = export.validate_examples(examples)
        if warnings:
            console.print("\n[yellow]Validation warnings:[/yellow]")
            for warning in warnings[:10]:
                console.print(f"  - {warning}")
            if len(warnings) > 10:
                console.print(f"  ... and {len(warnings) - 10} more")
            console.print()

    # Export
    console.print(f"Exporting {len(examples)} examples...")

    if fmt == "all":
        results = export.export_all(examples, output_dir)
        for format_name, count in results.items():
            console.print(f"[green]✓[/green] {format_name}: {count} examples")
    elif fmt == "jsonl":
        count = export.export_jsonl(examples, output_dir / "train.jsonl")
        console.print(f"[green]✓[/green] jsonl: {count} examples")
    elif fmt == "alpaca":
        count = export.export_alpaca(examples, output_dir / "train_alpaca.json")
        console.print(f"[green]✓[/green] alpaca: {count} examples")
    elif fmt == "sharegpt":
        count = export.export_sharegpt(examples, output_dir / "train_sharegpt.json")
        console.print(f"[green]✓[/green] sharegpt: {count} examples")

    console.print(f"\nExported to {output_dir}")


@cli.command()
@click.argument("pattern", required=False)
@click.option("--type", "file_type", help="Filter by file type")
@click.option("--repo", help="Filter by repository")
@click.option("--limit", default=20, help="Maximum results to show")
@click.pass_context
def search(
    ctx: click.Context, pattern: str | None, file_type: str | None, repo: str | None, limit: int
) -> None:
    """Search indexed files by pattern."""
    data_dir = ctx.obj["data_dir"]
    manifest_path = data_dir / "manifest.json"

    if not manifest_path.exists():
        console.print("[red]No manifest found. Run 'plan9-dataset index' first.[/red]")
        raise SystemExit(1)

    manifest = index.Manifest.load(manifest_path)

    # Filter files
    files = index.filter_files(
        manifest,
        file_types=[file_type] if file_type else None,
        repos=[repo] if repo else None,
    )

    # Search by pattern
    if pattern:
        pattern_lower = pattern.lower()
        files = [f for f in files if pattern_lower in f.path.lower()]

    # Show results
    table = Table(title=f"Search Results ({len(files)} files)")
    table.add_column("Repo", style="cyan")
    table.add_column("Path")
    table.add_column("Type")
    table.add_column("Size", justify="right")

    for f in files[:limit]:
        table.add_row(f.repo, f.path, f.type, f"{f.size:,}")

    console.print(table)

    if len(files) > limit:
        console.print(f"... and {len(files) - limit} more")


@cli.command()
@click.pass_context
def list_repos(ctx: click.Context) -> None:
    """List cloned repositories."""
    repos_dir = ctx.obj["repos_dir"]

    cloned = clone.list_cloned_repos(repos_dir)

    if not cloned:
        console.print("[yellow]No repositories cloned yet.[/yellow]")
        console.print("Run 'plan9-dataset clone' to clone repositories.")
        return

    table = Table(title="Cloned Repositories")
    table.add_column("Name", style="cyan")
    table.add_column("Branch")
    table.add_column("Head")
    table.add_column("Remote")

    for name in cloned:
        info = clone.get_repo_info(repos_dir, name)
        if info and "error" not in info:
            table.add_row(
                info["name"],
                info.get("branch", "?"),
                info.get("head", "?"),
                info.get("remote", "?")[:50] + "..." if info.get("remote") and len(info.get("remote", "")) > 50 else info.get("remote", "?"),
            )
        else:
            table.add_row(name, "?", "?", "error")

    console.print(table)


def _name_from_url(url: str) -> str:
    """Extract a repository name from URL."""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")
    return parts[-1] if parts else "unknown"


@cli.command()
@click.option("--qemu-dir", type=click.Path(path_type=Path, exists=True), help="Path to qemu directory")
@click.option("--quick", is_flag=True, help="Quick syntax validation without QEMU")
@click.option("--category", multiple=True, help="Only validate specific categories")
@click.pass_context
def validate(ctx: click.Context, qemu_dir: Path | None, quick: bool, category: tuple[str, ...]) -> None:
    """Validate dataset examples in Plan 9 QEMU VM.

    Compiles C code and checks rc script syntax in a Plan 9 environment.
    """
    from . import validate as val

    data_dir = ctx.obj["data_dir"]
    examples_path = data_dir / "raw" / "examples.json"

    if not examples_path.exists():
        console.print(f"[red]No examples file at {examples_path}[/red]")
        raise SystemExit(1)

    categories = list(category) if category else None

    if quick:
        console.print("Running quick syntax validation...")
        report = val.quick_validate_syntax(examples_path)
    else:
        # Find QEMU directory
        if qemu_dir is None:
            # Try to find it relative to the project
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
            console.print("[red]QEMU directory not found. Use --qemu-dir or --quick[/red]")
            console.print("The qemu directory should contain 9front.qcow2")
            raise SystemExit(1)

        console.print(f"Validating examples in Plan 9 QEMU VM...")
        console.print(f"QEMU dir: {qemu_dir}")

        try:
            report = val.validate_examples(examples_path, qemu_dir, categories)
        except Exception as e:
            console.print(f"[red]Validation failed: {e}[/red]")
            raise SystemExit(1)

    # Show results
    console.print(f"\n[bold]Validation Results[/bold]")
    console.print(f"Total: {report.total}")
    console.print(f"[green]Valid: {report.valid}[/green]")
    console.print(f"[red]Invalid: {report.invalid}[/red]")
    console.print(f"[yellow]Skipped: {report.skipped}[/yellow]")

    # Show invalid examples
    invalid_results = [r for r in report.results if not r.valid and r.error != "skipped"]
    if invalid_results:
        console.print("\n[bold red]Invalid Examples:[/bold red]")
        for r in invalid_results:
            console.print(f"  [{r.example_index}] {r.category} ({r.code_type}): {r.instruction}")
            console.print(f"      Error: {r.error[:100]}")


# ============================================================================
# GRPO Training Commands
# ============================================================================


@cli.command()
@click.option("--output", type=click.Path(path_type=Path), help="Output path for tasks.json")
@click.option("--complexity", type=click.Choice(["simple", "medium", "complex"]), help="Filter by complexity")
@click.pass_context
def grpo_tasks(ctx: click.Context, output: Path | None, complexity: str | None) -> None:
    """Generate or export GRPO task prompts.

    Creates a tasks.json file with prompts for GRPO training.
    """
    from . import grpo_data

    data_dir = ctx.obj["data_dir"]

    if output is None:
        output = data_dir / "grpo" / "tasks.json"

    # Get tasks
    tasks = grpo_data.get_tasks_by_complexity(complexity)

    # Export
    count = grpo_data.export_grpo_tasks(output, tasks)

    console.print(f"[green]✓[/green] Exported {count} GRPO tasks to {output}")

    # Show breakdown
    table = Table(title="Tasks by Complexity")
    table.add_column("Complexity", style="cyan")
    table.add_column("Count", justify="right")

    simple = len([t for t in tasks if t.complexity == "simple"])
    medium = len([t for t in tasks if t.complexity == "medium"])
    complex_ = len([t for t in tasks if t.complexity == "complex"])

    table.add_row("simple", str(simple))
    table.add_row("medium", str(medium))
    table.add_row("complex", str(complex_))

    console.print(table)


@cli.command()
@click.argument("task", required=True)
@click.option("--qemu-dir", type=click.Path(path_type=Path, exists=True), help="Path to qemu directory")
@click.option("--heuristic", is_flag=True, help="Use heuristic rewards (no VM)")
@click.pass_context
def test_reward(ctx: click.Context, task: str, qemu_dir: Path | None, heuristic: bool) -> None:
    """Test reward function on a single task prompt.

    Example:
        plan9-dataset test-reward "Write a hello world in Plan 9 C" --heuristic
    """
    from . import rewards, tools

    console.print(f"[bold]Testing reward for:[/bold] {task}")

    # Create a sample model output (mock for testing)
    sample_c_code = '#include <u.h>\n#include <libc.h>\n\nvoid\nmain(int argc, char *argv[])\n{\n\tprint("Hello, Plan 9!\\n");\n\texits(nil);\n}\n'
    thinking = tools.format_thinking("I need to write a Plan 9 C program.")
    tool_call = tools.format_tool_call("write_file", {"path": "hello.c", "content": sample_c_code})
    sample_output = f"{thinking}\n{tool_call}"

    console.print("\n[bold]Sample model output:[/bold]")
    console.print(sample_output[:500] + "..." if len(sample_output) > 500 else sample_output)

    # Parse tool calls
    tool_calls = tools.parse_tool_calls(sample_output)
    console.print(f"\n[bold]Parsed tool calls:[/bold] {len(tool_calls)}")
    for call in tool_calls:
        console.print(f"  - {call['name']}: {list(call['params'].keys())}")

    if heuristic:
        # Use heuristic rewards
        console.print("\n[bold]Using heuristic rewards (no VM)[/bold]")

        # Compute style bonus
        style_bonus = rewards.compute_style_bonus(tool_calls)
        reasoning_bonus = rewards.compute_reasoning_bonus(sample_output)

        console.print(f"\n[bold]Reward breakdown:[/bold]")
        console.print(f"  Style bonus: {style_bonus:.2f}")
        console.print(f"  Reasoning bonus: {reasoning_bonus:.2f}")
        console.print(f"  Total (heuristic): {style_bonus + reasoning_bonus:.2f}")
    else:
        # Use VM-based rewards
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
            console.print("[yellow]QEMU directory not found. Using heuristic rewards.[/yellow]")
            console.print("Use --qemu-dir to specify or --heuristic explicitly")

            style_bonus = rewards.compute_style_bonus(tool_calls)
            reasoning_bonus = rewards.compute_reasoning_bonus(sample_output)
            console.print(f"\n[bold]Reward (heuristic):[/bold] {style_bonus + reasoning_bonus:.2f}")
            return

        console.print(f"\n[bold]Using VM-based rewards from {qemu_dir}[/bold]")
        console.print("[yellow]Note: This will boot the Plan 9 VM[/yellow]")

        disk_image = str(qemu_dir / "9front.qcow2")
        shared_image = str(qemu_dir / "shared.img")

        try:
            with rewards.RewardEnvironment(disk_image, shared_image) as env:
                breakdown = env.compute_reward(sample_output, expected_output="Hello, Plan 9!")

                console.print(f"\n[bold]Reward breakdown:[/bold]")
                console.print(f"  Tool success: {breakdown.tool_success_reward:.2f}")
                console.print(f"  Tool failure: {breakdown.tool_failure_penalty:.2f}")
                console.print(f"  Output match: {breakdown.output_match_reward:.2f}")
                console.print(f"  Style bonus: {breakdown.style_bonus:.2f}")
                console.print(f"  Reasoning bonus: {breakdown.reasoning_bonus:.2f}")
                console.print(f"  [bold]Total: {breakdown.total:.2f}[/bold]")

                # Show tool results
                console.print(f"\n[bold]Tool execution results:[/bold]")
                for result in breakdown.tool_results:
                    status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                    console.print(f"  {status} {result.name}")
                    if result.output:
                        console.print(f"      Output: {result.output[:100]}")
                    if result.error:
                        console.print(f"      Error: {result.error}")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise SystemExit(1)


@cli.command()
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["sharegpt", "jsonl", "text"]),
    default="sharegpt",
    help="Output format",
)
@click.option("--output", type=click.Path(path_type=Path), help="Output file path")
@click.pass_context
def export_function_calls(ctx: click.Context, fmt: str, output: Path | None) -> None:
    """Export examples in function calling format for SFT training.

    Converts instruction-response pairs to multi-turn function calling conversations.
    """
    from . import grpo_data

    data_dir = ctx.obj["data_dir"]
    examples_path = data_dir / "raw" / "examples.json"

    if not examples_path.exists():
        console.print(f"[red]No examples file at {examples_path}[/red]")
        raise SystemExit(1)

    if output is None:
        ext = "json" if fmt in ["sharegpt"] else fmt
        output = data_dir / "processed" / f"train_function_calls.{ext}"

    console.print(f"Converting examples to function calling format...")
    count = grpo_data.export_sft_function_calls(examples_path, output, fmt)

    console.print(f"[green]✓[/green] Exported {count} examples to {output}")
    console.print(f"Format: {fmt}")


@cli.command()
@click.pass_context
def grpo_stats(ctx: click.Context) -> None:
    """Show GRPO dataset statistics."""
    from . import grpo_data

    data_dir = ctx.obj["data_dir"]
    examples_path = data_dir / "raw" / "examples.json"
    tasks_path = data_dir / "grpo" / "tasks.json"

    # Get stats
    stats = grpo_data.get_dataset_stats(
        examples_path=examples_path if examples_path.exists() else None,
    )

    # GRPO tasks
    console.print("\n[bold]GRPO Tasks[/bold]")
    grpo_stats = stats.get("grpo_tasks", {})
    console.print(f"Total tasks: {grpo_stats.get('total', 0)}")

    table = Table(title="By Complexity")
    table.add_column("Complexity", style="cyan")
    table.add_column("Count", justify="right")

    for complexity, count in grpo_stats.get("by_complexity", {}).items():
        table.add_row(complexity, str(count))

    console.print(table)

    table = Table(title="By Category")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")

    for category, count in sorted(grpo_stats.get("by_category", {}).items(), key=lambda x: -x[1]):
        table.add_row(category, str(count))

    console.print(table)

    # SFT examples
    if "sft_examples" in stats:
        sft_stats = stats["sft_examples"]
        console.print("\n[bold]SFT Examples[/bold]")
        console.print(f"Total examples: {sft_stats.get('total', 0)}")
        console.print(f"Convertible to function calls: {sft_stats.get('convertible_to_function_calls', 0)}")

    # Load tasks file if it exists
    if tasks_path.exists():
        tasks = grpo_data.load_grpo_tasks(tasks_path)
        console.print(f"\n[bold]Tasks file:[/bold] {tasks_path}")
        console.print(f"Tasks in file: {len(tasks)}")


# ============================================================================
# HuggingFace Publishing Commands
# ============================================================================


@cli.command()
@click.argument("repo_id")
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace API token")
@click.option("--private", is_flag=True, help="Create private repository")
@click.option("--no-card", is_flag=True, help="Skip auto-generating dataset card")
@click.pass_context
def publish_hf(
    ctx: click.Context, repo_id: str, token: str | None, private: bool, no_card: bool
) -> None:
    """Publish dataset to HuggingFace Hub.

    REPO_ID should be in format 'username/dataset-name'.

    Examples:
        plan9-dataset publish-hf myuser/plan9-sft
        plan9-dataset publish-hf myuser/plan9-sft --private
        plan9-dataset publish-hf myuser/plan9-sft --token $HF_TOKEN
    """
    try:
        from . import huggingface
    except ImportError as e:
        console.print("[red]HuggingFace dependencies not installed.[/red]")
        console.print("Install with: pip install 'plan9-dataset[huggingface]'")
        raise SystemExit(1)

    data_dir = ctx.obj["data_dir"]
    examples_path = data_dir / "raw" / "examples.json"

    if not examples_path.exists():
        console.print(f"[red]No examples file at {examples_path}[/red]")
        console.print("Add examples to data/raw/examples.json first")
        raise SystemExit(1)

    console.print(f"Publishing dataset to [bold]{repo_id}[/bold]...")

    try:
        url = huggingface.publish_to_hub(
            examples_path=examples_path,
            repo_id=repo_id,
            token=token,
            private=private,
            include_card=not no_card,
        )
        console.print(f"[green]✓[/green] Dataset published: {url}")

        # Also push GRPO tasks
        console.print("Pushing GRPO tasks...")
        tasks_url = huggingface.push_grpo_tasks_to_hub(repo_id, token)
        console.print(f"[green]✓[/green] GRPO tasks: {tasks_url}")

    except Exception as e:
        console.print(f"[red]Error publishing: {e}[/red]")
        raise SystemExit(1)


@cli.command()
@click.argument("repo_id")
@click.option("--config", type=click.Choice(["sft", "function_calling"]), default="sft")
@click.pass_context
def load_hf(ctx: click.Context, repo_id: str, config: str) -> None:
    """Load and preview dataset from HuggingFace Hub.

    Example:
        plan9-dataset load-hf myuser/plan9-sft --config sft
    """
    try:
        from . import huggingface
    except ImportError:
        console.print("[red]HuggingFace dependencies not installed.[/red]")
        console.print("Install with: pip install 'plan9-dataset[huggingface]'")
        raise SystemExit(1)

    console.print(f"Loading [bold]{repo_id}[/bold] config [bold]{config}[/bold]...")

    try:
        ds = huggingface.load_from_hub(repo_id, config)
        console.print(f"[green]✓[/green] Loaded {len(ds)} examples")

        # Show sample
        console.print("\n[bold]Sample example:[/bold]")
        if len(ds) > 0:
            sample = ds[0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                console.print(f"  [cyan]{key}[/cyan]: {value}")

    except Exception as e:
        console.print(f"[red]Error loading: {e}[/red]")
        raise SystemExit(1)


# ============================================================================
# QEMU Server Commands
# ============================================================================


@cli.command()
@click.option("--qemu-dir", type=click.Path(path_type=Path, exists=True), help="Path to qemu directory")
@click.option("--token", envvar="QEMU_TOKEN", help="Bearer token for authentication")
@click.option("--generate-token", is_flag=True, help="Generate and print a new token")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8080, type=int, help="Port to listen on")
@click.option("--rate-limit", default=60, type=int, help="Max requests per minute per client")
@click.option("--timeout", default=30, type=int, help="Command timeout in seconds")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def serve_qemu(
    ctx: click.Context,
    qemu_dir: Path | None,
    token: str | None,
    generate_token: bool,
    host: str,
    port: int,
    rate_limit: int,
    timeout: int,
    debug: bool,
) -> None:
    """Start remote QEMU API server for Plan 9 execution.

    Exposes Plan 9 VM execution as an HTTP API for remote reward computation.
    Useful for GRPO training from Google Colab or other remote environments.

    Examples:
        # Generate a token
        plan9-dataset serve-qemu --generate-token

        # Start server with token
        plan9-dataset serve-qemu --token SECRET

        # Start with custom port
        plan9-dataset serve-qemu --token SECRET --port 9000
    """
    try:
        from . import qemu_server
    except ImportError as e:
        console.print("[red]Server dependencies not installed.[/red]")
        console.print("Install with: pip install 'plan9-dataset[server]'")
        raise SystemExit(1)

    if generate_token:
        new_token = qemu_server.generate_token()
        console.print(f"[bold]Generated token:[/bold] {new_token}")
        console.print("\nSave this token securely. Use it with:")
        console.print(f"  plan9-dataset serve-qemu --token {new_token}")
        return

    if not token:
        console.print("[red]Token required. Use --token or set QEMU_TOKEN env var.[/red]")
        console.print("Generate a token with: plan9-dataset serve-qemu --generate-token")
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
        console.print("The qemu directory should contain 9front.qcow2")
        raise SystemExit(1)

    disk_image = str(qemu_dir / "9front.qcow2")
    shared_image = str(qemu_dir / "shared.img")

    console.print(f"[bold]Starting QEMU API Server[/bold]")
    console.print(f"  QEMU dir: {qemu_dir}")
    console.print(f"  Host: {host}:{port}")
    console.print(f"  Rate limit: {rate_limit} req/min")
    console.print()
    console.print("[yellow]Server starting... This will boot the Plan 9 VM.[/yellow]")

    try:
        qemu_server.run_server(
            disk_image=disk_image,
            shared_image=shared_image,
            token=token,
            host=host,
            port=port,
            rate_limit=rate_limit,
            timeout=timeout,
            debug=debug,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")
        raise SystemExit(1)


@cli.command()
@click.argument("server_url")
@click.option("--token", envvar="QEMU_TOKEN", help="Bearer token for authentication")
@click.pass_context
def test_remote(ctx: click.Context, server_url: str, token: str | None) -> None:
    """Test connection to remote QEMU API server.

    Example:
        plan9-dataset test-remote https://example.com --token SECRET
    """
    try:
        from . import qemu_client
    except ImportError:
        console.print("[red]Client dependencies not installed.[/red]")
        raise SystemExit(1)

    if not token:
        console.print("[red]Token required. Use --token or set QEMU_TOKEN env var.[/red]")
        raise SystemExit(1)

    console.print(f"Testing connection to [bold]{server_url}[/bold]...")

    try:
        client = qemu_client.RemoteQEMUClient(server_url=server_url, token=token)

        # Health check
        health = client.health()
        console.print(f"[green]✓[/green] Server healthy")
        console.print(f"  VM running: {health.get('vm_running')}")
        console.print(f"  Uptime: {health.get('uptime', 0):.1f}s")

        # Test write + run
        console.print("\nTesting tool execution...")
        result = client.write_file("test.c", '#include <u.h>\n#include <libc.h>\n\nvoid\nmain(int, char**)\n{\n\tprint("test\\n");\n\texits(nil);\n}\n')
        console.print(f"  write_file: {'[green]✓[/green]' if result.success else '[red]✗[/red]'}")

        result = client.run_command("6c test.c && 6l -o test test.6 && ./test")
        console.print(f"  run_command: {'[green]✓[/green]' if result.success else '[red]✗[/red]'}")
        if result.output:
            console.print(f"  Output: {result.output[:100]}")

        # Reset
        success = client.reset()
        console.print(f"  reset: {'[green]✓[/green]' if success else '[red]✗[/red]'}")

        console.print("\n[green]All tests passed![/green]")

    except qemu_client.RemoteQEMUError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
