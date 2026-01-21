"""
Extract source code from repos into a knowledge dataset for LoRA post-training.

Usage:
    python -m plan9_dataset.extract_knowledge repos/ data/knowledge.jsonl
"""

import json
import sys
from pathlib import Path


def is_plan9_c(content: str) -> bool:
    """Check if file is Plan 9 C (has u.h or libc.h)."""
    return "#include <u.h>" in content or "#include <libc.h>" in content


def is_rc_script(content: str) -> bool:
    """Check if file is an rc script."""
    return content.startswith("#!/bin/rc") or "\nfn " in content


def is_mkfile(content: str, filename: str) -> bool:
    """Check if file is a Plan 9 mkfile."""
    return filename == "mkfile" and "</$objtype/mkfile" in content


def extract_from_repos(repos_dir: Path) -> list[dict]:
    """Extract all Plan 9 source files from repos."""
    items = []

    for repo_path in repos_dir.iterdir():
        if not repo_path.is_dir():
            continue

        repo_name = repo_path.name

        # Find all relevant files
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Skip large files, binaries, etc.
            if file_path.stat().st_size > 100_000:  # 100KB max
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if not content.strip():
                continue

            filename = file_path.name
            rel_path = str(file_path.relative_to(repo_path))

            # Determine file type
            file_type = None

            if filename.endswith(".c") and is_plan9_c(content):
                file_type = "c"
            elif filename.endswith(".h") and is_plan9_c(content):
                file_type = "h"
            elif filename.endswith(".s") and ("TEXT " in content or "DATA " in content):
                file_type = "asm"
            elif filename == "mkfile" and is_mkfile(content, filename):
                file_type = "mkfile"
            elif (filename.endswith(".rc") or is_rc_script(content)):
                file_type = "rc"

            if file_type:
                items.append({
                    "text": content,
                    "source": f"{repo_name}/{rel_path}",
                    "file_type": file_type,
                })

    return items


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m plan9_dataset.extract_knowledge REPOS_DIR OUTPUT.jsonl")
        sys.exit(1)

    repos_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not repos_dir.exists():
        print(f"Error: {repos_dir} does not exist")
        sys.exit(1)

    print(f"Extracting from {repos_dir}...")
    items = extract_from_repos(repos_dir)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    # Stats
    by_type = {}
    total_chars = 0
    for item in items:
        ft = item["file_type"]
        by_type[ft] = by_type.get(ft, 0) + 1
        total_chars += len(item["text"])

    print(f"\nExtracted {len(items)} files:")
    for ft, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {ft}: {count}")
    print(f"\nTotal: {total_chars:,} characters")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
