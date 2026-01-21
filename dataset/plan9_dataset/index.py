"""Index files from cloned repositories."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# File type detection based on extensions and patterns
FILE_TYPES = {
    "c": [".c"],
    "header": [".h"],
    "rc": [".rc"],
    "asm": [".s", ".S"],
    "man": [".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"],
    "mkfile": ["mkfile"],
    "doc": [".txt", ".md", ".rst"],
}

# Files to skip
SKIP_PATTERNS = [
    ".git",
    "__pycache__",
    ".pyc",
    "node_modules",
    ".DS_Store",
    "Thumbs.db",
]

# Maximum file size to index (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


@dataclass
class FileInfo:
    """Information about an indexed file."""

    path: str
    repo: str
    type: str
    size: int
    description: str | None = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "repo": self.repo,
            "type": self.type,
            "size": self.size,
            "description": self.description,
        }


@dataclass
class Manifest:
    """Manifest of all indexed files."""

    files: list[FileInfo] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    indexed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "files": [f.to_dict() for f in self.files],
            "stats": self.stats,
            "indexed_at": self.indexed_at,
        }

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load manifest from JSON file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        manifest = cls(indexed_at=data.get("indexed_at", ""))
        manifest.stats = data.get("stats", {})

        for file_dict in data.get("files", []):
            manifest.files.append(FileInfo(**file_dict))

        return manifest


def detect_file_type(file_path: Path) -> str | None:
    """Detect the type of a file based on extension or content.

    Returns file type string or None if type is unknown/uninteresting.
    """
    name = file_path.name
    ext = file_path.suffix.lower()

    # Check for mkfile (exact name match)
    if name == "mkfile":
        return "mkfile"

    # Check by extension
    for file_type, extensions in FILE_TYPES.items():
        if ext in extensions:
            return file_type

    # Check for rc scripts without extension (shebang check)
    if not ext and file_path.is_file():
        try:
            with open(file_path, "rb") as f:
                first_line = f.readline(100)
                if b"#!/bin/rc" in first_line or b"#!rc" in first_line:
                    return "rc"
        except (OSError, PermissionError):
            pass

    return None


def should_skip(path: Path) -> bool:
    """Check if a path should be skipped during indexing."""
    name = path.name
    return any(skip in name for skip in SKIP_PATTERNS)


def index_repo(repo_path: Path, repo_name: str) -> list[FileInfo]:
    """Index all relevant files in a repository.

    Args:
        repo_path: Path to the repository
        repo_name: Name of the repository

    Returns:
        List of FileInfo for each indexed file
    """
    files = []

    for root, dirs, filenames in os.walk(repo_path):
        root_path = Path(root)

        # Skip hidden and unwanted directories
        dirs[:] = [d for d in dirs if not should_skip(root_path / d)]

        for filename in filenames:
            file_path = root_path / filename

            # Skip unwanted files
            if should_skip(file_path):
                continue

            # Check file size
            try:
                size = file_path.stat().st_size
                if size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            # Detect file type
            file_type = detect_file_type(file_path)
            if file_type is None:
                continue

            # Create relative path from repos directory
            rel_path = str(file_path.relative_to(repo_path.parent))

            files.append(
                FileInfo(
                    path=rel_path,
                    repo=repo_name,
                    type=file_type,
                    size=size,
                )
            )

    return files


def index_all_repos(repos_dir: Path) -> Manifest:
    """Index all files in all cloned repositories.

    Args:
        repos_dir: Directory containing cloned repositories

    Returns:
        Manifest with all indexed files
    """
    manifest = Manifest(indexed_at=datetime.now().isoformat())

    if not repos_dir.exists():
        return manifest

    for repo_dir in sorted(repos_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        if should_skip(repo_dir):
            continue

        repo_name = repo_dir.name
        files = index_repo(repo_dir, repo_name)
        manifest.files.extend(files)

    # Calculate stats
    for file_info in manifest.files:
        file_type = file_info.type
        manifest.stats[file_type] = manifest.stats.get(file_type, 0) + 1

    return manifest


def filter_files(
    manifest: Manifest,
    file_types: list[str] | None = None,
    repos: list[str] | None = None,
    min_size: int = 0,
    max_size: int = MAX_FILE_SIZE,
) -> list[FileInfo]:
    """Filter files from manifest based on criteria.

    Args:
        manifest: The manifest to filter
        file_types: List of file types to include (None = all)
        repos: List of repo names to include (None = all)
        min_size: Minimum file size in bytes
        max_size: Maximum file size in bytes

    Returns:
        Filtered list of FileInfo
    """
    result = []

    for file_info in manifest.files:
        # Filter by type
        if file_types and file_info.type not in file_types:
            continue

        # Filter by repo
        if repos and file_info.repo not in repos:
            continue

        # Filter by size
        if file_info.size < min_size or file_info.size > max_size:
            continue

        result.append(file_info)

    return result


def get_files_by_type(manifest: Manifest) -> dict[str, list[FileInfo]]:
    """Group files by type."""
    by_type: dict[str, list[FileInfo]] = {}
    for file_info in manifest.files:
        if file_info.type not in by_type:
            by_type[file_info.type] = []
        by_type[file_info.type].append(file_info)
    return by_type


def get_files_by_repo(manifest: Manifest) -> dict[str, list[FileInfo]]:
    """Group files by repository."""
    by_repo: dict[str, list[FileInfo]] = {}
    for file_info in manifest.files:
        if file_info.repo not in by_repo:
            by_repo[file_info.repo] = []
        by_repo[file_info.repo].append(file_info)
    return by_repo
