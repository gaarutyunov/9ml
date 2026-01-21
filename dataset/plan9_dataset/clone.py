"""Clone and manage Plan 9 repositories."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from git import GitCommandError, Repo
from git.exc import InvalidGitRepositoryError


@dataclass
class CloneStatus:
    """Status of a repository clone operation."""

    name: str
    url: str
    path: str
    success: bool
    message: str
    cloned_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "path": self.path,
            "success": self.success,
            "message": self.message,
            "cloned_at": self.cloned_at,
            "updated_at": self.updated_at,
        }


@dataclass
class CloneManifest:
    """Manifest tracking all cloned repositories."""

    repos: dict[str, CloneStatus] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        data = {"repos": {name: status.to_dict() for name, status in self.repos.items()}}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CloneManifest":
        """Load manifest from JSON file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        manifest = cls()
        for name, status_dict in data.get("repos", {}).items():
            manifest.repos[name] = CloneStatus(**status_dict)
        return manifest


def clone_repo(
    name: str, url: str, repos_dir: Path, update_if_exists: bool = True
) -> CloneStatus:
    """Clone a repository or update if it exists.

    Args:
        name: Short name for the repository
        url: Git URL to clone from
        repos_dir: Directory to clone into
        update_if_exists: If True, pull updates for existing repos

    Returns:
        CloneStatus with result of operation
    """
    repo_path = repos_dir / name
    now = datetime.now().isoformat()

    # Check if repo already exists
    if repo_path.exists():
        if not update_if_exists:
            return CloneStatus(
                name=name,
                url=url,
                path=str(repo_path),
                success=True,
                message="Already exists (skipped update)",
            )

        # Try to update existing repo
        try:
            repo = Repo(repo_path)
            origin = repo.remotes.origin
            origin.pull()
            return CloneStatus(
                name=name,
                url=url,
                path=str(repo_path),
                success=True,
                message="Updated successfully",
                updated_at=now,
            )
        except InvalidGitRepositoryError:
            return CloneStatus(
                name=name,
                url=url,
                path=str(repo_path),
                success=False,
                message="Directory exists but is not a git repository",
            )
        except GitCommandError as e:
            return CloneStatus(
                name=name,
                url=url,
                path=str(repo_path),
                success=False,
                message=f"Pull failed: {e}",
            )

    # Clone new repo
    try:
        repos_dir.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(url, repo_path, depth=1)
        return CloneStatus(
            name=name,
            url=url,
            path=str(repo_path),
            success=True,
            message="Cloned successfully",
            cloned_at=now,
        )
    except GitCommandError as e:
        return CloneStatus(
            name=name,
            url=url,
            path=str(repo_path),
            success=False,
            message=f"Clone failed: {e}",
        )


def clone_repos(
    repos: list[tuple[str, str]], repos_dir: Path, update_if_exists: bool = True
) -> list[CloneStatus]:
    """Clone multiple repositories.

    Args:
        repos: List of (name, url) tuples
        repos_dir: Directory to clone into
        update_if_exists: If True, pull updates for existing repos

    Returns:
        List of CloneStatus for each repo
    """
    results = []
    for name, url in repos:
        status = clone_repo(name, url, repos_dir, update_if_exists)
        results.append(status)
    return results


def list_cloned_repos(repos_dir: Path) -> list[str]:
    """List names of all cloned repositories."""
    if not repos_dir.exists():
        return []

    repos = []
    for item in repos_dir.iterdir():
        if item.is_dir() and (item / ".git").exists():
            repos.append(item.name)
    return sorted(repos)


def get_repo_info(repos_dir: Path, name: str) -> dict | None:
    """Get information about a cloned repository."""
    repo_path = repos_dir / name
    if not repo_path.exists():
        return None

    try:
        repo = Repo(repo_path)
        return {
            "name": name,
            "path": str(repo_path),
            "head": str(repo.head.commit)[:8],
            "branch": repo.active_branch.name if not repo.head.is_detached else "detached",
            "remote": repo.remotes.origin.url if repo.remotes else None,
        }
    except (InvalidGitRepositoryError, TypeError):
        return {"name": name, "path": str(repo_path), "error": "Invalid git repository"}
