"""Fetch repository list from awesome-plan9."""

import re
from dataclasses import dataclass

import requests


AWESOME_PLAN9_URL = "https://raw.githubusercontent.com/henesy/awesome-plan9/master/README.md"


@dataclass
class RepoInfo:
    """Information about a repository."""

    name: str
    url: str
    category: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "category": self.category,
            "description": self.description,
        }


def fetch_awesome_plan9() -> str:
    """Fetch the awesome-plan9 README content."""
    response = requests.get(AWESOME_PLAN9_URL, timeout=30)
    response.raise_for_status()
    return response.text


def parse_repos(readme_content: str) -> list[RepoInfo]:
    """Parse the README to extract repository information.

    The awesome-plan9 README uses markdown format with:
    - ## Category headers
    - * [name](url) - description format for entries
    """
    repos = []
    current_category = "uncategorized"

    # Pattern for markdown links: [text](url)
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    # Pattern for category headers
    header_pattern = re.compile(r"^##\s+(.+)$")

    for line in readme_content.split("\n"):
        line = line.strip()

        # Check for category header
        header_match = header_pattern.match(line)
        if header_match:
            current_category = header_match.group(1).strip().lower()
            # Normalize category names
            current_category = current_category.replace(" ", "-")
            continue

        # Check for list items with links
        if not line.startswith(("*", "-")):
            continue

        # Find all links in the line
        matches = link_pattern.findall(line)
        if not matches:
            continue

        for name, url in matches:
            # Skip non-repo links (anchors, non-git URLs)
            if url.startswith("#"):
                continue

            # Filter for git-hostable URLs
            if not _is_git_url(url):
                continue

            # Extract description (text after the link)
            desc_match = re.search(r"\]\([^)]+\)\s*[-–—:]\s*(.+)$", line)
            description = desc_match.group(1).strip() if desc_match else ""

            # Clean up the name
            name = name.strip()

            # Derive a short name from URL if name is generic
            if name.lower() in ("here", "link", "repo", "repository", "source"):
                name = _name_from_url(url)

            repos.append(
                RepoInfo(
                    name=name,
                    url=url,
                    category=current_category,
                    description=description,
                )
            )

    return repos


def _is_git_url(url: str) -> bool:
    """Check if URL is likely a git repository."""
    git_hosts = [
        "github.com",
        "gitlab.com",
        "bitbucket.org",
        "git.sr.ht",
        "codeberg.org",
        "9legacy.org",
        "git.9front.org",
        "shithub.us",
        "code.9front.org",
        "git.",
    ]
    url_lower = url.lower()
    return any(host in url_lower for host in git_hosts)


def _name_from_url(url: str) -> str:
    """Extract a repository name from URL."""
    # Remove trailing .git
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    # Get last path component
    parts = url.split("/")
    return parts[-1] if parts else "unknown"


def get_repos() -> list[RepoInfo]:
    """Fetch and parse the awesome-plan9 repository list."""
    readme = fetch_awesome_plan9()
    return parse_repos(readme)


def get_repos_by_category(repos: list[RepoInfo]) -> dict[str, list[RepoInfo]]:
    """Group repositories by category."""
    by_category: dict[str, list[RepoInfo]] = {}
    for repo in repos:
        if repo.category not in by_category:
            by_category[repo.category] = []
        by_category[repo.category].append(repo)
    return by_category
