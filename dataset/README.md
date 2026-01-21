# Plan 9 Dataset Preparation Tooling

Tools for building fine-tuning datasets that teach LLMs to work with Plan 9 OS:
rc shell, C programming, assembly, rio, 9P protocol, and namespaces.

## Installation

```bash
cd dataset
pip install -e .
```

## Quick Start

```bash
# Fetch list of Plan 9 repositories from awesome-plan9
plan9-dataset fetch-repos

# Clone specific repositories
plan9-dataset clone-repos git9 plan9port

# Clone all repositories (this may take a while)
plan9-dataset clone-repos --all

# Index all cloned files
plan9-dataset index-files

# Show statistics
plan9-dataset stats

# Search indexed files
plan9-dataset search "mkfile"
plan9-dataset search --type rc
plan9-dataset search --repo git9

# Export training data
plan9-dataset export-data --format all
plan9-dataset export-data --format jsonl
plan9-dataset export-data --format alpaca
```

## Directory Structure

```
dataset/
├── plan9_dataset/        # Python package
│   ├── __init__.py
│   ├── cli.py            # Click CLI commands
│   ├── fetch.py          # Fetch repos from awesome-plan9
│   ├── clone.py          # Clone repositories
│   ├── index.py          # Index files by type
│   └── export.py         # Export to training formats
├── repos/                # Cloned repositories (gitignored)
├── data/
│   ├── raw/              # Curated examples
│   │   └── examples.json
│   └── processed/        # Exported training files
│       ├── train.jsonl
│       ├── train_alpaca.json
│       └── train_sharegpt.json
├── pyproject.toml
└── README.md
```

## Curation Workflow

1. **Fetch repo list**: `plan9-dataset fetch-repos`
2. **Clone repos**: `plan9-dataset clone-repos git9`
3. **Index files**: `plan9-dataset index-files`
4. **Browse files**: Use Claude Code to read files from `repos/`
5. **Add examples**: Edit `data/raw/examples.json`
6. **Export**: `plan9-dataset export-data --format all`

## Example Format

The `data/raw/examples.json` file contains curated instruction-response pairs:

```json
[
  {
    "instruction": "Write an rc script to clone a git repository",
    "response": "#!/bin/rc\nrfork en\n\nif(~ $#* 0){\n    echo usage: clone url >[1=2]\n    exit usage\n}\n\ngit/clone $1",
    "source": "git9/clone",
    "category": "rc-basics"
  }
]
```

## Export Formats

### JSONL (`train.jsonl`)
```json
{"instruction": "...", "response": "..."}
{"instruction": "...", "response": "..."}
```

### Alpaca (`train_alpaca.json`)
```json
[
  {"instruction": "...", "input": "", "output": "..."},
  {"instruction": "...", "input": "", "output": "..."}
]
```

### ShareGPT (`train_sharegpt.json`)
```json
[
  {
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
    ]
  }
]
```

## Example Categories

| Category | Description | Example Sources |
|----------|-------------|-----------------|
| rc-basics | Basic rc shell syntax | git9 scripts |
| rc-flags | Argument parsing | aux/getflags usage |
| rc-functions | Function definitions | fn name{} patterns |
| c-headers | Plan 9 C includes | <u.h>, <libc.h> |
| c-9p | 9P file servers | git9/fs.c |
| c-errors | Error handling | String returns |
| asm-simd | SIMD assembly | 9ml simd_amd64.s |
| sys-mount | Mounting filesystems | mount commands |
| sys-namespace | Namespace manipulation | rfork patterns |

## File Types

The indexer recognizes these file types:

| Type | Extensions/Patterns |
|------|---------------------|
| c | `.c` |
| header | `.h` |
| rc | `.rc`, shebang `#!/bin/rc` |
| asm | `.s`, `.S` |
| man | `.1` - `.9` |
| mkfile | `mkfile` |
| doc | `.txt`, `.md`, `.rst` |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black plan9_dataset/
ruff check plan9_dataset/
```
