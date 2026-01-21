# Plan 9 Dataset Card Template

This is a template for the HuggingFace dataset card. The actual card is auto-generated
by `plan9-dataset publish-hf` using statistics from the dataset.

## YAML Frontmatter

```yaml
---
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
```

## Required Sections

### Quick Start

Show how to load both dataset configurations:

```python
from datasets import load_dataset

# Simple instruction-response format
ds = load_dataset("USER/plan9-sft", "sft")

# Function calling format
ds = load_dataset("USER/plan9-sft", "function_calling")
```

### Dataset Statistics

Auto-filled from examples.json:

| Metric | Value |
|--------|-------|
| Total SFT Examples | {{total}} |
| Function Calling Examples | {{fc_count}} |
| GRPO Tasks | {{grpo_count}} |
| Avg Instruction Length | {{avg_instruction_length}} chars |
| Avg Response Length | {{avg_response_length}} chars |

### Categories Table

| Category | Count |
|----------|-------|
| c-basics | N |
| rc-basics | N |
| c-io | N |
| ... | ... |

### Plan 9 Conventions Reference

This table helps users understand Plan 9 idioms:

| POSIX | Plan 9 |
|-------|--------|
| `printf()` | `print()` |
| `fprintf(stderr, ...)` | `fprint(2, ...)` |
| `exit(0)` | `exits(nil)` |
| `exit(1)` | `exits("error")` |
| `NULL` | `nil` |
| `int8_t` | `schar` |
| `uint8_t` | `uchar` |
| `int32_t` | `int` |
| `uint32_t` | `uint` |
| `int64_t` | `vlong` |
| `uint64_t` | `uvlong` |
| `size_t` | `ulong` |
| `clock_gettime()` | `nsec()` |
| `mmap()` | `open() + read() + malloc()` |

### Headers

```c
#include <u.h>      // Basic types, must come first
#include <libc.h>   // Standard library (print, exits, malloc)
#include <bio.h>    // Buffered I/O
#include <thread.h> // Threads/processes
```

### Compilation

```rc
# Compile C program (amd64)
6c program.c && 6l -o program program.6

# Run rc script
chmod +x script.rc && ./script.rc
```

## Example Formats

### SFT Format

```json
{
  "instruction": "Write a Plan 9 C program that prints 'Hello'",
  "response": "#include <u.h>\n#include <libc.h>\n\nvoid\nmain(int, char**)\n{\n\tprint(\"Hello\\n\");\n\texits(nil);\n}\n",
  "category": "c-basics",
  "source": "manual"
}
```

### Function Calling Format

```
<start_of_turn>system
You are a Plan 9 programming assistant...
[tool definitions]
<end_of_turn>
<start_of_turn>user
Write a Plan 9 C program that prints 'Hello'
<end_of_turn>
<start_of_turn>model
<think>
I need to write a Plan 9 C program. I'll use print() instead of printf().
</think>
<start_function_call>call:write_file{"path": "hello.c", "content": "..."}<end_function_call>
<end_of_turn>
<start_of_turn>user
<start_function_response>response:write_file{"success": true}<end_function_response>
<end_of_turn>
...
```

## Training Approaches

### 1. Standard SFT

Use the `sft` configuration with instruction-tuning frameworks:
- Axolotl (alpaca/sharegpt format)
- TRL SFTTrainer
- Unsloth

### 2. Function Calling SFT

Use the `function_calling` configuration to teach the model:
- When to use tools
- How to format tool calls
- How to interpret tool responses
- Plan 9 compilation workflow (write → compile → run)

### 3. GRPO with Execution Rewards

Use GRPO tasks with a running Plan 9 VM for execution-based rewards:

```python
from plan9_dataset.rewards import create_reward_function

reward_fn = create_reward_function(
    disk_image="/path/to/9front.qcow2",
    shared_image="/path/to/shared.img",
)

# In GRPO trainer
trainer = GRPOTrainer(
    reward_funcs=[reward_fn],
    ...
)
```

### 4. Remote GRPO (Colab-friendly)

For environments without local QEMU:

```python
from plan9_dataset.qemu_client import RemoteQEMUClient

client = RemoteQEMUClient(
    server_url="https://your-server.com",
    token="your-token",
)

# Use as reward function
def remote_reward(outputs):
    return [client.compute_reward(o) for o in outputs]
```

## Citation

```bibtex
@misc{plan9-dataset,
  title={Plan 9 Programming Dataset},
  author={9ml Contributors},
  year={2025},
  url={https://huggingface.co/datasets/USER/plan9-sft}
}
```
