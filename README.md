# 9ml - Machine Learning Models for Plan 9

A port of [llama2.c](https://github.com/karpathy/llama2.c) to Plan 9 (9front). Run Llama 2 inference natively on Plan 9.

## Features

- **FP32 inference** (`run.c`) - Full precision transformer inference
- **INT8 quantized inference** (`runq.c`) - 4x smaller models with quantization
- **Pure Plan 9 C** - No external dependencies, uses only Plan 9 libc
- **Automated testing** - Test suite comparing Plan 9 output against reference

## Prerequisites

### macOS

```bash
# Install QEMU via MacPorts
sudo port install qemu

# Or via Homebrew
brew install qemu
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install qemu-system-x86 qemu-utils dosfstools expect
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install qemu-system-x86 qemu-img dosfstools expect
```

## Setup

### 1. Download 9front

Download the 9front QEMU image from [9front.org](http://9front.org/iso/):

```bash
# Download the latest 9front ISO
curl -O http://9front.org/iso/9front-10522.amd64.iso.gz
gunzip 9front-10522.amd64.iso.gz

# Create a QEMU disk image and install 9front
qemu-img create -f qcow2 qemu/9front.qcow2 4G
qemu-system-x86_64 -m 512 -cpu max \
    -drive file=qemu/9front.qcow2,format=qcow2 \
    -cdrom 9front-10522.amd64.iso \
    -boot d
```

Follow the 9front installation prompts, then shut down and remove the `-cdrom` and `-boot` flags.

Alternatively, use a pre-built 9front QCOW2 image if available.

### 2. Create Shared Disk

```bash
./qemu/create-shared.sh
```

This creates a 128MB FAT32 disk image for sharing files between host and Plan 9.

### 3. Download Model Weights

Download the TinyStories models from Hugging Face:

```bash
# Small model (60MB) - good for testing
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

# Medium model (168MB)
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin

# Large model (440MB)
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin

# Tokenizer (required)
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

## Running Inference

### Quick Start

```bash
# Copy source and model to shared disk
./qemu/copy-to-shared.sh src/run.c stories15M.bin tokenizer.bin

# Compile and run in Plan 9
./qemu/run-cmd.sh "cd /mnt/host && 6c -w run.c && 6l -o run run.6 && ./run stories15M.bin -z tokenizer.bin -n 50"
```

### FP32 Inference

```bash
./qemu/run-cmd.sh "./run /mnt/host/stories15M.bin -z /mnt/host/tokenizer.bin -n 100 -i 'Once upon a time'"
```

### INT8 Quantized Inference

First, export a quantized model (requires Python):

```bash
python export.py stories15M_q80.bin --version 2 --checkpoint stories15M.pt
```

Then run:

```bash
./qemu/copy-to-shared.sh src/runq.c stories15M_q80.bin tokenizer.bin
./qemu/run-cmd.sh "cd /mnt/host && 6c -w runq.c && 6l -o runq runq.6 && ./runq stories15M_q80.bin -z tokenizer.bin -n 50"
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-t <float>` | Temperature (0.0 = greedy, 1.0 = default) |
| `-p <float>` | Top-p sampling (0.9 = default) |
| `-s <int>` | Random seed |
| `-n <int>` | Number of tokens to generate |
| `-i <string>` | Input prompt |
| `-z <string>` | Path to tokenizer |
| `-m generate\|chat` | Mode (default: generate) |

## Running Tests

```bash
python3 tests/run_tests.py
```

The test suite runs component tests comparing Plan 9 output against Python reference implementations.

## Project Structure

```
9ml/
├── src/
│   ├── run.c          # FP32 inference
│   └── runq.c         # INT8 quantized inference
├── tests/
│   └── run_tests.py   # Automated test suite
├── qemu/
│   ├── 9front.qcow2   # 9front disk image (not included)
│   ├── shared.img     # FAT32 shared disk (created by script)
│   ├── run-cmd.sh     # Execute commands in Plan 9
│   ├── copy-to-shared.sh    # Copy files to shared disk
│   ├── create-shared.sh     # Create shared disk image
│   └── os-helper.sh   # Cross-platform helper functions
└── CLAUDE.md          # Development documentation
```

## Plan 9 Notes

### Compilation

9front uses amd64 architecture:
- Compiler: `6c` (NOT `8c` which is for 386)
- Linker: `6l` (NOT `8l`)
- Object files: `.6` extension

```bash
# Compile
6c -w program.c

# Link
6l -o program program.6
```

### Key Differences from POSIX

| POSIX | Plan 9 |
|-------|--------|
| `printf(...)` | `print(...)` |
| `fprintf(stderr, ...)` | `fprint(2, ...)` |
| `exit(0)` | `exits(0)` |
| `exit(1)` | `exits("error")` |
| `NULL` | `nil` |
| `mmap()` | `malloc()` + `read()` |

## Resources

- [9front](https://9front.org) - Plan 9 fork
- [Plan 9 C Programming](http://doc.cat-v.org/plan_9/programming/c_programming_in_plan_9)
- [llama2.c](https://github.com/karpathy/llama2.c) - Original implementation

## License

MIT License - Same as llama2.c
