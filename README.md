# 9ml - Machine Learning Models for Plan 9

A port of [llama2.c](https://github.com/karpathy/llama2.c) to Plan 9 (9front). Run Llama 2 inference natively on Plan 9.

## Features

- **FP32 inference** (`run.c`) - Full precision transformer inference
- **INT8 quantized inference** (`runq.c`) - 4x smaller models with quantization
- **Pure Plan 9 C** - No external dependencies, uses only Plan 9 libc
- **Pure C test suite** - No Python or shell scripts required
- **Cross-platform** - Build and test on Linux, run natively on Plan 9

## Prerequisites

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install qemu-system-x86 qemu-utils mtools curl gcc
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install qemu-system-x86 qemu-img mtools curl gcc
```

### macOS

```bash
# Install via Homebrew
brew install qemu mtools curl

# Or via MacPorts
sudo port install qemu mtools curl
```

## Quick Start

### Build and Test

```bash
# Build everything and run tests
make test
```

The test harness automatically:
1. Downloads 9front VM image if needed
2. Creates a FAT shared disk
3. Boots Plan 9 in QEMU
4. Compiles and runs all tests inside Plan 9
5. Compares output against C reference implementations

### Download Model Weights

```bash
# Small model (60MB) - good for testing
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

# Tokenizer (required)
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

## Running Inference

### FP32 Inference

In Plan 9:
```rc
6c -w run.c && 6l -o run run.6
./run stories15M.bin -z tokenizer.bin -n 50 -i 'Once upon a time'
```

### INT8 Quantized Inference

First, quantize the model (on Linux):
```bash
make export
./export quantize stories15M.bin stories15M_q80.bin
```

Then run in Plan 9:
```rc
6c -w runq.c && 6l -o runq runq.6
./runq stories15M_q80.bin -z tokenizer.bin -n 50
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

## Model Export Tool

```bash
# Build the export tool
make export

# Show model info
./export info stories15M.bin

# Quantize FP32 model to Q8_0
./export quantize stories15M.bin stories15M_q80.bin
```

## Project Structure

```
9ml/
├── src/
│   ├── run.c              # FP32 inference
│   ├── runq.c             # INT8 quantized inference
│   ├── model.c            # Model loading helpers
│   ├── modelq.c           # Quantized model helpers
│   ├── export.c           # Model export/conversion tool
│   ├── mkfile             # Plan 9 build file
│   └── tests/             # Plan 9 test source files
├── test/                  # C test harness (Linux host)
│   ├── harness.c          # Main test driver
│   ├── reference.c/h      # Reference implementations
│   ├── qemu.c/h           # QEMU VM management
│   └── fat.c/h            # FAT disk operations
├── qemu/
│   ├── 9front.qcow2       # VM disk image (auto-downloaded)
│   └── shared.img         # FAT disk for file sharing
├── Makefile               # Linux build file
├── mkfile                 # Root Plan 9 build file
└── CLAUDE.md              # Development documentation
```

## Building in Plan 9

Clone the repo and build natively:
```rc
mk            # Build all: run, runq, export, tests
mk clean      # Clean all build artifacts
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
