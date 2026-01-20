# 9ml - Machine Learning Models for Plan 9

A port of [llama2.c](https://github.com/karpathy/llama2.c) to Plan 9 (9front). Run LLaMA inference natively on Plan 9.

## Features

- **Multi-format support** - Load safetensors (HuggingFace) and GGUF (llama.cpp) models
- **Quantized inference** - Run Q4_0 and Q8_0 quantized GGUF models
- **SIMD acceleration** - SSE2 assembly for 5.7x matmul speedup
- **Multi-threading** - Parallel attention heads via Plan 9 libthread
- **9P file server** (`llmfs`) - Multi-model inference over the network with LRU eviction
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
# Install huggingface-cli
pip install huggingface-hub

# Download safetensors model (60MB)
huggingface-cli download Xenova/llama2.c-stories15M model.safetensors \
    --local-dir models/

# Download pre-quantized GGUF model (17MB, Q8_0)
huggingface-cli download tensorblock/Xenova_llama2.c-stories15M-GGUF \
    llama2.c-stories15M-Q8_0.gguf --local-dir models/

# Download tokenizer
wget -O tokenizer.bin \
    https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

## Model Formats

9ml supports two model formats with automatic detection:

| Format | Source | Quantization | Use Case |
|--------|--------|--------------|----------|
| Safetensors | HuggingFace | FP32/FP16 | Full precision inference |
| GGUF | llama.cpp | Q4_0, Q8_0, etc. | Memory-efficient quantized inference |

The loader automatically detects format from file magic:
- **GGUF**: Magic `0x46554747` ("GGUF")
- **Safetensors**: JSON header with tensor metadata

For safetensors models, place a `config.json` in the same directory to provide model configuration (dim, n_heads, rope_theta, etc.).

## Running Inference

### Safetensors Model

In Plan 9:
```rc
./run model.safetensors -z tokenizer.bin -n 50 -i 'Once upon a time'
```

### GGUF Model (Quantized)

```rc
./run model-Q8_0.gguf -z tokenizer.bin -n 50 -i 'Once upon a time'
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
| `--no-simd` | Disable SIMD acceleration |
| `-j <int>` | Number of threads |

## LLM File Server (llmfs)

A 9P file server for multi-model inference with LRU eviction.

### File System Structure

```
/mnt/llm/
    ctl             # Server control: load, unload, limit
    info            # Pool status: loaded models, memory usage
    clone           # Create new session (returns ID)
    0/              # Session directory
        ctl         # Session control: model, temp, generate
        info        # Session info and status
        prompt      # Write prompt text
        output      # Read generated output (blocks until done)
        stream      # Streaming output
```

### Usage

```rc
# Start server and mount
./llmfs -s llm
mount /srv/llm /mnt/llm

# Load a model
echo 'load small stories15M.safetensors tokenizer.bin' > /mnt/llm/ctl

# Create session and generate
session=`{cat /mnt/llm/clone}
echo 'model small' > /mnt/llm/$session/ctl
echo 'Once upon a time' > /mnt/llm/$session/prompt
echo 'generate' > /mnt/llm/$session/ctl
cat /mnt/llm/$session/output
```

### Remote Inference

On server:
```rc
./llmfs -s llm
echo 'load small model.safetensors tokenizer.bin' > /srv/llm/ctl
aux/listen1 -tv tcp!*!564 /bin/exportfs -r /srv/llm
```

On client:
```rc
srv tcp!server!564 llm
mount /srv/llm /mnt/llm
# Use as if local
```

## Performance

### SIMD Vectorization (SSE2)

| Operation | Implementation | Speedup |
|-----------|---------------|---------|
| matmul | SSE2 assembly | ~5.7x |
| rmsnorm | SSE2 assembly | ~3x |
| dot_product | SSE2 assembly | ~4x |
| vec_add, vec_scale | SSE2 assembly | ~4x |
| softmax | C with unrolling | ~2x |

### Benchmark Results

Tested on stories15M model:

| Mode | GFLOPS | Speedup |
|------|--------|---------|
| Scalar (1 thread) | 3.4 | 1.0x |
| SIMD (1 thread) | 19.0 | 5.7x |
| SIMD (4 threads) | 18.7 | 5.6x |

Token generation: ~180-240 tok/s on stories15M.

## Project Structure

```
9ml/
├── src/
│   ├── run.c              # Main inference driver
│   ├── model.c            # Model loading (format detection)
│   ├── llmfs.c            # 9P file server
│   ├── simd_amd64.s       # SSE2 assembly
│   ├── parallel.c         # Thread pool
│   ├── arch/              # Architecture plugins
│   │   ├── arch.c         # Plugin registry
│   │   └── llama2.c       # LLaMA 2 architecture
│   ├── format/            # File format parsers
│   │   ├── gguf.c         # GGUF parser + dequantization
│   │   └── safetensors.c  # Safetensors parser + config.json
│   ├── pool/              # Model pool management
│   │   └── pool.c         # LRU eviction, reference counting
│   └── tests/             # Plan 9 test source files
├── test/                  # C test harness (Linux host)
│   ├── harness.c          # Main test driver
│   ├── reference.c        # Reference implementations
│   └── qemu.c             # QEMU VM management
├── models/                # Model files (download separately)
├── Makefile               # Linux build file
├── mkfile                 # Plan 9 build file
└── CLAUDE.md              # Development documentation
```

## Building

### Linux (Test Harness)

```bash
make test      # Build and run all tests in Plan 9 VM
make clean     # Clean build artifacts
```

### Plan 9 Native

```rc
mk            # Build all targets
mk clean      # Clean build artifacts
```

### Manual Compilation (Plan 9)

```rc
# Compile
6c -w run.c

# Link with libraries
6l -o run run.6 simd_amd64.6 arch/arch.a6 format/format.a6
```

## Test Coverage

| Test | Description |
|------|-------------|
| rmsnorm, softmax, matmul | Core math operations |
| simd_validation | SIMD vs scalar correctness |
| generation | End-to-end text generation |
| format_generation | Safetensors vs GGUF output match |
| config_json | HuggingFace config.json parsing |
| gguf_dequant | Q4_0/Q8_0 dequantization |
| pool_lru | Model pool LRU eviction |
| llmfs_local | 9P server local mount |
| llmfs_remote | Dual-VM remote inference |

## Resources

- [9front](https://9front.org) - Plan 9 fork
- [Plan 9 C Programming](http://doc.cat-v.org/plan_9/programming/c_programming_in_plan_9)
- [llama2.c](https://github.com/karpathy/llama2.c) - Original implementation
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - GGUF specification

## License

MIT License - Same as llama2.c
