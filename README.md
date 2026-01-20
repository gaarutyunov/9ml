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

llmfs is a 9P file server that exposes LLM inference as a Plan 9 filesystem. It supports multiple models with LRU eviction and enables distributed inference where a powerful server runs models and thin clients access them over the network.

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

### Building llmfs

In Plan 9:
```rc
cd /usr/glenda/9ml/src
mk llmfs
```

To install globally:
```rc
cp llmfs /amd64/bin/
```

### Local Usage

```rc
# Start the file server
llmfs -s llm

# Mount it
mount /srv/llm /mnt/llm

# Load a model (name, model file, tokenizer)
echo 'load small stories15M.safetensors tokenizer.bin' > /mnt/llm/ctl

# Check pool info
cat /mnt/llm/info

# Create a session
session=`{cat /mnt/llm/clone}

# Bind model to session and configure
echo 'model small' > /mnt/llm/$session/ctl
echo 'temp 0.0' > /mnt/llm/$session/ctl
echo 'steps 50' > /mnt/llm/$session/ctl

# Set prompt and generate
echo 'Once upon a time' > /mnt/llm/$session/prompt
echo 'generate' > /mnt/llm/$session/ctl

# Read output (blocks until complete)
cat /mnt/llm/$session/output

# Check session info
cat /mnt/llm/$session/info
```

### Remote Usage with Drawterm

This setup allows you to run llmfs on a remote server and access it from your local machine using drawterm.

#### Server Setup (9front in QEMU)

1. **Start QEMU with port forwarding:**
```bash
qemu-system-x86_64 -m 1024 -cpu max -enable-kvm \
  -drive file=9front.qcow2,format=qcow2,if=virtio \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp::9564-:564,hostfwd=tcp::17019-:17019 \
  -vnc :1
```

2. **Boot 9front and configure network:**
```rc
# Accept defaults at boot prompts (press Enter twice)
# Configure network (QEMU user networking)
ip/ipconfig -6
```

3. **Set up authentication (factotum):**
```rc
# Add authentication key
echo 'key proto=dp9ik dom=llmfs user=glenda !password=yourpassword' > /mnt/factotum/ctl
```

4. **Start CPU listener:**
```rc
# Create service script
mkdir -p /rc/bin/service
cat > /rc/bin/service/tcp17019 <<'EOF'
#!/bin/rc
exec /bin/cpu -R
EOF
chmod +x /rc/bin/service/tcp17019

# Start listener
aux/listen1 -t tcp!*!17019 /rc/bin/service/tcp17019
```

5. **Start llmfs and load model:**
```rc
llmfs -s llm
mount /srv/llm /mnt/llm
echo 'load small /path/to/model.safetensors /path/to/tokenizer.bin' > /mnt/llm/ctl
```

#### Client Setup (macOS/Linux)

1. **Install drawterm:**
```bash
# macOS / Linux (build from source)
git clone https://github.com/9front/drawterm
cd drawterm
# macOS
CONF=osx-cocoa make
# Linux
make
```

2. **Connect to remote server:**
```bash
drawterm -h <server-ip>:17019 -a <server-ip>:17019 -u glenda
```

When prompted, enter the password you set in factotum.

3. **Use llmfs from drawterm:**
```rc
# Mount llmfs (already running on server)
mount /srv/llm /mnt/llm

# Create session and generate text
session=`{cat /mnt/llm/clone}
echo 'model small' > /mnt/llm/$session/ctl
echo 'Once upon a time' > /mnt/llm/$session/prompt
echo generate > /mnt/llm/$session/ctl
cat /mnt/llm/$session/output
```

#### Drawterm Tips

- **Resize text:** Right-click on window, select "Resize" or use keyboard shortcuts
- **Previous command:** Use up arrow or Ctrl-P
- **Copy/paste:** Select text with mouse to copy, middle-click to paste
- **Exit:** Type `exit` or close the window

### Session Control Commands

| Command | Description |
|---------|-------------|
| `model <name>` | Bind session to named model |
| `temp <float>` | Set temperature (0.0 = greedy) |
| `topp <float>` | Set top-p sampling (0.0-1.0) |
| `seed <int>` | Set random seed |
| `steps <int>` | Set max tokens to generate |
| `generate` | Start generation |
| `reset` | Reset session state |
| `close` | Close session |

### Server Control Commands

| Command | Description |
|---------|-------------|
| `load <name> <model> <tokenizer>` | Load model into pool with given name |
| `unload <name>` | Unload model from pool |
| `limit <max_models> <max_memory>` | Set pool limits |

## Performance

### SIMD Vectorization (SSE2)

| Operation | Implementation | Speedup |
|-----------|---------------|---------|
| matmul | SSE2 assembly | ~5.7x |
| rmsnorm | SSE2 assembly | ~3x |
| dot_product | SSE2 assembly | ~4x |
| vec_add, vec_scale | SSE2 assembly | ~4x |
| softmax | SSE2 assembly | ~60x |

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
