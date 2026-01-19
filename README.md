# 9ml - Machine Learning Models for Plan 9

A port of [llama2.c](https://github.com/karpathy/llama2.c) to Plan 9 (9front). Run Llama 2 inference natively on Plan 9.

## Features

- **FP32 inference** (`run.c`) - Full precision transformer inference
- **INT8 quantized inference** (`runq.c`) - 4x smaller models with quantization
- **SIMD acceleration** - SSE2 assembly for 5.7x matmul speedup
- **Multi-threading** - Parallel attention heads via Plan 9 libthread
- **9P file server** (`llmfs`) - Distributed inference over the network
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

## LLM File Server (llmfs)

llmfs is a 9P file server that exposes LLM inference as a Plan 9 filesystem. This enables distributed inference where a powerful server runs the model and thin clients access it over the network.

### File System Structure

```
/mnt/llm/
    ctl             # RW: load/unload model, server status
    model           # R:  model info (dim, layers, vocab, memory)
    clone           # R:  read to create new session, returns ID
    0/              # Session 0 directory
        ctl         # RW: temp, topp, seed, steps, generate, reset
        prompt      # W:  write prompt text
        output      # R:  complete output (blocks until done)
        stream      # R:  streaming output (returns tokens as generated)
        status      # R:  idle|generating N/M|done tok/s|error msg
    1/              # Session 1...
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

# Load a model
echo 'load stories15M.bin tokenizer.bin' > /mnt/llm/ctl

# Check model info
cat /mnt/llm/model

# Create a session
session=`{cat /mnt/llm/clone}

# Configure session
echo 'temp 0.0' > /mnt/llm/$session/ctl
echo 'steps 50' > /mnt/llm/$session/ctl

# Set prompt and generate
echo 'Once upon a time' > /mnt/llm/$session/prompt
echo 'generate' > /mnt/llm/$session/ctl

# Read output (blocks until complete)
cat /mnt/llm/$session/output

# Check status
cat /mnt/llm/$session/status
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
echo 'load /path/to/model.bin /path/to/tokenizer.bin' > /mnt/llm/ctl
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
| `load <model> <tokenizer>` | Load model and tokenizer |
| `unload` | Unload current model |

## Performance Optimizations

### SIMD Vectorization (SSE2)

The inference engine uses hand-written SSE2 assembly for critical operations:

| Operation | Implementation | Speedup |
|-----------|---------------|---------|
| matmul | SSE2 assembly | ~5.7x |
| rmsnorm | SSE2 assembly | ~3x |
| dot_product | SSE2 assembly | ~4x |
| vec_add, vec_scale | SSE2 assembly | ~4x |
| softmax | C with unrolling | ~2x |

The SIMD implementation uses:
- 8-element unrolled loops with 2 accumulators
- 4-element cleanup loop for remainders
- Scalar fallback for non-aligned tails
- Horizontal sum via SHUFPS for reductions

### Thread Pool

Parallel execution uses Plan 9's libthread:
- Auto-detects CPU count from `/dev/sysstat`
- Channel-based work distribution
- Parallel attention head computation

### Benchmark Results

Tested on stories15M model with 1024x1024 matmul:

| Mode | GFLOPS | Speedup |
|------|--------|---------|
| Scalar (1 thread) | 3.4 | 1.0x |
| SIMD (1 thread) | 19.0 | 5.7x |
| SIMD (4 threads) | 18.7 | 5.6x |

Token generation throughput: ~180-200 tok/s on stories15M.

Note: Multi-threading overhead can exceed benefit for small matrices.

### Runtime Configuration

Command-line flags:
```rc
./run model.bin -z tok.bin --no-simd     # Disable SIMD
./run model.bin -z tok.bin --threads 2   # Set thread count
./run model.bin -z tok.bin -j 4          # Short form for threads
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
│   ├── llmfs.c            # 9P file server for remote inference
│   ├── simd.h             # SIMD function declarations
│   ├── simd_amd64.s       # SSE2 assembly (matmul, rmsnorm, etc.)
│   ├── parallel.h         # Thread pool declarations
│   ├── parallel.c         # Thread pool implementation
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
