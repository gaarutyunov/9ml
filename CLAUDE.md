# 9ml - Machine Learning Models for Plan 9

Port of [llama2.c](https://github.com/karpathy/llama2.c) to Plan 9 (9front).

## Rules

1. **NEVER skip tests.** All tests are mandatory. If a test cannot run, fix the environment - do not skip.
2. **All changes must be tested.** Run tests before committing:
   - **Linux host:** `make test` (or `cd test && gcc -o harness *.c -lm && ./harness`)
   - **Plan 9 native:** `mk` (compiles all targets)
3. **Tests must pass.** Do not merge if tests fail.

---

## Quick Start

### Running Tests (Linux Host)

```bash
# Build and run all tests
make test

# Or manually:
cd test
gcc -o harness *.c -lm
./harness
```

The test harness:
1. Starts a Plan 9 QEMU VM
2. Compiles and runs tests inside Plan 9
3. Compares output against C reference implementations
4. Reports pass/fail

### Building in Plan 9

```bash
# Clone repo and build natively in Plan 9
mk            # Build all: run, runq, export, tests
mk clean      # Clean all build artifacts
```

### Model Tools

```bash
# Show model info
./export info stories15M.bin

# Quantize a model (FP32 -> Q8_0)
./export quantize stories15M.bin stories15M_q80.bin
```

---

## Project Structure

```
9ml/
├── src/
│   ├── run.c              # FP32 inference (ported)
│   ├── runq.c             # INT8 quantized inference (ported)
│   ├── model.c            # Model loading helpers
│   ├── modelq.c           # Quantized model helpers
│   ├── export.c           # Model export/conversion tool
│   ├── llmfs.c            # 9P file server for multi-model inference
│   ├── simd.h             # SIMD function declarations
│   ├── simd_amd64.s       # SSE2 assembly implementations
│   ├── simdq_amd64.s      # Quantized SIMD (stub, uses C fallback)
│   ├── parallel.h         # Thread pool declarations
│   ├── parallel.c         # Thread pool implementation
│   ├── arch/              # Model architecture plugins
│   │   ├── arch.h         # Architecture interface
│   │   ├── arch.c         # Plugin registry
│   │   ├── llama2.c       # LLaMA 2 (rope_theta=10000)
│   │   ├── llama3.c       # LLaMA 3 (rope_theta=500000)
│   │   └── mistral.c      # Mistral (sliding window)
│   ├── format/            # File format parsers
│   │   ├── gguf.c         # GGUF format parser
│   │   └── safetensors.c  # Safetensors parser
│   ├── pool/              # Model pool management
│   │   ├── pool.h         # Pool interface
│   │   └── pool.c         # LRU eviction, memory tracking
│   ├── mkfile             # Plan 9 build file
│   └── tests/             # Plan 9 test source files
│       ├── mkfile         # Plan 9 test build file
│       ├── test_*.c       # Various unit tests
│       └── ...
├── test/                  # C test harness (Linux host)
│   ├── harness.c          # Main test driver (supports dual-VM testing)
│   ├── reference.c/h      # Reference implementations
│   ├── qemu.c/h           # QEMU VM management (supports socket networking)
│   └── fat.c/h            # FAT disk operations (mtools)
├── qemu/
│   ├── 9front.qcow2       # VM disk image
│   └── shared.img         # FAT disk for file sharing
├── mkfile                 # Root Plan 9 build file
├── stories15M.bin         # Model weights (FP32)
└── tokenizer.bin          # Tokenizer data
```

---

## Running Tests

### Linux Host Testing

The test harness compiles and runs all tests in a Plan 9 QEMU VM:

```bash
make test
```

**Requirements:**
- `qemu-system-x86_64`
- `mtools` (mcopy, mkfs.vfat)
- `curl` (for downloading 9front if needed)

### Test Coverage

| Test | Description |
|------|-------------|
| rmsnorm | RMS normalization |
| softmax | Softmax function |
| matmul | Matrix multiplication |
| rng | Random number generator (xorshift) |
| model_loading | Config and weights loading |
| generation | End-to-end text generation (FP32) |
| generation_simd | FP32 generation with SIMD optimizations |
| quantize | INT8 quantize/dequantize roundtrip |
| quantized_matmul | Quantized matrix multiplication |
| generation_quantized | End-to-end text generation (Q8_0, must match FP32) |
| llmfs_local | 9P file server local mount and generation |
| llmfs_remote | Dual-VM remote 9P inference (CPU serves, terminal mounts) |
| benchmark | Performance benchmark (scalar vs SIMD vs threaded) |
| simd_validation | SIMD correctness vs scalar baseline |
| simd_debug | Minimal SIMD debug test |
| softmax_simd | Softmax SIMD optimization tests |
| rmsnorm_simd | RMSNorm SIMD optimization tests |
| arch_detect | Architecture auto-detection from model file |
| arch_llama3 | LLaMA 3 architecture (rope_theta=500000) |
| format_detect | File format detection (native, GGUF, safetensors) |
| softmax_benchmark | Softmax performance benchmark |
| softmax_accuracy | Softmax numerical accuracy tests |
| gguf_dequant | GGUF Q4_0/Q8_0 dequantization |
| gguf_parse | GGUF header and metadata parsing |
| http | HTTP client (Plan 9 dial) |
| safetensors | Safetensors format parsing |
| pool_lru | Model pool LRU eviction and reference counting |

---

## Building in Plan 9

### Using mkfiles

```bash
# Build everything (from repo root)
mk

# Build only src targets (run, runq, export)
cd src && mk

# Build only test binaries
cd src/tests && mk

# Clean
mk clean
```

### Architecture

9front uses **amd64** (64-bit):
- Compiler: `6c` (NOT `8c` which is for 386)
- Linker: `6l` (NOT `8l`)
- Object files: `.6` extension

### Manual Compilation

```bash
# Compile
6c -w program.c

# Link
6l -o program program.6

# Combined
6c -w program.c && 6l -o program program.6
```

---

## Performance Optimizations

The inference engine supports SIMD vectorization and multi-threading for improved performance.

### SIMD (SSE2)

Matrix-vector multiplication is accelerated using SSE2 packed float instructions:

| Operation | Implementation | Speedup |
|-----------|---------------|---------|
| matmul | SSE2 assembly (simd_amd64.s) | ~5.7x |
| dot_product | SSE2 assembly | ~4x |
| rmsnorm | SSE2 assembly | ~3x |
| softmax | C with 4x unrolling (needs exp()) | ~2x |
| vec_add, vec_scale | SSE2 assembly | ~4x |

The SIMD implementation uses:
- 8-element unrolled loops with 2 accumulators
- 4-element cleanup loop
- Scalar remainder for non-aligned sizes
- Horizontal sum via SHUFPS for final reduction

### Thread Pool

Parallel execution uses Plan 9's libthread:
- Auto-detects CPU count from `/dev/sysstat`
- Channel-based work distribution
- Parallel attention head computation

### Benchmark Results (stories15M, 1024x1024 matmul)

| Mode | GFLOPS | Speedup |
|------|--------|---------|
| Scalar (1 thread) | 3.4 | 1.0x |
| SIMD (1 thread) | 19.0 | 5.7x |
| SIMD (4 threads) | 18.7 | 5.6x |

Note: Multi-threading overhead can exceed benefit for small matrices.

### Runtime Configuration

```c
/* In model.c / modelq.c */
extern OptConfig opt_config;
opt_config.use_simd = 1;    /* Enable SIMD (default) */
opt_config.nthreads = 4;    /* Set thread count (0 = auto) */
```

Command-line flags:
```rc
./run model.bin -z tok.bin --no-simd    # Disable SIMD
./run model.bin -z tok.bin --threads 2  # Set thread count
```

---

## Running Inference

### FP32 Inference (run.c)

In Plan 9:
```rc
6c -w run.c && 6l -o run run.6
./run stories15M.bin -z tokenizer.bin -n 50 -i 'Once upon a time'
```

### INT8 Quantized Inference (runq.c)

First, quantize the model:
```bash
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

---

## Model Export Tool

The `export` tool handles model inspection and conversion:

```bash
# Show model info (works on Linux or Plan 9)
./export info model.bin

# Quantize FP32 model to Q8_0 (32-element groups)
./export quantize model.bin model_q80.bin
```

Build on Linux:
```bash
gcc -o export src/export.c -lm
```

Build on Plan 9:
```rc
6c export.c && 6l -o export export.6
```

---

## LLM File Server (llmfs)

A 9P file server that exposes LLM inference as a Plan 9 filesystem, enabling distributed inference across machines. Supports multiple models with LRU eviction and per-session model binding.

### File System Structure

```
/mnt/llm/
    ctl             # RW: load/unload model, pool commands, server status
    model           # R:  model info (dim, layers, vocab, memory)
    clone           # R:  read to create new session, returns ID
    pool/           # Model pool directory
        count       # R:  number of loaded models
        memory      # R:  total memory used by pool (bytes)
        list        # R:  comma-separated list of loaded model names
    0/              # Session 0 directory
        ctl         # RW: temp, topp, seed, steps, generate, reset
        model       # RW: model name for this session (or "(default)")
        prompt      # W:  write prompt text
        output      # R:  complete output (blocks until done)
        stream      # R:  streaming output (returns tokens as generated)
        status      # R:  idle|generating N/M|done tok/s|error msg
    1/              # Session 1...
```

### Building llmfs

In Plan 9:
```rc
6c -w llmfs.c && 6l -o llmfs llmfs.6
```

### Local Usage

```rc
# Start the file server
./llmfs -s llm

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

### Remote Usage (Distributed Inference)

On the server machine (cpu):
```rc
# Start llmfs
./llmfs -s llm
echo 'load stories15M.bin tokenizer.bin' > /srv/llm/ctl

# Export over network
aux/listen1 -tv tcp!*!564 /bin/exportfs -r /srv/llm
```

On the client machine (terminal):
```rc
# Connect to remote server
srv tcp!cpu!564 llm
mount /srv/llm /mnt/llm

# Use it as if local
cat /mnt/llm/clone
echo 'Once upon a time' > /mnt/llm/0/prompt
echo generate > /mnt/llm/0/ctl
cat /mnt/llm/0/output
```

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
| `load <model> <tokenizer>` | Load default model and tokenizer |
| `unload` | Unload default model |
| `pool-load <name> <model> <tokenizer>` | Load model into pool with given name |
| `pool-unload <name>` | Unload model from pool (fails if in use) |
| `pool-limit <max_models> <max_memory>` | Set pool limits (models and bytes) |

### Multi-Model Usage

```rc
# Load multiple models into the pool
echo 'pool-load small stories15M.bin tokenizer.bin' > /mnt/llm/ctl
echo 'pool-load large llama2-7b.bin tokenizer.bin' > /mnt/llm/ctl

# Check pool status
cat /mnt/llm/pool/count    # "2"
cat /mnt/llm/pool/list     # "small,large"
cat /mnt/llm/pool/memory   # Total memory used

# Create session and bind to specific model
session=`{cat /mnt/llm/clone}
echo 'large' > /mnt/llm/$session/model

# Generate using bound model
echo 'Once upon a time' > /mnt/llm/$session/prompt
echo 'generate' > /mnt/llm/$session/ctl
cat /mnt/llm/$session/output

# Check which model is bound
cat /mnt/llm/$session/model    # "large"
```

The pool uses LRU eviction: when memory or model count limits are reached, the least recently used models with zero references are unloaded. Sessions hold references to their bound models, preventing eviction while in use.

### Downloading Models

HuggingFace integration for automatic model downloads is not yet available. HuggingFace now uses git-lfs and the xet format for model storage, which requires additional tooling support.

To use models with 9ml:

1. **Download manually** from HuggingFace website or using `huggingface-cli`:
   ```bash
   # Install huggingface-cli
   pip install huggingface-hub

   # Download a model
   huggingface-cli download karpathy/tinyllamas --include "*.bin"
   ```

2. **Copy to shared disk** for use in Plan 9:
   ```bash
   # Copy model files to the FAT shared disk
   mcopy -i qemu/shared.img model.bin tokenizer.bin ::
   ```

3. **Load in Plan 9** via llmfs:
   ```rc
   echo 'pool-load mymodel /mnt/host/model.bin /mnt/host/tokenizer.bin' > /mnt/llm/ctl
   ```

---

## Plan 9 C Porting Guide

### Headers

```c
#include <u.h>
#include <libc.h>
```

### Type Mappings

| POSIX | Plan 9 |
|-------|--------|
| `int8_t` | `schar` |
| `uint8_t` | `uchar` |
| `int32_t` | `int` |
| `uint32_t` | `uint` |
| `int64_t` | `vlong` |
| `uint64_t` | `uvlong` |
| `ssize_t` | `vlong` |
| `size_t` | `ulong` |
| `NULL` | `nil` |

### Function Mappings

| POSIX | Plan 9 |
|-------|--------|
| `printf(...)` | `print(...)` |
| `fprintf(stderr, ...)` | `fprint(2, ...)` |
| `exit(0)` | `exits(0)` |
| `exit(1)` | `exits("error")` |
| `clock_gettime()` | `nsec()` |
| `mmap()` | Use `open()` + `read()` + `malloc()` |

### Main Function

```c
void
main(int argc, char *argv[])
{
    // ... code ...
    exits(0);  // or exits("error message")
}
```

### Critical: Struct Padding

Plan 9 may pad structs differently. When reading binary files with struct headers:

```c
// WRONG - struct may be padded
read(fd, &config, sizeof(Config));

// RIGHT - read raw bytes, then copy fields
#define CONFIG_FILE_SIZE (7 * sizeof(int))
int buf[7];
read(fd, buf, CONFIG_FILE_SIZE);
config.dim = buf[0];
config.hidden_dim = buf[1];
// ...
```

### No OpenMP

Remove all `#pragma omp` directives - Plan 9 doesn't support OpenMP.

### Plan 9 amd64 Assembly Calling Convention

When writing assembly for Plan 9 amd64:

```
First argument:     BP (RARG register)
Subsequent args:    Stack at 16(SP), 24(SP), 32(SP), 40(SP)...
Return value:       AX (integer), X0 (float)
Callee-saved:       None (caller saves all)
```

Stack layout (verified empirically):
```
0(SP)   = return address
8(SP)   = padding/frame
16(SP)  = 2nd argument
24(SP)  = 3rd argument
32(SP)  = 4th argument
40(SP)  = 5th argument
```

Important: Use `SUBL/TESTL/JLE` pattern instead of `CMPL/JGE` for loop comparisons - the Plan 9 assembler's comparison semantics differ from standard x86.

### SIMD Assembly Implementation Notes

The `simd_amd64.s` file contains SSE2 implementations for performance-critical operations:

#### Frame Size Matters
Use `$0` frame size (no local stack variables) for simple functions:
```asm
TEXT matmul_simd(SB), $0    // Works - no local frame
TEXT rmsnorm_simd(SB), $0   // Works - no local frame
```

Using `$8` or other frame sizes changes stack argument offsets and can cause memory faults. If you need temp storage, use registers instead of stack.

#### BYTE-Encoded Instructions
Plan 9 assembler doesn't support all SSE instructions. Use BYTE encoding:

```asm
// CVTSI2SS R14, X1 (convert int64 in R14 to float in X1)
// F3 49 0F 2A CE = REX.WB prefix + opcode + ModR/M
BYTE $0xF3; BYTE $0x49; BYTE $0x0F; BYTE $0x2A; BYTE $0xCE

// MOVD R8d, X0 (move 32-bit from R8 to XMM0)
// 66 41 0F 6E C0 = operand-size + REX.B + opcode + ModR/M
BYTE $0x66; BYTE $0x41; BYTE $0x0F; BYTE $0x6E; BYTE $0xC0

// SQRTSS X0, X1 (sqrt of X0 into X1)
// F3 0F 51 C8
BYTE $0xF3; BYTE $0x0F; BYTE $0x51; BYTE $0xC8

// RSQRTSS X0, X1 (approximate 1/sqrt of X0 into X1)
// F3 0F 52 C8
BYTE $0xF3; BYTE $0x0F; BYTE $0x52; BYTE $0xC8
```

#### Approximate vs Exact Instructions
- `RSQRTSS` - approximate reciprocal sqrt, relative error ~0.0004 (fast but inaccurate)
- `SQRTSS` + `DIVSS` - exact sqrt (slower but matches scalar `sqrtf()`)

For rmsnorm, use exact SQRTSS to match scalar output:
```asm
// Compute exact 1/sqrt
SQRTSS X0, X1           // X1 = sqrt(X0)
MOVL $0x3F800000, R8    // 1.0f in IEEE 754
MOVD R8, X0             // X0 = 1.0
DIVSS X1, X0            // X0 = 1.0 / sqrt(...)
```

#### Plan 9 FPU Exception Handling
Plan 9 enables floating-point exceptions by default. To disable:
```c
setfcr(getfcr() & ~(FPINVAL|FPZDIV|FPOVFL|FPUNFL|FPINEX));
```

This affects x87 FPU but SSE MXCSR is typically initialized with exceptions masked (0x1F80).

#### Debugging Tips
1. Memory faults often indicate wrong stack offsets - verify with matmul_simd pattern
2. Denormal exceptions suggest RSQRTSS with very small values - use SQRTSS+DIVSS
3. Add pointer validation at function entry for debugging:
```asm
TESTQ DI, DI
JZ bad_ptr
TESTQ SI, SI
JZ bad_ptr
```

### No bsearch

Implement binary search manually:

```c
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    int lo = 0, hi = vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(str, sorted_vocab[mid].str);
        if (cmp == 0) return sorted_vocab[mid].id;
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}
```

---

## QEMU VM (for Linux host testing)

The test harness manages QEMU automatically. For manual debugging:

### Boot Sequence

1. `bootargs` prompt -> Press Enter (accept default)
2. `user` prompt -> Press Enter (accept default: glenda)
3. Reach `term%` prompt (rc shell)

### Mounting Shared Disk in Plan 9

```rc
dossrv -f /dev/sdG0/data shared
mount -c /srv/shared /mnt/host
```

---

## Troubleshooting

### "file does not exist" in Plan 9

The shared disk may not be mounted. Mount it manually:
```rc
dossrv -f /dev/sdG0/data shared
mount -c /srv/shared /mnt/host
```

### Compilation errors about missing functions

Common missing functions in Plan 9:
- `bsearch` - implement manually
- `round` - use `floor(x + 0.5f)`
- `sqrtf/expf/etc` - use `sqrt/exp` (Plan 9 has double versions)

### Generation output is garbage

Check struct padding - use raw byte reading for binary file headers.

---

## Resources

- [9front](https://9front.org)
- [Plan 9 C Programming](http://doc.cat-v.org/plan_9/programming/c_programming_in_plan_9)
- [llama2.c](https://github.com/karpathy/llama2.c)
