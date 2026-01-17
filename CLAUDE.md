# 9ml - Machine Learning Models for Plan 9

Port of [llama2.c](https://github.com/karpathy/llama2.c) to Plan 9 (9front).

## Rules

1. **NEVER skip tests.** All tests are mandatory. If a test cannot run, fix the environment - do not skip.
2. **All changes must be tested.** Run tests before committing:
   - **Linux host:** `cd test && gcc -o harness *.c -lm && ./harness`
   - **Plan 9 native:** `mk` (compiles all targets)
3. **Tests must pass.** Do not merge if tests fail.

---

## Quick Start

### Running Tests (Linux Host)

```bash
# Compile and run the test harness
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
│   ├── mkfile             # Plan 9 build file
│   └── tests/             # Plan 9 test source files
│       ├── mkfile         # Plan 9 test build file
│       ├── test_rmsnorm.c
│       ├── test_softmax.c
│       ├── test_matmul.c
│       ├── test_rng.c
│       ├── test_model_loading.c
│       ├── test_quantize.c
│       └── test_quantized_matmul.c
├── test/                  # C test harness (Linux host)
│   ├── harness.c          # Main test driver
│   ├── reference.c/h      # Reference implementations
│   ├── qemu.c/h           # QEMU VM management
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
cd test
gcc -o harness *.c -lm
./harness
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
| generation | End-to-end text generation |
| quantize | INT8 quantize/dequantize roundtrip |
| quantized_matmul | Quantized matrix multiplication |

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
