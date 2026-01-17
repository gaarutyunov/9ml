# 9ml - Machine Learning Models for Plan 9

Port of [llama2.c](https://github.com/karpathy/llama2.c) to Plan 9 (9front).

## Quick Start

```bash
# Run the test suite
python3 tests/run_tests.py

# Copy files to Plan 9 and run inference
./qemu/copy-to-shared.sh src/run.c model.bin tokenizer.bin
./qemu/run-cmd.sh "6c /mnt/host/run.c && 6l -o run run.6 && ./run /mnt/host/model.bin -z /mnt/host/tokenizer.bin"
```

---

## Project Structure

```
9ml/
├── src/
│   ├── run.c          # FP32 inference (ported)
│   └── runq.c         # INT8 quantized inference (ported)
├── tests/
│   └── run_tests.py   # Automated test suite
├── qemu/
│   ├── 9front.qcow2   # 9front disk image
│   ├── shared.img     # FAT32 shared disk (128MB)
│   ├── run-cmd.sh     # Execute commands in Plan 9
│   ├── run-qemu.sh    # Start QEMU manually
│   ├── copy-to-shared.sh    # Copy files to shared disk
│   └── copy-from-shared.sh  # Copy files from shared disk
└── feature_list.json  # Task tracking (56/56 complete)
```

---

## Running Commands in Plan 9

### Execute a Single Command

```bash
./qemu/run-cmd.sh "your command here"
```

This script:
1. Starts QEMU with 9front
2. Boots to the rc shell
3. Mounts the shared FAT disk at `/mnt/host`
4. Executes your command
5. Captures and returns output
6. Shuts down QEMU

### Examples

```bash
# List files in Plan 9
./qemu/run-cmd.sh "ls /mnt/host"

# Compile C code
./qemu/run-cmd.sh "6c /mnt/host/myfile.c"

# Link binary
./qemu/run-cmd.sh "6l -o myprogram myfile.6"

# Run a program
./qemu/run-cmd.sh "./myprogram arg1 arg2"

# Chain commands
./qemu/run-cmd.sh "6c /mnt/host/run.c && 6l -o run run.6 && ./run /mnt/host/model.bin"
```

---

## File Sharing with Plan 9

Files are shared via a FAT32 disk image mounted at `/mnt/host` in Plan 9.

### Copy Files TO Plan 9

```bash
# Single file
./qemu/copy-to-shared.sh /path/to/file.c

# Multiple files
./qemu/copy-to-shared.sh file1.c file2.c model.bin

# Files appear at /mnt/host/ in Plan 9
```

### Copy Files FROM Plan 9

```bash
# Copy all files from shared disk to destination
./qemu/copy-from-shared.sh /destination/directory/
```

### Manual Access (macOS)

```bash
# Mount the shared disk on macOS
hdiutil attach -mountpoint /tmp/shared qemu/shared.img

# Access files
ls /tmp/shared/

# Unmount before running QEMU
hdiutil detach /tmp/shared
```

**Important:** The shared disk cannot be mounted on both macOS and QEMU simultaneously.

---

## Running Tests

The test suite compares Plan 9 output against Python reference implementations.

```bash
# Run all tests
python3 tests/run_tests.py
```

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

## Compiling in Plan 9

### Architecture

9front uses **amd64** (64-bit):
- Compiler: `6c` (NOT `8c` which is for 386)
- Linker: `6l` (NOT `8l`)
- Object files: `.6` extension

### Compile and Link

```bash
# Compile
./qemu/run-cmd.sh "6c /mnt/host/program.c"

# Link
./qemu/run-cmd.sh "6l -o program program.6"

# Or chain them
./qemu/run-cmd.sh "6c /mnt/host/program.c && 6l -o program program.6"
```

### Compiler Flags

```bash
# Suppress warnings
./qemu/run-cmd.sh "6c -w /mnt/host/program.c"
```

---

## Running Inference

### FP32 Inference (run.c)

```bash
# Copy files
./qemu/copy-to-shared.sh src/run.c stories15M.bin tokenizer.bin

# Compile and run
./qemu/run-cmd.sh "6c -w /mnt/host/run.c && 6l -o run run.6 && ./run /mnt/host/stories15M.bin -z /mnt/host/tokenizer.bin -n 50 -i 'Once upon a time'"
```

### INT8 Quantized Inference (runq.c)

First, export a quantized model:
```bash
python export.py stories15M_q80.bin --version 2 --checkpoint stories15M.pt
```

Then run:
```bash
./qemu/copy-to-shared.sh src/runq.c stories15M_q80.bin tokenizer.bin

./qemu/run-cmd.sh "6c -w /mnt/host/runq.c && 6l -o runq runq.6 && ./runq /mnt/host/stories15M_q80.bin -z /mnt/host/tokenizer.bin -n 50"
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

## QEMU Setup Details

### Starting QEMU Manually

```bash
./qemu/run-qemu.sh
```

Or directly:
```bash
/opt/local/bin/qemu-system-x86_64 \
    -m 512 -cpu max -accel hvf \
    -drive file=qemu/9front.qcow2,format=qcow2,if=virtio \
    -drive file=qemu/shared.img,format=raw,if=virtio \
    -display none -serial mon:stdio
```

### Boot Sequence

1. `bootargs` prompt → Press Enter (accept default)
2. `user` prompt → Press Enter (accept default: glenda)
3. Reach `term%` prompt (rc shell)

### Mounting Shared Disk in Plan 9

```rc
dossrv -f /dev/sdG0/dos shared
mount -c /srv/shared /mnt/host
```

(This is done automatically by `run-cmd.sh`)

### Resizing Shared Disk

```bash
# Remove old disk
rm qemu/shared.img

# Create larger disk (e.g., 256MB)
hdiutil create -size 256m -fs MS-DOS -volname SHARED qemu/shared.img
mv qemu/shared.img.dmg qemu/shared.img
```

---

## Troubleshooting

### "file does not exist" in Plan 9

The shared disk may not be mounted. Check:
```bash
./qemu/run-cmd.sh "ls /mnt/host"
```

If empty, the mount failed. Try restarting QEMU.

### "No space left on device"

The shared disk is full. Either:
1. Remove files from the shared disk
2. Create a larger shared disk (see above)

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
