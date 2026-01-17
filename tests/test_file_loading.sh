#!/bin/bash
# test_file_loading.sh - Automated test for file loading (Task 14)
# Compares model loading output between host and Plan 9

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
QEMU_DIR="$PROJECT_DIR/qemu"
SRC_DIR="$PROJECT_DIR/src"
MODEL_PATH="/Users/germanarutyunov/Projects/plan9/llama2.c/stories15M.bin"

echo "=== Test: File Loading Matches Reference ==="

# Create test program
cat > "$SRC_DIR/test_load.c" << 'EOF'
#include <u.h>
#include <libc.h>

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

void
main(int argc, char *argv[])
{
    int fd;
    Config config;
    long n;
    float floats[100];

    USED(argc);
    if (argc < 2) exits("usage");

    fd = open(argv[1], OREAD);
    if (fd < 0) exits("open");

    n = read(fd, &config, sizeof(Config));
    if (n != sizeof(Config)) exits("read config");

    print("%d %d %d %d %d %d %d\n",
        config.dim, config.hidden_dim, config.n_layers,
        config.n_heads, config.n_kv_heads, config.vocab_size, config.seq_len);

    n = read(fd, floats, sizeof(floats));
    if (n != sizeof(floats)) exits("read floats");

    for (int i = 0; i < 100; i++) {
        print("%.6f\n", floats[i]);
    }
    close(fd);
    exits(0);
}
EOF

# Create host version
cat > "$SRC_DIR/test_load_host.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

int main(int argc, char *argv[]) {
    FILE *f;
    Config config;
    float floats[100];

    if (argc < 2) return 1;
    f = fopen(argv[1], "rb");
    if (!f) return 1;

    if (fread(&config, sizeof(Config), 1, f) != 1) return 1;
    printf("%d %d %d %d %d %d %d\n",
        config.dim, config.hidden_dim, config.n_layers,
        config.n_heads, config.n_kv_heads, config.vocab_size, config.seq_len);

    if (fread(floats, sizeof(floats), 1, f) != 1) return 1;
    for (int i = 0; i < 100; i++) {
        printf("%.6f\n", floats[i]);
    }
    fclose(f);
    return 0;
}
EOF

# Compile and run host version
echo "Running on host..."
gcc -o "$PROJECT_DIR/test_load_host" "$SRC_DIR/test_load_host.c"
"$PROJECT_DIR/test_load_host" "$MODEL_PATH" > /tmp/host_output.txt

# Copy files and run in Plan 9
echo "Running in Plan 9..."
"$QEMU_DIR/copy-to-shared.sh" "$SRC_DIR/test_load.c" "$MODEL_PATH" > /dev/null
"$QEMU_DIR/run-cmd.sh" "6c /mnt/host/test_load.c && 6l -o test_load test_load.6 && ./test_load /mnt/host/stories15M.bin" > /tmp/plan9_output.txt

# Compare
echo "Comparing outputs..."
if diff -q /tmp/host_output.txt /tmp/plan9_output.txt > /dev/null 2>&1; then
    echo "PASS: File loading matches reference"
    exit 0
else
    echo "FAIL: Outputs differ"
    echo "--- Host output (first 20 lines):"
    head -20 /tmp/host_output.txt
    echo "--- Plan 9 output (first 20 lines):"
    head -20 /tmp/plan9_output.txt
    exit 1
fi
