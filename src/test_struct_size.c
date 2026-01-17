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

void main(int, char**) {
    print("sizeof(int)=%d\n", (int)sizeof(int));
    print("sizeof(Config)=%d\n", (int)sizeof(Config));

    /* Read first 40 bytes from file and print as hex */
    int fd = open("/mnt/host/stories15M.bin", OREAD);
    if (fd < 0) {
        print("error: cannot open model file\n");
        exits("open");
    }

    uchar buf[40];
    long n = read(fd, buf, 40);
    print("read %ld bytes\n", n);

    print("raw bytes: ");
    for (int i = 0; i < 40; i++) {
        print("%02x ", buf[i]);
    }
    print("\n");

    /* Interpret as 10 ints */
    int *ints = (int*)buf;
    print("as ints: ");
    for (int i = 0; i < 10; i++) {
        print("%d ", ints[i]);
    }
    print("\n");

    /* Interpret as floats starting at byte 28 */
    float *floats = (float*)(buf + 28);
    print("floats at offset 28: ");
    for (int i = 0; i < 3; i++) {
        print("%.6f ", floats[i]);
    }
    print("\n");

    close(fd);
    exits(0);
}
