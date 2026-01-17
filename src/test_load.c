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
    float first_floats[10];

    USED(argc);

    if (argc < 2) {
        fprint(2, "usage: test_load model.bin\n");
        exits("usage");
    }

    fd = open(argv[1], OREAD);
    if (fd < 0) {
        fprint(2, "can't open %s\n", argv[1]);
        exits("open");
    }

    n = read(fd, &config, sizeof(Config));
    if (n != sizeof(Config)) {
        fprint(2, "failed to read config: got %ld, expected %d\n", n, (int)sizeof(Config));
        exits("read");
    }

    print("Config:\n");
    print("  dim: %d\n", config.dim);
    print("  hidden_dim: %d\n", config.hidden_dim);
    print("  n_layers: %d\n", config.n_layers);
    print("  n_heads: %d\n", config.n_heads);
    print("  n_kv_heads: %d\n", config.n_kv_heads);
    print("  vocab_size: %d\n", config.vocab_size);
    print("  seq_len: %d\n", config.seq_len);

    // Read first 10 floats of weights
    n = read(fd, first_floats, sizeof(first_floats));
    if (n != sizeof(first_floats)) {
        fprint(2, "failed to read floats\n");
        exits("read");
    }

    print("\nFirst 10 weight floats:\n");
    for (int i = 0; i < 10; i++) {
        print("  [%d] %f\n", i, first_floats[i]);
    }

    close(fd);
    exits(0);
}
