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
    float weights[100];

    if (argc < 2) {
        fprint(2, "usage: test_model_load model.bin\n");
        exits("usage");
    }

    fd = open(argv[1], OREAD);
    if (fd < 0) {
        fprint(2, "can't open %s\n", argv[1]);
        exits("open");
    }

    n = read(fd, &config, sizeof(Config));
    print("config_read=%ld expected=%d\n", n, (int)sizeof(Config));
    print("dim=%d\n", config.dim);
    print("hidden_dim=%d\n", config.hidden_dim);
    print("n_layers=%d\n", config.n_layers);
    print("n_heads=%d\n", config.n_heads);
    print("n_kv_heads=%d\n", config.n_kv_heads);
    print("vocab_size=%d\n", config.vocab_size);
    print("seq_len=%d\n", config.seq_len);

    n = read(fd, weights, sizeof(weights));
    print("weights_read=%ld\n", n);
    for (int i = 0; i < 10; i++) {
        print("w[%d]=%.6f\n", i, weights[i]);
    }

    close(fd);
    exits(0);
}
