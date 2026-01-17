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
    float first_floats[10];

    if (argc < 2) {
        fprintf(stderr, "usage: test_load model.bin\n");
        return 1;
    }

    f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "can't open %s\n", argv[1]);
        return 1;
    }

    if (fread(&config, sizeof(Config), 1, f) != 1) {
        fprintf(stderr, "failed to read config\n");
        return 1;
    }

    printf("Config:\n");
    printf("  dim: %d\n", config.dim);
    printf("  hidden_dim: %d\n", config.hidden_dim);
    printf("  n_layers: %d\n", config.n_layers);
    printf("  n_heads: %d\n", config.n_heads);
    printf("  n_kv_heads: %d\n", config.n_kv_heads);
    printf("  vocab_size: %d\n", config.vocab_size);
    printf("  seq_len: %d\n", config.seq_len);

    // Read first 10 floats of weights
    if (fread(first_floats, sizeof(first_floats), 1, f) != 1) {
        fprintf(stderr, "failed to read floats\n");
        return 1;
    }

    printf("\nFirst 10 weight floats:\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] %f\n", i, first_floats[i]);
    }

    fclose(f);
    return 0;
}
