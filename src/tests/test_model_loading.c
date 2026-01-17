/* Test model config and weights loading */
#include "model.c"

void
main(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    int fd = open("/mnt/host/stories15M.bin", OREAD);
    if (fd < 0) {
        fprint(2, "Could not open model file\n");
        exits("open");
    }

    /* Read config as raw ints to avoid struct padding issues */
    int buf[7];
    if (read(fd, buf, 7 * sizeof(int)) != 7 * sizeof(int)) {
        fprint(2, "Failed to read config\n");
        close(fd);
        exits("read");
    }

    Config c;
    c.dim = buf[0];
    c.hidden_dim = buf[1];
    c.n_layers = buf[2];
    c.n_heads = buf[3];
    c.n_kv_heads = buf[4];
    c.vocab_size = buf[5];
    c.seq_len = buf[6];
    if (c.vocab_size < 0) c.vocab_size = -c.vocab_size;

    print("dim=%d\n", c.dim);
    print("hidden_dim=%d\n", c.hidden_dim);
    print("n_layers=%d\n", c.n_layers);
    print("n_heads=%d\n", c.n_heads);
    print("n_kv_heads=%d\n", c.n_kv_heads);
    print("vocab_size=%d\n", c.vocab_size);
    print("seq_len=%d\n", c.seq_len);

    float w[10];
    if (read(fd, w, sizeof(w)) != sizeof(w)) {
        fprint(2, "Failed to read weights\n");
        close(fd);
        exits("read");
    }

    for (int i = 0; i < 10; i++) {
        print("w%d=%.6f\n", i, w[i]);
    }

    close(fd);
    exits(0);
}
