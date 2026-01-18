/* Test quantized matmul */
#define DISABLE_OPTIMIZATIONS
#include "modelq.c"

void
main(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    /* Set global group size */
    GS = 32;

    int n = 64, d = 2;
    float x[64], w[128];

    for (int i = 0; i < n; i++) {
        x[i] = (float)(i % 10) / 10.0f;
    }
    for (int i = 0; i < d * n; i++) {
        w[i] = (float)((i / n + i % n) % 10) / 10.0f;
    }

    QuantizedTensor xq, wq;
    xq.q = malloc(n * sizeof(schar));
    xq.s = malloc((n / GS) * sizeof(float));
    wq.q = malloc(d * n * sizeof(schar));
    wq.s = malloc((d * n / GS) * sizeof(float));

    quantize(&xq, x, n);
    quantize(&wq, w, d * n);

    float out[2];
    matmul(out, &xq, &wq, n, d);

    for (int i = 0; i < d; i++) {
        print("%.6f\n", out[i]);
    }

    free(xq.q);
    free(xq.s);
    free(wq.q);
    free(wq.s);
    exits(0);
}
