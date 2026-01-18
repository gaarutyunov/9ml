/* Minimal SIMD debug test */
#include "model.c"

void
threadmain(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    int n = 16, d = 16;

    float *w = malloc(d * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *out_scalar = malloc(d * sizeof(float));
    float *out_simd = malloc(d * sizeof(float));

    /* Initialize: w[i][j] = 0.1, x[j] = 0.1 */
    /* Expected: out[i] = sum(w[i][j] * x[j]) = 16 * 0.01 = 0.16 */
    for (int i = 0; i < d * n; i++) w[i] = 0.1f;
    for (int i = 0; i < n; i++) x[i] = 0.1f;
    for (int i = 0; i < d; i++) {
        out_scalar[i] = -999.0f;
        out_simd[i] = -999.0f;
    }

    print("Test: 16x16 matmul, all weights and inputs = 0.1\n");
    print("Expected output: 16 * 0.1 * 0.1 = 0.16 for each row\n\n");

    /* Run scalar */
    opt_config.use_simd = 0;
    matmul(out_scalar, x, w, n, d);
    print("Scalar[0..3]: %.6f %.6f %.6f %.6f\n",
          out_scalar[0], out_scalar[1], out_scalar[2], out_scalar[3]);

    /* Run SIMD */
    opt_config.use_simd = 1;
    matmul(out_simd, x, w, n, d);
    print("SIMD[0..3]:   %.6f %.6f %.6f %.6f\n",
          out_simd[0], out_simd[1], out_simd[2], out_simd[3]);

    /* Compare */
    int mismatches = 0;
    for (int i = 0; i < d; i++) {
        float diff = out_scalar[i] - out_simd[i];
        if (diff < 0) diff = -diff;
        if (diff > 1e-5f) mismatches++;
    }

    if (mismatches == 0) {
        print("\nPASS: SIMD matches scalar\n");
    } else {
        print("\nFAIL: %d mismatches\n", mismatches);
    }

    free(w);
    free(x);
    free(out_scalar);
    free(out_simd);

    threadexits(mismatches == 0 ? 0 : "failed");
}
