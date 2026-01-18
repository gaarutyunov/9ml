/* Quantized matmul benchmark
 *
 * Compares scalar vs unrolled implementations for quantized matrix multiplication.
 */
#include "modelq.c"

#define BENCH_D 1024
#define BENCH_N 1024
#define BENCH_ITERS 100

/* External scalar and unrolled implementations for direct comparison */
extern void matmul_q8_scalar(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);
extern void matmul_q8_unrolled(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);

/* Compute checksum */
static float checksum(float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

void
threadmain(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    /* Set group size */
    GS = 32;

    /* Allocate weight matrix (d x n) - quantized */
    schar *wq = malloc(BENCH_D * BENCH_N * sizeof(schar));
    float *ws = malloc((BENCH_D * BENCH_N / GS) * sizeof(float));

    /* Allocate input vector (n) - quantized */
    schar *xq = malloc(BENCH_N * sizeof(schar));
    float *xs = malloc((BENCH_N / GS) * sizeof(float));

    /* Output vector */
    float *out = malloc(BENCH_D * sizeof(float));

    if (!wq || !ws || !xq || !xs || !out) {
        fprint(2, "malloc failed\n");
        exits("malloc");
    }

    /* Initialize with pseudo-random data */
    uvlong seed = 12345;
    for (int i = 0; i < BENCH_D * BENCH_N; i++) {
        seed = seed * 1103515245 + 12345;
        wq[i] = (schar)((seed >> 16) & 0x7F) - 64;  /* Range: -64 to 63 */
    }
    for (int i = 0; i < BENCH_D * BENCH_N / GS; i++) {
        seed = seed * 1103515245 + 12345;
        ws[i] = ((float)(seed & 0xFFFF) / 65536.0f) * 0.1f;  /* Small positive scales */
    }
    for (int i = 0; i < BENCH_N; i++) {
        seed = seed * 1103515245 + 12345;
        xq[i] = (schar)((seed >> 16) & 0x7F) - 64;
    }
    for (int i = 0; i < BENCH_N / GS; i++) {
        seed = seed * 1103515245 + 12345;
        xs[i] = ((float)(seed & 0xFFFF) / 65536.0f) * 0.1f;
    }

    /* Create QuantizedTensor structs */
    QuantizedTensor w_tensor = { .q = wq, .s = ws };
    QuantizedTensor x_tensor = { .q = xq, .s = xs };

    print("=== Quantized Matmul Benchmark ===\n");
    print("Matrix size: %d x %d, GS=%d\n", BENCH_D, BENCH_N, GS);
    print("Iterations: %d\n\n", BENCH_ITERS);

    vlong start, end;
    double elapsed_ms, ops_per_sec, gflops;
    float ref_checksum, test_checksum;

    /* Benchmark scalar version */
    print("Mode SCALAR:\n");
    start = nsec();
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        matmul_q8_scalar(out, &x_tensor, &w_tensor, BENCH_N, BENCH_D);
    }
    end = nsec();
    elapsed_ms = (double)(end - start) / 1000000.0;
    ops_per_sec = (double)BENCH_ITERS * BENCH_D * BENCH_N * 2 / (elapsed_ms / 1000.0);
    gflops = ops_per_sec / 1e9;
    ref_checksum = checksum(out, BENCH_D);
    print("  %.3f GFLOPS (%.2f ms per matmul)\n", gflops, elapsed_ms / BENCH_ITERS);
    print("  Checksum: %.6f (reference)\n", ref_checksum);

    /* Benchmark unrolled version */
    print("Mode UNROLLED:\n");
    start = nsec();
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        matmul_q8_unrolled(out, &x_tensor, &w_tensor, BENCH_N, BENCH_D);
    }
    end = nsec();
    elapsed_ms = (double)(end - start) / 1000000.0;
    ops_per_sec = (double)BENCH_ITERS * BENCH_D * BENCH_N * 2 / (elapsed_ms / 1000.0);
    gflops = ops_per_sec / 1e9;
    test_checksum = checksum(out, BENCH_D);
    print("  %.3f GFLOPS (%.2f ms per matmul) [%.1fx speedup]\n",
          gflops, elapsed_ms / BENCH_ITERS, gflops / (ref_checksum != 0 ? gflops : 1));
    print("  Checksum: %.6f", test_checksum);

    /* Verify checksum matches */
    float diff = ref_checksum - test_checksum;
    if (diff < 0) diff = -diff;
    if (diff > 0.001f) {
        print(" MISMATCH (diff=%.6f)\n", diff);
        print("\nFAIL: Checksum mismatch\n");
    } else {
        print(" OK\n");
        print("\nPASS: Checksums match\n");
    }

    free(wq);
    free(ws);
    free(xq);
    free(xs);
    free(out);

    threadexits(diff > 0.001f ? "checksum mismatch" : 0);
}
