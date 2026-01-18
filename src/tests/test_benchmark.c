/* Performance benchmark test for SIMD and parallel optimizations */
#include "model.c"

/* Benchmark dimensions - large enough to see performance differences */
#define BENCH_D     1024   /* output dimension */
#define BENCH_N     1024   /* input dimension */
#define BENCH_ITERS 100    /* iterations per test */

/* Checksum tolerance for comparing SIMD vs scalar */
#define CHECKSUM_TOLERANCE 0.001f

/* Simple linear congruential generator for reproducible data */
static uvlong bench_seed = 12345;
static float rand_float(void) {
    bench_seed = bench_seed * 1103515245 + 12345;
    return (float)(bench_seed & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

/* Compute checksum of output array */
static float compute_checksum(float *arr, int n) {
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

    /* Allocate test data */
    float *w = malloc(BENCH_D * BENCH_N * sizeof(float));
    float *x = malloc(BENCH_N * sizeof(float));
    float *out = malloc(BENCH_D * sizeof(float));

    if (!w || !x || !out) {
        fprint(2, "malloc failed\n");
        threadexits("malloc");
    }

    /* Initialize with reproducible random data */
    bench_seed = 12345;
    for (int i = 0; i < BENCH_D * BENCH_N; i++) {
        w[i] = rand_float() - 0.5f;
    }
    for (int i = 0; i < BENCH_N; i++) {
        x[i] = rand_float() - 0.5f;
    }

    print("=== 9ml Performance Benchmark ===\n");
    print("Matrix: %dx%d, Iterations: %d\n\n", BENCH_D, BENCH_N, BENCH_ITERS);

    vlong start, elapsed;
    double gflops, ms_per_iter;
    double flops_per_iter = 2.0 * BENCH_D * BENCH_N; /* multiply-add per element */

    /* Store results for comparison */
    double baseline_gflops = 0;
    float baseline_checksum = 0.0f;
    int checksum_failures = 0;

    /* Test 1: Baseline (scalar, single-threaded) */
    opt_config.use_simd = 0;
    opt_config.nthreads = 1;
    opt_init();

    /* Warmup */
    matmul(out, x, w, BENCH_N, BENCH_D);

    start = nsec();
    for (int i = 0; i < BENCH_ITERS; i++) {
        matmul(out, x, w, BENCH_N, BENCH_D);
    }
    elapsed = nsec() - start;

    ms_per_iter = (double)elapsed / (BENCH_ITERS * 1000000.0);
    gflops = (flops_per_iter * BENCH_ITERS) / (double)elapsed;
    baseline_gflops = gflops;

    /* Compute baseline checksum - this is the reference for all other modes */
    baseline_checksum = compute_checksum(out, BENCH_D);

    print("Mode BASELINE (scalar, 1 thread):\n");
    print("  %.3f GFLOPS (%.2f ms per matmul)\n", gflops, ms_per_iter);
    print("  Checksum: %.6f (reference)\n\n", baseline_checksum);

    opt_cleanup();

    /* Test 2: Threading only (scalar, multi-threaded) - RUN BEFORE SIMD */
    opt_config.use_simd = 0;
    opt_config.nthreads = 0; /* auto-detect */
    opt_init();

    /* Clear output buffer to avoid stale data masking bugs */
    for (int i = 0; i < BENCH_D; i++) out[i] = 0.0f;

    /* Warmup */
    matmul(out, x, w, BENCH_N, BENCH_D);

    start = nsec();
    for (int i = 0; i < BENCH_ITERS; i++) {
        matmul(out, x, w, BENCH_N, BENCH_D);
    }
    elapsed = nsec() - start;

    ms_per_iter = (double)elapsed / (BENCH_ITERS * 1000000.0);
    gflops = (flops_per_iter * BENCH_ITERS) / (double)elapsed;

    /* Validate checksum against baseline */
    float thread_checksum = compute_checksum(out, BENCH_D);
    float thread_diff = thread_checksum - baseline_checksum;
    if (thread_diff < 0) thread_diff = -thread_diff;

    print("Mode THREAD_ONLY (scalar, %d threads):\n", opt_config.nthreads);
    print("  %.3f GFLOPS (%.2f ms per matmul) [%.1fx speedup]\n",
          gflops, ms_per_iter, gflops / baseline_gflops);
    print("  Checksum: %.6f", thread_checksum);
    if (thread_diff > CHECKSUM_TOLERANCE) {
        print(" FAIL (diff=%.9f)\n\n", thread_diff);
        checksum_failures++;
    } else {
        print(" OK\n\n");
    }

    opt_cleanup();

    /* Test 3: SIMD only (single-threaded) */
    opt_config.use_simd = 1;
    opt_config.nthreads = 1;
    opt_init();

    /* Clear output buffer to avoid stale data masking bugs */
    for (int i = 0; i < BENCH_D; i++) out[i] = 0.0f;

    /* Warmup */
    matmul(out, x, w, BENCH_N, BENCH_D);

    start = nsec();
    for (int i = 0; i < BENCH_ITERS; i++) {
        matmul(out, x, w, BENCH_N, BENCH_D);
    }
    elapsed = nsec() - start;

    ms_per_iter = (double)elapsed / (BENCH_ITERS * 1000000.0);
    gflops = (flops_per_iter * BENCH_ITERS) / (double)elapsed;

    /* Validate checksum against baseline */
    float simd_checksum = compute_checksum(out, BENCH_D);
    float simd_diff = simd_checksum - baseline_checksum;
    if (simd_diff < 0) simd_diff = -simd_diff;

    print("Mode SIMD_ONLY (SSE2, 1 thread):\n");
    print("  %.3f GFLOPS (%.2f ms per matmul) [%.1fx speedup]\n",
          gflops, ms_per_iter, gflops / baseline_gflops);
    print("  Checksum: %.6f", simd_checksum);
    if (simd_diff > CHECKSUM_TOLERANCE) {
        print(" FAIL (diff=%.9f)\n\n", simd_diff);
        checksum_failures++;
    } else {
        print(" OK\n\n");
    }

    opt_cleanup();

    /* Test 4: Full optimization (SIMD + threading) */
    opt_config.use_simd = 1;
    opt_config.nthreads = 0; /* auto-detect */
    opt_init();

    /* Clear output buffer to avoid stale data masking bugs */
    for (int i = 0; i < BENCH_D; i++) out[i] = 0.0f;

    /* Warmup */
    matmul(out, x, w, BENCH_N, BENCH_D);

    start = nsec();
    for (int i = 0; i < BENCH_ITERS; i++) {
        matmul(out, x, w, BENCH_N, BENCH_D);
    }
    elapsed = nsec() - start;

    ms_per_iter = (double)elapsed / (BENCH_ITERS * 1000000.0);
    gflops = (flops_per_iter * BENCH_ITERS) / (double)elapsed;

    /* Validate checksum against baseline */
    float full_checksum = compute_checksum(out, BENCH_D);
    float full_diff = full_checksum - baseline_checksum;
    if (full_diff < 0) full_diff = -full_diff;

    print("Mode FULL (SSE2, %d threads):\n", opt_config.nthreads);
    print("  %.3f GFLOPS (%.2f ms per matmul) [%.1fx speedup]\n",
          gflops, ms_per_iter, gflops / baseline_gflops);
    print("  Checksum: %.6f", full_checksum);
    if (full_diff > CHECKSUM_TOLERANCE) {
        print(" FAIL (diff=%.9f)\n\n", full_diff);
        checksum_failures++;
    } else {
        print(" OK\n\n");
    }

    opt_cleanup();

    /* Summary */
    print("Benchmark complete.\n");
    print("Baseline checksum: %.6f\n", baseline_checksum);
    if (checksum_failures > 0) {
        print("CHECKSUM VALIDATION FAILED: %d mode(s) produced incorrect results\n", checksum_failures);
    } else {
        print("CHECKSUM VALIDATION PASSED: All modes match baseline\n");
    }

    free(w);
    free(x);
    free(out);

    threadexits(checksum_failures > 0 ? "checksum validation failed" : 0);
}
