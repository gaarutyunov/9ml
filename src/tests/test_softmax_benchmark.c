/* Softmax benchmark - compare all softmax modes */
#include "model.c"

#define BENCH_SIZE 32000   /* Typical vocab size */
#define BENCH_ITERS 100
#define WARMUP_ITERS 10

typedef struct {
    char *name;
    int mode;
    double ms_per_call;
    double speedup;
    float max_err;
    float avg_err;
    float checksum;
} BenchResult;

static BenchResult results[5];

/* Get time in nanoseconds */
static uvlong
get_time(void)
{
    return nsec();
}

void
threadmain(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    float *x = malloc(BENCH_SIZE * sizeof(float));
    float *baseline = malloc(BENCH_SIZE * sizeof(float));
    float *scratch = malloc(BENCH_SIZE * sizeof(float));

    if (!x || !baseline || !scratch) {
        print("FAIL: malloc failed\n");
        threadexits("malloc");
    }

    /* Initialize with reproducible data */
    uvlong seed = 12345;
    for (int i = 0; i < BENCH_SIZE; i++) {
        seed = seed * 1103515245 + 12345;
        x[i] = (float)(seed & 0xFFFF) / 65536.0f * 10.0f - 5.0f;  /* Range -5 to 5 */
    }

    print("=== Softmax Benchmark ===\n");
    print("Size: %d, Iterations: %d\n\n", BENCH_SIZE, BENCH_ITERS);

    /* Run scalar baseline first */
    memcpy(baseline, x, BENCH_SIZE * sizeof(float));
    opt_config.softmax_mode = 0;
    softmax(baseline, BENCH_SIZE);

    char *mode_names[] = {"Scalar", "Partial SIMD", "Schraudolph", "Polynomial", "LUT"};

    /* Benchmark each mode */
    for (int mode = 0; mode <= 4; mode++) {
        results[mode].name = mode_names[mode];
        results[mode].mode = mode;
        opt_config.softmax_mode = mode;

        /* Warmup */
        for (int iter = 0; iter < WARMUP_ITERS; iter++) {
            memcpy(scratch, x, BENCH_SIZE * sizeof(float));
            softmax(scratch, BENCH_SIZE);
        }

        /* Timed runs */
        uvlong start = get_time();
        for (int iter = 0; iter < BENCH_ITERS; iter++) {
            memcpy(scratch, x, BENCH_SIZE * sizeof(float));
            softmax(scratch, BENCH_SIZE);
        }
        uvlong end = get_time();

        double total_ms = (end - start) / 1000000.0;
        results[mode].ms_per_call = total_ms / BENCH_ITERS;

        /* Calculate error vs baseline */
        memcpy(scratch, x, BENCH_SIZE * sizeof(float));
        softmax(scratch, BENCH_SIZE);

        float max_err = 0.0f;
        float sum_err = 0.0f;
        float checksum = 0.0f;
        for (int i = 0; i < BENCH_SIZE; i++) {
            float diff = scratch[i] - baseline[i];
            if (diff < 0) diff = -diff;
            if (diff > max_err) max_err = diff;
            sum_err += diff;
            checksum += scratch[i];
        }
        results[mode].max_err = max_err;
        results[mode].avg_err = sum_err / BENCH_SIZE;
        results[mode].checksum = checksum;
    }

    /* Calculate speedups */
    double baseline_ms = results[0].ms_per_call;
    for (int mode = 0; mode <= 4; mode++) {
        results[mode].speedup = baseline_ms / results[mode].ms_per_call;
    }

    /* Print results table */
    print("Mode           ms/call  Speedup  MaxErr     AvgErr     Checksum\n");
    print("-------------------------------------------------------------\n");
    for (int mode = 0; mode <= 4; mode++) {
        print("%-14s %7.3f  %5.2fx   %.2e   %.2e   %.6f\n",
              results[mode].name,
              results[mode].ms_per_call,
              results[mode].speedup,
              results[mode].max_err,
              results[mode].avg_err,
              results[mode].checksum);
    }

    /* Determine best mode */
    print("\n=== Recommendation ===\n");
    int best_mode = 0;
    double best_speedup = 1.0;
    for (int mode = 1; mode <= 4; mode++) {
        /* Select if speedup > 1.5x AND error < 1e-3 */
        if (results[mode].speedup > best_speedup &&
            results[mode].max_err < 1e-3f) {
            best_mode = mode;
            best_speedup = results[mode].speedup;
        }
    }

    if (best_mode > 0) {
        print("Best mode: %d (%s) - %.2fx speedup, max error %.2e\n",
              best_mode, results[best_mode].name,
              results[best_mode].speedup, results[best_mode].max_err);
    } else {
        print("Best mode: 0 (Scalar) - no faster mode meets accuracy requirements\n");
    }

    /* Verify checksums are reasonable (should all be ~1.0 for valid probability distributions) */
    int checksum_ok = 1;
    for (int mode = 0; mode <= 4; mode++) {
        float diff = results[mode].checksum - 1.0f;
        if (diff < 0) diff = -diff;
        if (diff > 0.01f) {
            print("WARNING: Mode %d checksum %.6f deviates from 1.0\n",
                  mode, results[mode].checksum);
            checksum_ok = 0;
        }
    }

    if (checksum_ok) {
        print("\nCHECKSUM VALIDATION PASSED\n");
    } else {
        print("\nCHECKSUM VALIDATION FAILED\n");
    }

    free(x);
    free(baseline);
    free(scratch);

    threadexits(checksum_ok ? 0 : "checksum failed");
}
