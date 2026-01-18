/* SIMD Validation Test
 *
 * Compares SIMD output directly against scalar baseline to verify correctness.
 * This test does NOT use DISABLE_OPTIMIZATIONS so both matmul_simd and matmul_scalar
 * are available via opt_config.use_simd.
 */
#include "model.c"

/* Tolerance for floating point comparison */
#define EPSILON 1e-5f

/* Simple LCG for reproducible random data */
static uvlong test_seed = 12345;
static float rand_float(void) {
    test_seed = test_seed * 1103515245 + 12345;
    return (float)(test_seed & 0x7FFFFFFF) / (float)0x7FFFFFFF - 0.5f;
}

/* Compare arrays with tolerance, return number of mismatches */
static int compare_arrays(float *a, float *b, int n, float eps, int verbose) {
    int mismatches = 0;
    float max_diff = 0.0f;
    int max_idx = 0;

    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        if (diff < 0) diff = -diff;
        if (diff > eps) {
            mismatches++;
            if (diff > max_diff) {
                max_diff = diff;
                max_idx = i;
            }
        }
    }

    if (mismatches > 0 && verbose) {
        print("  Max diff: %.9f at index %d (scalar=%.9f, simd=%.9f)\n",
              max_diff, max_idx, a[max_idx], b[max_idx]);
    }

    return mismatches;
}

/* Test matmul SIMD vs scalar for a given size */
static int test_matmul_size(int n, int d, int verbose) {
    /* Allocate buffers */
    float *w = malloc(d * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *out_scalar = malloc(d * sizeof(float));
    float *out_simd = malloc(d * sizeof(float));

    if (!w || !x || !out_scalar || !out_simd) {
        print("FAIL: malloc failed for %dx%d\n", d, n);
        free(w); free(x); free(out_scalar); free(out_simd);
        return 1;
    }

    /* Initialize with reproducible random data */
    test_seed = 12345 + d * 1000 + n;
    for (int i = 0; i < d * n; i++) {
        w[i] = rand_float();
    }
    for (int i = 0; i < n; i++) {
        x[i] = rand_float();
    }

    /* Initialize buffers to known values */
    for (int i = 0; i < d; i++) {
        out_scalar[i] = -999.0f;
        out_simd[i] = -999.0f;
    }

    /* Run scalar version */
    opt_config.use_simd = 0;
    matmul(out_scalar, x, w, n, d);

    /* Run SIMD version */
    opt_config.use_simd = 1;
    matmul(out_simd, x, w, n, d);

    /* Compare results */
    int mismatches = compare_arrays(out_scalar, out_simd, d, EPSILON, verbose);

    if (verbose) {
        /* Print first few values for debugging */
        print("  Scalar[0..2]: %.6f, %.6f, %.6f\n", out_scalar[0], out_scalar[1], out_scalar[2]);
        print("  Stack: [0-1]=%08ux_%08ux [2-3]=%08ux_%08ux [4-5]=%08ux_%08ux [6-7]=%08ux_%08ux\n",
              *(uint*)&out_simd[1], *(uint*)&out_simd[0],
              *(uint*)&out_simd[3], *(uint*)&out_simd[2],
              *(uint*)&out_simd[5], *(uint*)&out_simd[4],
              *(uint*)&out_simd[7], *(uint*)&out_simd[6]);
        print("  Stack: [8-9]=%08ux_%08ux [10-11]=%08ux_%08ux [12-13]=%08ux_%08ux [14-15]=%08ux_%08ux\n",
              *(uint*)&out_simd[9], *(uint*)&out_simd[8],
              *(uint*)&out_simd[11], *(uint*)&out_simd[10],
              *(uint*)&out_simd[13], *(uint*)&out_simd[12],
              *(uint*)&out_simd[15], *(uint*)&out_simd[14]);
        print("  Size %dx%d: ", d, n);
        if (mismatches == 0) {
            print("PASS (all %d elements match)\n", d);
        } else {
            print("FAIL (%d/%d mismatches)\n", mismatches, d);
        }
    }

    free(w);
    free(x);
    free(out_scalar);
    free(out_simd);

    return mismatches;
}

/* Compute checksum of array */
static float compute_checksum(float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

/* Test that benchmarks the SIMD correctness at scale */
static int test_benchmark_correctness(void) {
    int d = 1024;
    int n = 1024;

    float *w = malloc(d * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *out = malloc(d * sizeof(float));

    if (!w || !x || !out) {
        print("FAIL: malloc failed for benchmark\n");
        free(w); free(x); free(out);
        return 1;
    }

    /* Initialize with reproducible data */
    test_seed = 42;
    for (int i = 0; i < d * n; i++) {
        w[i] = rand_float();
    }
    for (int i = 0; i < n; i++) {
        x[i] = rand_float();
    }

    /* Run scalar version - this is the reference */
    opt_config.use_simd = 0;
    matmul(out, x, w, n, d);
    float scalar_checksum = compute_checksum(out, d);

    /* Run SIMD version */
    opt_config.use_simd = 1;
    matmul(out, x, w, n, d);
    float simd_checksum = compute_checksum(out, d);

    /* Compare checksums */
    float diff = scalar_checksum - simd_checksum;
    if (diff < 0) diff = -diff;

    print("  Scalar checksum: %.6f\n", scalar_checksum);
    print("  SIMD checksum:   %.6f\n", simd_checksum);
    print("  Difference:      %.9f\n", diff);

    /* Allow small tolerance for floating point accumulation differences */
    float tolerance = 0.001f;
    if (diff > tolerance) {
        print("FAIL: Checksum mismatch (diff=%.9f, tolerance=%.9f)\n", diff, tolerance);
        free(w); free(x); free(out);
        return 1;
    }

    free(w);
    free(x);
    free(out);
    return 0;
}

void
threadmain(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    int total_failures = 0;

    print("=== SIMD Validation Test ===\n\n");

    /* Test 1: Small matrices with detailed output */
    print("Test 1: Small matrix validation (detailed)\n");

    /* Start with 16x16 to get stack dump */
    total_failures += test_matmul_size(16, 16, 1);

    /* Test various sizes that exercise different SIMD code paths */
    /* Sizes that are multiples of 8 (full SIMD loop) */
    total_failures += test_matmul_size(8, 8, 1);
    total_failures += test_matmul_size(32, 32, 1);

    /* Sizes that require 4-element cleanup (n % 8 >= 4) */
    total_failures += test_matmul_size(12, 12, 1);
    total_failures += test_matmul_size(20, 20, 1);

    /* Sizes that require scalar cleanup (n % 4 != 0) */
    total_failures += test_matmul_size(10, 10, 1);
    total_failures += test_matmul_size(15, 15, 1);
    total_failures += test_matmul_size(17, 17, 1);

    /* Test 2: Medium matrices */
    print("\nTest 2: Medium matrix validation\n");
    total_failures += test_matmul_size(64, 64, 1);
    total_failures += test_matmul_size(128, 128, 1);
    total_failures += test_matmul_size(256, 256, 1);

    /* Test 3: Large matrix (same as benchmark) */
    print("\nTest 3: Large matrix (1024x1024) checksum comparison\n");
    total_failures += test_benchmark_correctness();

    /* Final result */
    print("\n=== Result ===\n");
    if (total_failures == 0) {
        print("PASS: All SIMD validation tests passed\n");
    } else {
        print("FAIL: %d test(s) failed\n", total_failures);
    }

    threadexits(total_failures == 0 ? 0 : "simd validation failed");
}
