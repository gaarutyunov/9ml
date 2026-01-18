/* Softmax accuracy validation test - verify all modes produce correct results */
#include "model.c"

/* Tolerance levels per mode */
#define EPSILON_EXACT  1e-5f   /* For partial SIMD (uses real exp) */
#define EPSILON_APPROX 5e-3f   /* For polynomial and LUT approximations */
#define EPSILON_SCHRAUDOLPH 1.5e-2f  /* Schraudolph has ~1-2% error at extremes */

static int total_failures = 0;

/* Test a single mode with various sizes and patterns */
static void
test_mode(int mode, char *name, float tolerance)
{
    int sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024, 32000};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int mode_failures = 0;

    print("Testing mode %d (%s) with tolerance %.2e\n", mode, name, tolerance);

    for (int s = 0; s < nsizes; s++) {
        int size = sizes[s];
        float *x_scalar = malloc(size * sizeof(float));
        float *x_test = malloc(size * sizeof(float));

        if (!x_scalar || !x_test) {
            print("  Size %d: FAIL (malloc failed)\n", size);
            mode_failures++;
            free(x_scalar);
            free(x_test);
            continue;
        }

        /* Test 4 patterns */
        int pattern_fails = 0;
        char *patterns[] = {"normal[-5,5]", "wide[-20,20]", "uniform", "one-hot"};

        for (int p = 0; p < 4; p++) {
            /* Initialize based on pattern */
            uvlong seed = 12345 + size + p * 1000;
            for (int i = 0; i < size; i++) {
                seed = seed * 1103515245 + 12345;
                switch (p) {
                case 0:  /* Normal range [-5, 5] */
                    x_scalar[i] = (float)(seed & 0xFFFF) / 65536.0f * 10.0f - 5.0f;
                    break;
                case 1:  /* Wide range [-20, 20] */
                    x_scalar[i] = (float)(seed & 0xFFFF) / 65536.0f * 40.0f - 20.0f;
                    break;
                case 2:  /* Uniform (all same value) */
                    x_scalar[i] = 1.0f;
                    break;
                case 3:  /* One-hot (one large, rest small) */
                    x_scalar[i] = (i == size/2) ? 10.0f : 0.0f;
                    break;
                }
                x_test[i] = x_scalar[i];
            }

            /* Run scalar baseline */
            opt_config.softmax_mode = 0;
            softmax(x_scalar, size);

            /* Run test mode */
            opt_config.softmax_mode = mode;
            softmax(x_test, size);

            /* Compare */
            int mismatches = 0;
            float max_diff = 0;
            int max_idx = 0;
            for (int i = 0; i < size; i++) {
                float diff = x_scalar[i] - x_test[i];
                if (diff < 0) diff = -diff;
                if (diff > tolerance) mismatches++;
                if (diff > max_diff) {
                    max_diff = diff;
                    max_idx = i;
                }
            }

            if (mismatches > 0) {
                print("  Size %d pattern %s: FAIL (%d mismatches, max %.2e at [%d])\n",
                      size, patterns[p], mismatches, max_diff, max_idx);
                print("    scalar[%d]=%.9f, test[%d]=%.9f\n",
                      max_idx, x_scalar[max_idx], max_idx, x_test[max_idx]);
                pattern_fails++;
            }

            /* Check for NaN/Inf */
            int has_nan = 0;
            for (int i = 0; i < size; i++) {
                if (x_test[i] != x_test[i] || x_test[i] > 1e30f || x_test[i] < -1e30f) {
                    if (!has_nan) {
                        print("  WARNING: NaN/Inf detected at [%d] = %.9f\n", i, x_test[i]);
                        has_nan = 1;
                    }
                }
            }

            /* Check probability sum is ~1.0 */
            float sum = 0.0f;
            for (int i = 0; i < size; i++) {
                sum += x_test[i];
            }
            float sum_diff = sum - 1.0f;
            if (sum_diff < 0) sum_diff = -sum_diff;
            if (sum_diff > 0.01f) {
                print("  Size %d pattern %s: WARNING sum=%.6f (expected 1.0)\n",
                      size, patterns[p], sum);
            }
        }

        if (pattern_fails == 0) {
            print("  Size %d: PASS (4 patterns)\n", size);
        } else {
            mode_failures += pattern_fails;
        }

        free(x_scalar);
        free(x_test);
    }

    print("%s: %s\n\n", name, mode_failures == 0 ? "PASS" : "FAIL");
    total_failures += mode_failures;
}

void
threadmain(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    print("=== Softmax Accuracy Validation Test ===\n\n");

    /* Test each mode */
    print("--- Mode 0: Scalar (baseline, should match itself) ---\n");
    test_mode(0, "Scalar", 0.0f);  /* Exact match with itself */

    print("--- Mode 1: Partial SIMD (uses real exp, should be exact) ---\n");
    test_mode(1, "Partial SIMD", EPSILON_EXACT);

    print("--- Mode 2: Schraudolph (fast approximation, ~1-2%% error at extremes) ---\n");
    test_mode(2, "Schraudolph", EPSILON_SCHRAUDOLPH);

    print("--- Mode 3: Polynomial (range reduction + polynomial) ---\n");
    test_mode(3, "Polynomial", EPSILON_APPROX);

    print("--- Mode 4: LUT (lookup table with interpolation) ---\n");
    test_mode(4, "LUT", EPSILON_APPROX);

    print("=== Summary ===\n");
    if (total_failures == 0) {
        print("PASS: All softmax accuracy tests passed\n");
    } else {
        print("FAIL: %d test(s) failed\n", total_failures);
    }

    threadexits(total_failures == 0 ? 0 : "accuracy tests failed");
}
