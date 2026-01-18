/* Test rmsnorm SIMD vs scalar */
#include "model.c"

#define EPSILON 1e-5f

void
threadmain(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    int sizes[] = {8, 16, 32, 64, 128, 256, 288, 512, 768, 1024};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int total_failures = 0;

    print("=== RMSNorm SIMD Validation Test ===\n\n");

    for (int t = 0; t < nsizes; t++) {
        int size = sizes[t];

        float *x = malloc(size * sizeof(float));
        float *weight = malloc(size * sizeof(float));
        float *out_scalar = malloc(size * sizeof(float));
        float *out_simd = malloc(size * sizeof(float));

        if (!x || !weight || !out_scalar || !out_simd) {
            print("FAIL: malloc failed for size %d\n", size);
            total_failures++;
            continue;
        }

        /* Initialize with reproducible random data */
        uvlong seed = 12345 + size;
        for (int i = 0; i < size; i++) {
            seed = seed * 1103515245 + 12345;
            x[i] = (float)(seed & 0xFFFF) / 65536.0f * 2.0f - 1.0f;  /* Range -1 to 1 */
            seed = seed * 1103515245 + 12345;
            weight[i] = (float)(seed & 0xFFFF) / 65536.0f * 2.0f;  /* Range 0 to 2 */
            out_scalar[i] = -999.0f;
            out_simd[i] = -999.0f;
        }

        /* Run scalar version */
        opt_config.use_simd = 0;
        rmsnorm(out_scalar, x, weight, size);

        /* Run SIMD version */
        opt_config.use_simd = 1;
        rmsnorm(out_simd, x, weight, size);

        /* Compare results */
        int mismatches = 0;
        float max_diff = 0.0f;
        int max_idx = 0;

        for (int i = 0; i < size; i++) {
            float diff = out_scalar[i] - out_simd[i];
            if (diff < 0) diff = -diff;
            if (diff > EPSILON) {
                mismatches++;
                if (diff > max_diff) {
                    max_diff = diff;
                    max_idx = i;
                }
            }
        }

        if (mismatches == 0) {
            print("Size %d: PASS\n", size);
        } else {
            print("Size %d: FAIL (%d mismatches, max diff %.9f at [%d])\n",
                  size, mismatches, max_diff, max_idx);
            print("  Scalar[%d] = %.9f, SIMD[%d] = %.9f\n",
                  max_idx, out_scalar[max_idx], max_idx, out_simd[max_idx]);
            total_failures++;
        }

        /* Check for NaN/Inf */
        for (int i = 0; i < size; i++) {
            if (out_simd[i] != out_simd[i] || out_simd[i] > 1e30f || out_simd[i] < -1e30f) {
                print("  WARNING: SIMD[%d] = %.9f (NaN or overflow)\n", i, out_simd[i]);
                break;
            }
        }

        free(x);
        free(weight);
        free(out_scalar);
        free(out_simd);
    }

    print("\n=== Result ===\n");
    if (total_failures == 0) {
        print("PASS: All rmsnorm SIMD tests passed\n");
    } else {
        print("FAIL: %d test(s) failed\n", total_failures);
    }

    threadexits(total_failures == 0 ? 0 : "rmsnorm simd failed");
}
