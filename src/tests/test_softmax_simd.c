/* Test softmax SIMD vs scalar */
#include "model.c"

#define EPSILON 1e-5f

void
threadmain(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    int sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024, 32000};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int total_failures = 0;

    print("=== Softmax SIMD Validation Test ===\n\n");

    for (int t = 0; t < nsizes; t++) {
        int size = sizes[t];

        float *x_scalar = malloc(size * sizeof(float));
        float *x_simd = malloc(size * sizeof(float));

        if (!x_scalar || !x_simd) {
            print("FAIL: malloc failed for size %d\n", size);
            total_failures++;
            continue;
        }

        /* Initialize with reproducible random data */
        uvlong seed = 12345 + size;
        for (int i = 0; i < size; i++) {
            seed = seed * 1103515245 + 12345;
            float val = (float)(seed & 0xFFFF) / 65536.0f * 10.0f - 5.0f;  /* Range -5 to 5 */
            x_scalar[i] = val;
            x_simd[i] = val;
        }

        /* Run scalar version */
        opt_config.use_simd = 0;
        softmax(x_scalar, size);

        /* Run SIMD version */
        opt_config.use_simd = 1;
        softmax(x_simd, size);

        /* Compare results */
        int mismatches = 0;
        float max_diff = 0.0f;
        int max_idx = 0;

        for (int i = 0; i < size; i++) {
            float diff = x_scalar[i] - x_simd[i];
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
                  max_idx, x_scalar[max_idx], max_idx, x_simd[max_idx]);
            total_failures++;
        }

        /* Check for NaN/Inf */
        for (int i = 0; i < size; i++) {
            if (x_simd[i] != x_simd[i] || x_simd[i] > 1e30f || x_simd[i] < -1e30f) {
                print("  WARNING: SIMD[%d] = %.9f (NaN or overflow)\n", i, x_simd[i]);
                break;
            }
        }

        free(x_scalar);
        free(x_simd);
    }

    print("\n=== Result ===\n");
    if (total_failures == 0) {
        print("PASS: All softmax SIMD tests passed\n");
    } else {
        print("FAIL: %d test(s) failed\n", total_failures);
    }

    threadexits(total_failures == 0 ? 0 : "softmax simd failed");
}
