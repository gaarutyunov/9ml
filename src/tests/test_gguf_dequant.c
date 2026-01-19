/*
 * test_gguf_dequant.c - Test GGUF dequantization functions
 *
 * Tests FP16->FP32 conversion and Q4_0/Q8_0 dequantization.
 * These are unit tests that don't require actual GGUF files.
 */

/* Include the GGUF implementation directly
 * Note: In VM, test files are in root with format/ subdirectory */
#include "format/gguf.c"

/* Helper: compare floats with epsilon */
static int
float_eq(float a, float b, float eps)
{
    float diff = a - b;
    if (diff < 0) diff = -diff;
    return diff < eps;
}

/* Convert FP32 to FP16 for test data generation */
static ushort
fp32_to_fp16(float f)
{
    union { uint i; float f; } u;
    u.f = f;

    uint sign = (u.i >> 31) & 1;
    int exp = ((u.i >> 23) & 0xff) - 127 + 15;
    uint mant = (u.i >> 13) & 0x3ff;

    if (exp <= 0) {
        /* Denormal or zero */
        return (ushort)(sign << 15);
    } else if (exp >= 31) {
        /* Inf */
        return (ushort)((sign << 15) | 0x7c00);
    }

    return (ushort)((sign << 15) | (exp << 10) | mant);
}

void
main(int argc, char *argv[])
{
    int passed = 0;
    int failed = 0;
    float out[QK4_0];
    int i;

    USED(argc);
    USED(argv);

    print("=== GGUF Dequantization Test ===\n");

    /* Test 1: FP16 -> FP32 conversion - positive values */
    {
        float test_vals[] = {0.0f, 1.0f, 2.0f, 0.5f, 0.25f, 10.0f, 100.0f};
        int n = sizeof(test_vals) / sizeof(test_vals[0]);
        int ok = 1;

        for (i = 0; i < n; i++) {
            ushort fp16 = fp32_to_fp16(test_vals[i]);
            float result = fp16_to_fp32(fp16);
            if (!float_eq(result, test_vals[i], 0.01f)) {
                print("  FP16 mismatch: expected %f, got %f\n", test_vals[i], result);
                ok = 0;
            }
        }

        if (ok) {
            print("Test 1 (FP16 positive values): PASS\n");
            passed++;
        } else {
            print("Test 1 (FP16 positive values): FAIL\n");
            failed++;
        }
    }

    /* Test 2: FP16 -> FP32 conversion - negative values */
    {
        float test_vals[] = {-1.0f, -2.0f, -0.5f, -10.0f};
        int n = sizeof(test_vals) / sizeof(test_vals[0]);
        int ok = 1;

        for (i = 0; i < n; i++) {
            ushort fp16 = fp32_to_fp16(test_vals[i]);
            float result = fp16_to_fp32(fp16);
            if (!float_eq(result, test_vals[i], 0.01f)) {
                print("  FP16 neg mismatch: expected %f, got %f\n", test_vals[i], result);
                ok = 0;
            }
        }

        if (ok) {
            print("Test 2 (FP16 negative values): PASS\n");
            passed++;
        } else {
            print("Test 2 (FP16 negative values): FAIL\n");
            failed++;
        }
    }

    /* Test 3: Q4_0 dequantization - zero scale */
    {
        BlockQ4_0 block;
        block.d = 0;  /* zero scale */
        for (i = 0; i < QK4_0/2; i++) {
            block.qs[i] = 0x88;  /* 8 in both nibbles -> 0 after -8 */
        }

        dequant_q4_0(out, &block);

        int ok = 1;
        for (i = 0; i < QK4_0; i++) {
            if (out[i] != 0.0f) {
                print("  Q4_0 zero: expected 0, got %f at %d\n", out[i], i);
                ok = 0;
            }
        }

        if (ok) {
            print("Test 3 (Q4_0 zero scale): PASS\n");
            passed++;
        } else {
            print("Test 3 (Q4_0 zero scale): FAIL\n");
            failed++;
        }
    }

    /* Test 4: Q4_0 dequantization - unit scale, varying values */
    {
        BlockQ4_0 block;
        block.d = fp32_to_fp16(1.0f);  /* scale = 1.0 */

        /* Fill with pattern: low nibble=0xF (15-8=7), high nibble=0x0 (0-8=-8) */
        for (i = 0; i < QK4_0/2; i++) {
            block.qs[i] = 0x0F;  /* low=15->7, high=0->-8 */
        }

        dequant_q4_0(out, &block);

        int ok = 1;
        /* First 16 values should be 7*1.0 = 7.0 */
        for (i = 0; i < QK4_0/2; i++) {
            if (!float_eq(out[i], 7.0f, 0.01f)) {
                print("  Q4_0 first half: expected 7.0, got %f at %d\n", out[i], i);
                ok = 0;
            }
        }
        /* Second 16 values should be -8*1.0 = -8.0 */
        for (i = QK4_0/2; i < QK4_0; i++) {
            if (!float_eq(out[i], -8.0f, 0.01f)) {
                print("  Q4_0 second half: expected -8.0, got %f at %d\n", out[i], i);
                ok = 0;
            }
        }

        if (ok) {
            print("Test 4 (Q4_0 unit scale): PASS\n");
            passed++;
        } else {
            print("Test 4 (Q4_0 unit scale): FAIL\n");
            failed++;
        }
    }

    /* Test 5: Q4_0 dequantization - non-unit scale */
    {
        BlockQ4_0 block;
        block.d = fp32_to_fp16(0.5f);  /* scale = 0.5 */

        /* All values = 8 (0x88) -> 0 after subtracting 8 */
        for (i = 0; i < QK4_0/2; i++) {
            block.qs[i] = 0x88;
        }

        dequant_q4_0(out, &block);

        int ok = 1;
        for (i = 0; i < QK4_0; i++) {
            if (!float_eq(out[i], 0.0f, 0.01f)) {
                print("  Q4_0 scale 0.5: expected 0.0, got %f at %d\n", out[i], i);
                ok = 0;
            }
        }

        if (ok) {
            print("Test 5 (Q4_0 non-unit scale): PASS\n");
            passed++;
        } else {
            print("Test 5 (Q4_0 non-unit scale): FAIL\n");
            failed++;
        }
    }

    /* Test 6: Q8_0 dequantization - zero scale */
    {
        BlockQ8_0 block;
        block.d = 0;  /* zero scale */
        for (i = 0; i < QK8_0; i++) {
            block.qs[i] = (schar)i;
        }

        dequant_q8_0(out, &block);

        int ok = 1;
        for (i = 0; i < QK8_0; i++) {
            if (out[i] != 0.0f) {
                print("  Q8_0 zero: expected 0, got %f at %d\n", out[i], i);
                ok = 0;
            }
        }

        if (ok) {
            print("Test 6 (Q8_0 zero scale): PASS\n");
            passed++;
        } else {
            print("Test 6 (Q8_0 zero scale): FAIL\n");
            failed++;
        }
    }

    /* Test 7: Q8_0 dequantization - unit scale */
    {
        BlockQ8_0 block;
        block.d = fp32_to_fp16(1.0f);  /* scale = 1.0 */
        for (i = 0; i < QK8_0; i++) {
            block.qs[i] = (schar)(i - 16);  /* -16 to 15 */
        }

        dequant_q8_0(out, &block);

        int ok = 1;
        for (i = 0; i < QK8_0; i++) {
            float expected = (float)(i - 16);
            if (!float_eq(out[i], expected, 0.01f)) {
                print("  Q8_0 unit: expected %f, got %f at %d\n", expected, out[i], i);
                ok = 0;
            }
        }

        if (ok) {
            print("Test 7 (Q8_0 unit scale): PASS\n");
            passed++;
        } else {
            print("Test 7 (Q8_0 unit scale): FAIL\n");
            failed++;
        }
    }

    /* Test 8: Q8_0 dequantization - varying scale */
    {
        BlockQ8_0 block;
        block.d = fp32_to_fp16(2.0f);  /* scale = 2.0 */
        for (i = 0; i < QK8_0; i++) {
            block.qs[i] = (schar)10;  /* all 10s */
        }

        dequant_q8_0(out, &block);

        int ok = 1;
        for (i = 0; i < QK8_0; i++) {
            float expected = 20.0f;  /* 10 * 2.0 */
            if (!float_eq(out[i], expected, 0.1f)) {
                print("  Q8_0 scale 2: expected %f, got %f at %d\n", expected, out[i], i);
                ok = 0;
            }
        }

        if (ok) {
            print("Test 8 (Q8_0 varying scale): PASS\n");
            passed++;
        } else {
            print("Test 8 (Q8_0 varying scale): FAIL\n");
            failed++;
        }
    }

    /* Test 9: Q8_0 negative values */
    {
        BlockQ8_0 block;
        block.d = fp32_to_fp16(1.0f);
        for (i = 0; i < QK8_0; i++) {
            block.qs[i] = (schar)-127;  /* max negative */
        }

        dequant_q8_0(out, &block);

        int ok = 1;
        for (i = 0; i < QK8_0; i++) {
            float expected = -127.0f;
            if (!float_eq(out[i], expected, 0.01f)) {
                print("  Q8_0 negative: expected %f, got %f at %d\n", expected, out[i], i);
                ok = 0;
            }
        }

        if (ok) {
            print("Test 9 (Q8_0 negative values): PASS\n");
            passed++;
        } else {
            print("Test 9 (Q8_0 negative values): FAIL\n");
            failed++;
        }
    }

    /* Summary */
    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d GGUF dequantization tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    exits(failed ? "fail" : 0);
}
