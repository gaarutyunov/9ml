/*
 * test_arch_llama3.c - Test LLaMA 3 specific features
 *
 * Verifies that:
 * 1. rope_theta=500000 produces different rotations than rope_theta=10000
 * 2. The RoPE implementation handles large theta correctly
 */

#include <u.h>
#include <libc.h>

/* RoPE implementation for testing */
void apply_rope(float *q, float *k, int dim, int kv_dim, int pos,
                int head_size, float rope_theta)
{
    int i, head_dim, rotn, v;
    float freq, val, fcr, fci, v0, v1;
    float *vec;

    for (i = 0; i < dim; i += 2) {
        head_dim = i % head_size;
        freq = 1.0f / pow(rope_theta, head_dim / (float)head_size);
        val = pos * freq;
        fcr = cos(val);
        fci = sin(val);

        rotn = i < kv_dim ? 2 : 1;

        for (v = 0; v < rotn; v++) {
            vec = v == 0 ? q : k;
            v0 = vec[i];
            v1 = vec[i + 1];
            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

/* Compare two float arrays */
int arrays_differ(float *a, float *b, int n, float epsilon) {
    int i;
    for (i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        if (diff < 0) diff = -diff;
        if (diff > epsilon) return 1;
    }
    return 0;
}

void
main(int argc, char *argv[])
{
    int dim = 64;
    int kv_dim = 64;
    int head_size = 64;
    int passed = 0;
    int failed = 0;
    int i;

    float q_llama2[64], k_llama2[64];
    float q_llama3[64], k_llama3[64];
    float q_orig[64], k_orig[64];

    USED(argc);
    USED(argv);

    print("=== LLaMA 3 Architecture Test ===\n");

    /* Initialize with same values */
    for (i = 0; i < dim; i++) {
        q_orig[i] = (float)(i % 10) / 10.0f;
        k_orig[i] = (float)((i + 5) % 10) / 10.0f;
    }

    /* Test 1: Different theta produces different results */
    print("\nTest 1: RoPE theta difference\n");

    /* Apply LLaMA 2 RoPE (theta=10000) */
    memcpy(q_llama2, q_orig, dim * sizeof(float));
    memcpy(k_llama2, k_orig, dim * sizeof(float));
    apply_rope(q_llama2, k_llama2, dim, kv_dim, 100, head_size, 10000.0f);

    /* Apply LLaMA 3 RoPE (theta=500000) */
    memcpy(q_llama3, q_orig, dim * sizeof(float));
    memcpy(k_llama3, k_orig, dim * sizeof(float));
    apply_rope(q_llama3, k_llama3, dim, kv_dim, 100, head_size, 500000.0f);

    print("  LLaMA 2 (theta=10000) q[0..3]: %f %f %f %f\n",
          q_llama2[0], q_llama2[1], q_llama2[2], q_llama2[3]);
    print("  LLaMA 3 (theta=500000) q[0..3]: %f %f %f %f\n",
          q_llama3[0], q_llama3[1], q_llama3[2], q_llama3[3]);

    if (arrays_differ(q_llama2, q_llama3, dim, 0.0001f)) {
        print("  Result: PASS (different theta produces different rotations)\n");
        passed++;
    } else {
        print("  Result: FAIL (rotations should differ)\n");
        failed++;
    }

    /* Test 2: LLaMA 3 rotates less at same position (higher theta = slower rotation) */
    print("\nTest 2: Higher theta means slower rotation\n");

    /* At position 100, LLaMA 3 should rotate less because freq is lower */
    /* freq = 1/theta^(head_dim/head_size), so higher theta = lower freq */

    /* Check that LLaMA 3 values are closer to original (less rotation) */
    float llama2_diff = 0, llama3_diff = 0;
    for (i = 0; i < dim; i++) {
        float d2 = q_llama2[i] - q_orig[i];
        float d3 = q_llama3[i] - q_orig[i];
        llama2_diff += d2 * d2;
        llama3_diff += d3 * d3;
    }
    llama2_diff = sqrt(llama2_diff);
    llama3_diff = sqrt(llama3_diff);

    print("  LLaMA 2 total rotation magnitude: %f\n", llama2_diff);
    print("  LLaMA 3 total rotation magnitude: %f\n", llama3_diff);

    if (llama3_diff < llama2_diff) {
        print("  Result: PASS (LLaMA 3 rotates less due to higher theta)\n");
        passed++;
    } else {
        print("  Result: FAIL (expected LLaMA 3 to rotate less)\n");
        failed++;
    }

    /* Test 3: At position 0, both should be unchanged (no rotation) */
    print("\nTest 3: Position 0 has no rotation\n");

    memcpy(q_llama2, q_orig, dim * sizeof(float));
    memcpy(k_llama2, k_orig, dim * sizeof(float));
    apply_rope(q_llama2, k_llama2, dim, kv_dim, 0, head_size, 10000.0f);

    memcpy(q_llama3, q_orig, dim * sizeof(float));
    memcpy(k_llama3, k_orig, dim * sizeof(float));
    apply_rope(q_llama3, k_llama3, dim, kv_dim, 0, head_size, 500000.0f);

    int llama2_unchanged = !arrays_differ(q_llama2, q_orig, dim, 0.0001f);
    int llama3_unchanged = !arrays_differ(q_llama3, q_orig, dim, 0.0001f);

    if (llama2_unchanged && llama3_unchanged) {
        print("  Result: PASS (position 0 = no rotation for both)\n");
        passed++;
    } else {
        print("  Result: FAIL (position 0 should have no rotation)\n");
        failed++;
    }

    /* Test 4: Large position test (LLaMA 3's extended context) */
    print("\nTest 4: Large position (extended context)\n");

    int large_pos = 8192;  /* Beyond typical LLaMA 2 context */

    memcpy(q_llama2, q_orig, dim * sizeof(float));
    memcpy(k_llama2, k_orig, dim * sizeof(float));
    apply_rope(q_llama2, k_llama2, dim, kv_dim, large_pos, head_size, 10000.0f);

    memcpy(q_llama3, q_orig, dim * sizeof(float));
    memcpy(k_llama3, k_orig, dim * sizeof(float));
    apply_rope(q_llama3, k_llama3, dim, kv_dim, large_pos, head_size, 500000.0f);

    /* LLaMA 3 should still produce reasonable values at large positions */
    int llama3_valid = 1;
    for (i = 0; i < dim; i++) {
        if (q_llama3[i] != q_llama3[i]) {  /* NaN check */
            llama3_valid = 0;
            break;
        }
    }

    print("  Position %d test:\n", large_pos);
    print("  LLaMA 3 q[0..3]: %f %f %f %f\n",
          q_llama3[0], q_llama3[1], q_llama3[2], q_llama3[3]);

    if (llama3_valid) {
        print("  Result: PASS (LLaMA 3 handles large positions)\n");
        passed++;
    } else {
        print("  Result: FAIL (invalid values at large position)\n");
        failed++;
    }

    /* Summary */
    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d LLaMA 3 tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    exits(failed ? "fail" : 0);
}
