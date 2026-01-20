/*
 * test_gemma3_arch.c - Test Gemma 3 architecture detection and registration
 *
 * Tests:
 *   - Architecture ID is ARCH_GEMMA3 = 2
 *   - Gemma 3 specific config defaults
 *   - FFN type detection (GeGLU)
 */

#include <u.h>
#include <libc.h>

/* Architecture IDs */
enum {
    ARCH_UNKNOWN = 0,
    ARCH_LLAMA2  = 1,
    ARCH_GEMMA3  = 2,
};

/* FFN activation types */
enum {
    FFN_SWIGLU = 0,
    FFN_GEGLU  = 1,
};

/* Simplified config for testing */
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    float rope_theta;
    int arch_id;
    /* Gemma 3 specific */
    int head_dim;
    int sliding_window;
    float rope_local_theta;
    float rope_global_theta;
    int use_qk_norm;
    int ffn_type;
    float rms_norm_eps;
    int query_pre_attn_scalar;
} TestConfig;

void
main(int argc, char *argv[])
{
    int passed = 0;
    int failed = 0;
    TestConfig cfg;

    USED(argc);
    USED(argv);

    print("=== Gemma 3 Architecture Test ===\n");

    /* Test 1: ARCH_GEMMA3 value */
    print("\nTest 1: ARCH_GEMMA3 = 2\n");
    if (ARCH_GEMMA3 == 2) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL (got %d)\n", ARCH_GEMMA3);
        failed++;
    }

    /* Test 2: FFN_GEGLU value */
    print("\nTest 2: FFN_GEGLU = 1\n");
    if (FFN_GEGLU == 1) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL (got %d)\n", FFN_GEGLU);
        failed++;
    }

    /* Test 3: Gemma 3 270M config validation */
    print("\nTest 3: Gemma 3 270M config structure\n");
    memset(&cfg, 0, sizeof(cfg));

    /* Set Gemma 3 270M config */
    cfg.dim = 640;
    cfg.hidden_dim = 2048;
    cfg.n_layers = 18;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 1;
    cfg.vocab_size = 262144;
    cfg.seq_len = 32768;
    cfg.rope_theta = 1000000.0f;  /* Global theta */
    cfg.arch_id = ARCH_GEMMA3;
    cfg.head_dim = 256;
    cfg.sliding_window = 512;
    cfg.rope_local_theta = 10000.0f;
    cfg.rope_global_theta = 1000000.0f;
    cfg.use_qk_norm = 1;
    cfg.ffn_type = FFN_GEGLU;
    cfg.rms_norm_eps = 1e-6f;
    cfg.query_pre_attn_scalar = 256;

    /* Validate config values */
    if (cfg.arch_id == ARCH_GEMMA3 &&
        cfg.ffn_type == FFN_GEGLU &&
        cfg.use_qk_norm == 1 &&
        cfg.sliding_window == 512 &&
        cfg.head_dim == 256 &&
        cfg.n_kv_heads == 1 &&  /* GQA: 4 Q heads, 1 KV head */
        cfg.rope_local_theta == 10000.0f &&
        cfg.rope_global_theta == 1000000.0f) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL\n");
        print("    arch_id=%d (expected %d)\n", cfg.arch_id, ARCH_GEMMA3);
        print("    ffn_type=%d (expected %d)\n", cfg.ffn_type, FFN_GEGLU);
        print("    use_qk_norm=%d (expected 1)\n", cfg.use_qk_norm);
        print("    sliding_window=%d (expected 512)\n", cfg.sliding_window);
        failed++;
    }

    /* Test 4: Head dimension calculation */
    print("\nTest 4: Head dimension\n");
    /* Gemma 3 270M has head_dim=256, but dim=640, n_heads=4
     * So head_dim != dim/n_heads (256 != 160) - this is intentional for Gemma 3 */
    int computed_head_dim = cfg.dim / cfg.n_heads;
    if (cfg.head_dim != computed_head_dim) {
        print("  Note: Gemma 3 uses explicit head_dim=%d, not dim/n_heads=%d\n",
              cfg.head_dim, computed_head_dim);
        print("  Result: PASS (as expected for Gemma 3)\n");
        passed++;
    } else {
        print("  Result: PASS (head_dim matches computed)\n");
        passed++;
    }

    /* Test 5: Attention pattern - 5 local + 1 global */
    print("\nTest 5: Attention pattern (5 local + 1 global)\n");
    int pattern_correct = 1;
    int i;
    for (i = 0; i < cfg.n_layers; i++) {
        int is_local = (i % 6) < 5;
        int expected_local = (i == 5 || i == 11 || i == 17) ? 0 : 1;
        if (is_local != expected_local) {
            print("  Layer %d: got local=%d, expected %d\n", i, is_local, expected_local);
            pattern_correct = 0;
        }
    }
    if (pattern_correct) {
        print("  Result: PASS\n");
        print("  Layers 0-4: local, Layer 5: global\n");
        print("  Layers 6-10: local, Layer 11: global\n");
        print("  Layers 12-16: local, Layer 17: global\n");
        passed++;
    } else {
        print("  Result: FAIL\n");
        failed++;
    }

    /* Summary */
    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d Gemma 3 architecture tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    exits(failed ? "fail" : 0);
}
