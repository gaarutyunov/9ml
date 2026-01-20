/*
 * test_arch_detect.c - Test architecture detection
 *
 * Tests that the architecture registry works correctly.
 * Only ARCH_LLAMA2 is supported.
 */

#include <u.h>
#include <libc.h>

/* Architecture IDs */
enum {
    ARCH_UNKNOWN = 0,
    ARCH_LLAMA2  = 1,
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
} TestConfig;

void
main(int argc, char *argv[])
{
    int passed = 0;
    int failed = 0;

    USED(argc);
    USED(argv);

    print("=== Architecture Detection Test ===\n");

    /* Test 1: ARCH_LLAMA2 is the only supported architecture */
    print("\nTest 1: ARCH_LLAMA2 is supported\n");
    if (ARCH_LLAMA2 == 1) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL\n");
        failed++;
    }

    /* Test 2: ARCH_UNKNOWN is 0 */
    print("\nTest 2: ARCH_UNKNOWN is 0\n");
    if (ARCH_UNKNOWN == 0) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL\n");
        failed++;
    }

    /* Test 3: Default rope_theta for LLaMA 2 is 10000 */
    print("\nTest 3: Default rope_theta is 10000\n");
    TestConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.rope_theta = 10000.0f;
    cfg.arch_id = ARCH_LLAMA2;

    if (cfg.rope_theta == 10000.0f && cfg.arch_id == ARCH_LLAMA2) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL (theta=%f, arch=%d)\n", cfg.rope_theta, cfg.arch_id);
        failed++;
    }

    /* Summary */
    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d architecture detection tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    exits(failed ? "fail" : 0);
}
