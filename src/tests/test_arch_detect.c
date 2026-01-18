/*
 * test_arch_detect.c - Test architecture auto-detection
 *
 * Tests that the architecture registry and detection work correctly.
 */

#include <u.h>
#include <libc.h>

/* Simplified architecture detection test
 * Tests the detection logic without full model loading
 */

/* Architecture IDs */
enum {
    ARCH_UNKNOWN = 0,
    ARCH_LLAMA2  = 1,
    ARCH_LLAMA3  = 2,
    ARCH_MISTRAL = 3,
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

/* Detect architecture from header */
int detect_arch(int *header, TestConfig *cfg) {
    int dim, vocab_size;

    dim = header[0];
    vocab_size = header[5];
    if (vocab_size < 0) vocab_size = -vocab_size;

    /* Sanity checks */
    if (dim < 64 || dim > 8192) return ARCH_UNKNOWN;

    /* LLaMA 3: large vocabulary (128K+) */
    if (vocab_size >= 100000) {
        cfg->rope_theta = 500000.0f;
        cfg->arch_id = ARCH_LLAMA3;
        return ARCH_LLAMA3;
    }

    /* Default: LLaMA 2 */
    cfg->rope_theta = 10000.0f;
    cfg->arch_id = ARCH_LLAMA2;
    return ARCH_LLAMA2;
}

void
main(int argc, char *argv[])
{
    TestConfig cfg;
    int header[7];
    int arch;
    int passed = 0;
    int failed = 0;

    USED(argc);
    USED(argv);

    print("=== Architecture Detection Test ===\n");

    /* Test 1: LLaMA 2 format (small vocab) */
    header[0] = 288;      /* dim */
    header[1] = 768;      /* hidden_dim */
    header[2] = 6;        /* n_layers */
    header[3] = 6;        /* n_heads */
    header[4] = 6;        /* n_kv_heads */
    header[5] = 32000;    /* vocab_size - typical LLaMA 2 */
    header[6] = 256;      /* seq_len */

    memset(&cfg, 0, sizeof(cfg));
    arch = detect_arch(header, &cfg);

    if (arch == ARCH_LLAMA2 && cfg.rope_theta == 10000.0f) {
        print("Test 1 (LLaMA 2 detection): PASS\n");
        passed++;
    } else {
        print("Test 1 (LLaMA 2 detection): FAIL (arch=%d, theta=%f)\n",
              arch, cfg.rope_theta);
        failed++;
    }

    /* Test 2: LLaMA 3 format (large vocab) */
    header[5] = 128256;   /* vocab_size - typical LLaMA 3 */

    memset(&cfg, 0, sizeof(cfg));
    arch = detect_arch(header, &cfg);

    if (arch == ARCH_LLAMA3 && cfg.rope_theta == 500000.0f) {
        print("Test 2 (LLaMA 3 detection): PASS\n");
        passed++;
    } else {
        print("Test 2 (LLaMA 3 detection): FAIL (arch=%d, theta=%f)\n",
              arch, cfg.rope_theta);
        failed++;
    }

    /* Test 3: Negative vocab_size (shared weights indicator) */
    header[5] = -32000;   /* negative = shared weights */

    memset(&cfg, 0, sizeof(cfg));
    arch = detect_arch(header, &cfg);

    if (arch == ARCH_LLAMA2) {
        print("Test 3 (negative vocab_size): PASS\n");
        passed++;
    } else {
        print("Test 3 (negative vocab_size): FAIL (arch=%d)\n", arch);
        failed++;
    }

    /* Test 4: Invalid header (dim too small) */
    header[0] = 32;       /* dim too small */
    header[5] = 32000;

    memset(&cfg, 0, sizeof(cfg));
    arch = detect_arch(header, &cfg);

    if (arch == ARCH_UNKNOWN) {
        print("Test 4 (invalid header): PASS\n");
        passed++;
    } else {
        print("Test 4 (invalid header): FAIL (arch=%d)\n", arch);
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
