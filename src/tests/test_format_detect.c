/*
 * test_format_detect.c - Test 9ml format detection
 *
 * Tests:
 * 1. Legacy format detection
 * 2. Extended format detection
 * 3. Architecture info extraction
 * 4. RoPE theta extraction
 *
 * Only ARCH_LLAMA2 is supported.
 */

#include <u.h>
#include <libc.h>

/* Format constants (must match model.c) */
#define FORMAT_MAGIC_9ML_V1   0x394D4C01
#define LEGACY_CONFIG_SIZE    28
#define EXTENDED_HEADER_SIZE  60

#define ARCH_UNKNOWN  0
#define ARCH_LLAMA2   1

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
} Config;

/* Detect format and extract config */
int detect_format(uchar *data, vlong size, Config *cfg, int *is_extended) {
    uint magic;

    if (size < 4) return 0;

    magic = *(uint*)data;

    if (magic == FORMAT_MAGIC_9ML_V1) {
        /* Extended format */
        if (size < EXTENDED_HEADER_SIZE) return 0;

        int *ext_buf = (int*)(data + 4);
        cfg->arch_id = ext_buf[1];

        /* rope_theta stored as float bits */
        union { int i; float f; } theta_conv;
        theta_conv.i = ext_buf[2];
        cfg->rope_theta = theta_conv.f;

        cfg->dim = ext_buf[7];
        cfg->hidden_dim = ext_buf[8];
        cfg->n_layers = ext_buf[9];
        cfg->n_heads = ext_buf[10];
        cfg->n_kv_heads = ext_buf[11];
        cfg->vocab_size = ext_buf[12];
        cfg->seq_len = ext_buf[13];

        if (cfg->vocab_size < 0) cfg->vocab_size = -cfg->vocab_size;

        *is_extended = 1;
        return 1;
    } else {
        /* Legacy format - magic is actually dim */
        if (size < LEGACY_CONFIG_SIZE) return 0;

        int *config_ptr = (int*)data;
        cfg->dim = config_ptr[0];
        cfg->hidden_dim = config_ptr[1];
        cfg->n_layers = config_ptr[2];
        cfg->n_heads = config_ptr[3];
        cfg->n_kv_heads = config_ptr[4];
        cfg->vocab_size = config_ptr[5];
        cfg->seq_len = config_ptr[6];

        if (cfg->vocab_size < 0) cfg->vocab_size = -cfg->vocab_size;

        /* Default to LLaMA 2 */
        cfg->rope_theta = 10000.0f;
        cfg->arch_id = ARCH_LLAMA2;

        *is_extended = 0;
        return 1;
    }
}

/* Create a fake legacy header */
void create_legacy_header(uchar *buf, int dim, int vocab_size) {
    int *header = (int*)buf;
    header[0] = dim;
    header[1] = dim * 4;  /* hidden_dim */
    header[2] = 6;        /* n_layers */
    header[3] = 6;        /* n_heads */
    header[4] = 6;        /* n_kv_heads */
    header[5] = vocab_size;
    header[6] = 256;      /* seq_len */
}

/* Create a fake extended header */
void create_extended_header(uchar *buf, int dim, int vocab_size, int arch_id, float rope_theta) {
    int *header = (int*)buf;
    union { float f; int i; } theta_conv;

    header[0] = FORMAT_MAGIC_9ML_V1;
    header[1] = EXTENDED_HEADER_SIZE;
    header[2] = arch_id;

    theta_conv.f = rope_theta;
    header[3] = theta_conv.i;

    header[4] = 0;   /* ffn_type */
    header[5] = 0;   /* flags */
    header[6] = 0;   /* sliding_window */
    header[7] = 0;   /* reserved */

    header[8] = dim;
    header[9] = dim * 4;
    header[10] = 6;
    header[11] = 6;
    header[12] = 6;
    header[13] = vocab_size;
    header[14] = 256;
}

void
main(int argc, char *argv[])
{
    uchar buf[128];
    Config cfg;
    int is_extended;
    int passed = 0;
    int failed = 0;

    USED(argc);
    USED(argv);

    print("=== Format Detection Test ===\n");

    /* Test 1: Legacy LLaMA 2 format */
    print("\nTest 1: Legacy LLaMA 2 format\n");
    create_legacy_header(buf, 288, 32000);

    if (detect_format(buf, 64, &cfg, &is_extended)) {
        if (!is_extended && cfg.arch_id == ARCH_LLAMA2 &&
            cfg.rope_theta == 10000.0f && cfg.dim == 288) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL (wrong values: extended=%d arch=%d theta=%f)\n",
                  is_extended, cfg.arch_id, cfg.rope_theta);
            failed++;
        }
    } else {
        print("  Result: FAIL (detection failed)\n");
        failed++;
    }

    /* Test 2: Legacy format with large vocab still uses LLaMA 2 */
    print("\nTest 2: Legacy format defaults to LLaMA 2\n");
    create_legacy_header(buf, 512, 128256);

    if (detect_format(buf, 64, &cfg, &is_extended)) {
        if (!is_extended && cfg.arch_id == ARCH_LLAMA2 &&
            cfg.rope_theta == 10000.0f) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL (wrong values: arch=%d theta=%f)\n",
                  cfg.arch_id, cfg.rope_theta);
            failed++;
        }
    } else {
        print("  Result: FAIL (detection failed)\n");
        failed++;
    }

    /* Test 3: Extended LLaMA 2 format */
    print("\nTest 3: Extended LLaMA 2 format\n");
    create_extended_header(buf, 288, 32000, ARCH_LLAMA2, 10000.0f);

    if (detect_format(buf, 128, &cfg, &is_extended)) {
        if (is_extended && cfg.arch_id == ARCH_LLAMA2 &&
            cfg.rope_theta == 10000.0f && cfg.dim == 288) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL (wrong values: extended=%d arch=%d theta=%f)\n",
                  is_extended, cfg.arch_id, cfg.rope_theta);
            failed++;
        }
    } else {
        print("  Result: FAIL (detection failed)\n");
        failed++;
    }

    /* Test 4: Custom rope_theta in extended format */
    print("\nTest 4: Custom rope_theta (250000.0)\n");
    create_extended_header(buf, 512, 50000, ARCH_LLAMA2, 250000.0f);

    if (detect_format(buf, 128, &cfg, &is_extended)) {
        float diff = cfg.rope_theta - 250000.0f;
        if (diff < 0) diff = -diff;
        if (is_extended && diff < 1.0f) {
            print("  Result: PASS (theta=%f)\n", cfg.rope_theta);
            passed++;
        } else {
            print("  Result: FAIL (theta=%f, expected 250000)\n", cfg.rope_theta);
            failed++;
        }
    } else {
        print("  Result: FAIL (detection failed)\n");
        failed++;
    }

    /* Summary */
    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d format detection tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    exits(failed ? "fail" : 0);
}
