/*
 * test_gguf_parse.c - Test GGUF file parsing with real model files
 *
 * Tests the full GGUF parser using actual GGUF model files.
 * Requires: tinyllama-15M-stories-Q4_0.gguf or tinyllama-15M-stories-Q8_0.gguf
 */

/* Include the GGUF implementation directly
 * Note: In VM, test files are in root with format/ subdirectory */
#include "format/gguf.c"

void
main(int argc, char *argv[])
{
    GGUFFile gf;
    GGUFModelConfig cfg;
    GGUFMetadata *meta;
    GGUFTensorInfo *tensor;
    char *model_path;
    int passed = 0;
    int failed = 0;

    USED(argc);
    USED(argv);

    print("=== GGUF Parse Test ===\n");

    /* Try Q4_0 model first, fall back to Q8_0 */
    model_path = "tinyllama-15M-stories-Q4_0.gguf";
    if (access(model_path, AREAD) < 0) {
        model_path = "tinyllama-15M-stories-Q8_0.gguf";
        if (access(model_path, AREAD) < 0) {
            print("SKIP: No GGUF model file found\n");
            print("Expected: tinyllama-15M-stories-Q4_0.gguf or tinyllama-15M-stories-Q8_0.gguf\n");
            exits("skip");
        }
    }
    print("Using model: %s\n\n", model_path);

    /* Test 1: Open GGUF file */
    print("Test 1: Open GGUF file\n");
    if (gguf_open(&gf, model_path) == 0) {
        print("  Result: PASS\n");
        print("  Magic: 0x%08ux (expected 0x%08ux)\n", gf.magic, GGUF_MAGIC);
        print("  Version: %ud\n", gf.version);
        print("  Tensors: %ulld\n", gf.n_tensors);
        print("  Metadata entries: %ulld\n", gf.n_kv);
        passed++;
    } else {
        print("  Result: FAIL (could not open file)\n");
        failed++;
        exits("fail");
    }

    /* Test 2: Verify magic and version */
    print("\nTest 2: Verify magic and version\n");
    if (gf.magic == GGUF_MAGIC && gf.version >= GGUF_VERSION_MIN && gf.version <= GGUF_VERSION_MAX) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL (magic=0x%08ux, version=%ud)\n", gf.magic, gf.version);
        failed++;
    }

    /* Test 3: Find architecture metadata */
    print("\nTest 3: Find architecture metadata\n");
    meta = gguf_find_metadata(&gf, "general.architecture");
    if (meta != nil && meta->type == GGUF_TYPE_STRING) {
        print("  Result: PASS\n");
        print("  Architecture: %s\n", meta->value.str.data);
        passed++;
    } else {
        print("  Result: FAIL (architecture metadata not found)\n");
        failed++;
    }

    /* Test 4: Extract model config */
    print("\nTest 4: Extract model config\n");
    if (gguf_get_model_config(&gf, &cfg) == 0) {
        print("  Result: PASS\n");
        print("  arch_name: %s\n", cfg.arch_name);
        print("  dim: %d\n", cfg.dim);
        print("  hidden_dim: %d\n", cfg.hidden_dim);
        print("  n_layers: %d\n", cfg.n_layers);
        print("  n_heads: %d\n", cfg.n_heads);
        print("  n_kv_heads: %d\n", cfg.n_kv_heads);
        print("  vocab_size: %d\n", cfg.vocab_size);
        print("  seq_len: %d\n", cfg.seq_len);
        print("  rope_theta: %f\n", cfg.rope_theta);
        print("  arch_id: %d\n", cfg.arch_id);
        passed++;
    } else {
        print("  Result: FAIL (could not extract config)\n");
        failed++;
    }

    /* Test 5: Verify config values for TinyLlama 15M */
    print("\nTest 5: Verify config values\n");
    {
        int ok = 1;
        /* TinyLlama 15M has: dim=288, n_layers=6, n_heads=6 */
        if (cfg.dim != 288) {
            print("  dim mismatch: expected 288, got %d\n", cfg.dim);
            ok = 0;
        }
        if (cfg.n_layers != 6) {
            print("  n_layers mismatch: expected 6, got %d\n", cfg.n_layers);
            ok = 0;
        }
        if (cfg.n_heads != 6) {
            print("  n_heads mismatch: expected 6, got %d\n", cfg.n_heads);
            ok = 0;
        }
        if (ok) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    }

    /* Test 6: Find embedding tensor */
    print("\nTest 6: Find embedding tensor\n");
    tensor = gguf_find_tensor(&gf, "token_embd.weight");
    if (tensor != nil) {
        print("  Result: PASS\n");
        print("  Name: %s\n", tensor->name.data);
        print("  Dims: %d [", tensor->n_dims);
        for (uint i = 0; i < tensor->n_dims; i++) {
            print("%ulld", tensor->dims[i]);
            if (i < tensor->n_dims - 1) print(", ");
        }
        print("]\n");
        print("  Type: %d\n", tensor->type);
        print("  Offset: %ulld\n", tensor->offset);
        passed++;
    } else {
        print("  Result: FAIL (tensor not found)\n");
        failed++;
    }

    /* Test 7: List some tensor names */
    print("\nTest 7: List tensors\n");
    print("  Found %ulld tensors:\n", gf.n_tensors);
    {
        uvlong max_show = 10;
        if (gf.n_tensors < max_show) max_show = gf.n_tensors;
        for (uvlong i = 0; i < max_show; i++) {
            print("    [%ulld] %s (type=%d, dims=%d)\n",
                  i, gf.tensors[i].name.data, gf.tensors[i].type, gf.tensors[i].n_dims);
        }
        if (gf.n_tensors > max_show) {
            print("    ... and %ulld more\n", gf.n_tensors - max_show);
        }
        passed++;  /* Always pass if we get here */
    }

    /* Test 8: Dequantize first block of embedding tensor */
    print("\nTest 8: Dequantize embedding tensor (first 32 floats)\n");
    if (tensor != nil) {
        float out[32];
        vlong nfloats = gguf_dequant_tensor(&gf, tensor, out, 32);
        if (nfloats > 0) {
            print("  Result: PASS (dequantized %lld floats)\n", nfloats);
            print("  First 8 values: ");
            for (int i = 0; i < 8 && i < nfloats; i++) {
                print("%.4f ", out[i]);
            }
            print("\n");
            passed++;
        } else {
            print("  Result: FAIL (dequantization failed)\n");
            failed++;
        }
    } else {
        print("  Result: SKIP (no tensor)\n");
    }

    /* Test 9: Find attention weight tensor */
    print("\nTest 9: Find attention Q weight tensor\n");
    tensor = gguf_find_tensor(&gf, "blk.0.attn_q.weight");
    if (tensor != nil) {
        print("  Result: PASS\n");
        print("  Name: %s\n", tensor->name.data);
        print("  Dims: [");
        for (uint i = 0; i < tensor->n_dims; i++) {
            print("%ulld", tensor->dims[i]);
            if (i < tensor->n_dims - 1) print(", ");
        }
        print("]\n");
        passed++;
    } else {
        print("  Result: FAIL (tensor not found)\n");
        failed++;
    }

    /* Test 10: Get various metadata types */
    print("\nTest 10: Get metadata values\n");
    {
        int ok = 1;

        /* Get string */
        char *arch = gguf_get_string(&gf, "general.architecture");
        if (arch != nil) {
            print("  general.architecture: %s\n", arch);
            free(arch);
        } else {
            print("  general.architecture: NOT FOUND\n");
            ok = 0;
        }

        /* Get int (block_count = n_layers) */
        int n_layers = gguf_get_int(&gf, "llama.block_count", -1);
        print("  llama.block_count: %d\n", n_layers);

        /* Get float (rope_freq_base = rope_theta) */
        float rope = gguf_get_float(&gf, "llama.rope.freq_base", -1.0f);
        print("  llama.rope.freq_base: %f\n", rope);

        if (ok) {
            print("  Result: PASS\n");
            passed++;
        } else {
            print("  Result: FAIL\n");
            failed++;
        }
    }

    /* Close file */
    gguf_close(&gf);

    /* Summary */
    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d GGUF parse tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    exits(failed ? "fail" : 0);
}
