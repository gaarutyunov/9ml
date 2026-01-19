/*
 * test_format_generation.c - Compare text generation across model formats
 *
 * Tests that safetensors (.safetensors) and GGUF (.gguf) model formats
 * produce identical generation output with deterministic settings.
 *
 * This test requires:
 *   - stories15M.safetensors (safetensors format)
 *   - stories15M-Q8_0.gguf (GGUF Q8_0 format)
 *   - tokenizer.bin
 */

/* Disable threading to avoid libthread conflicts - we use single-threaded inference */
#define DISABLE_OPTIMIZATIONS
#include "model.c"

/* Model file paths - these will be on the shared disk */
#define SAFETENSORS_MODEL "/mnt/host/stories15M.safetensors"
#define GGUF_MODEL        "/mnt/host/stories15M-Q8_0.gguf"
#define TOKENIZER_PATH    "/mnt/host/tokenizer.bin"

/* Test parameters for deterministic generation */
#define TEST_STEPS  20
#define TEST_SEED   42
#define TEST_TEMP   0.0f  /* Greedy sampling */
#define TEST_TOPP   0.9f

/* Maximum output buffer size */
#define MAX_OUTPUT 4096

/*
 * Generate text and return as string.
 * Caller must free the returned string.
 */
static char *
generate_text(Transformer *t, Tokenizer *tok, char *prompt, int steps, float temp, uvlong seed)
{
    Sampler sampler;
    int *prompt_tokens;
    int num_prompt_tokens;
    int token, next, pos;
    char *output;
    int output_len = 0;
    char *piece;

    /* Allocate output buffer */
    output = malloc(MAX_OUTPUT);
    if (output == nil) return nil;
    output[0] = '\0';

    /* Build sampler */
    build_sampler(&sampler, t->config.vocab_size, temp, TEST_TOPP, seed);

    /* Encode prompt */
    prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    if (prompt_tokens == nil) {
        free(output);
        free_sampler(&sampler);
        return nil;
    }
    encode(tok, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        free(output);
        free(prompt_tokens);
        free_sampler(&sampler);
        return nil;
    }

    /* Generation loop */
    token = prompt_tokens[0];
    for (pos = 0; pos < steps; pos++) {
        float *logits = forward(t, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(&sampler, logits);
        }

        if (next == 1) break;  /* EOS */

        /* Decode and append */
        piece = decode(tok, token, next);
        if (piece != nil && output_len + strlen(piece) < MAX_OUTPUT - 1) {
            strcpy(output + output_len, piece);
            output_len += strlen(piece);
        }

        token = next;
    }

    free(prompt_tokens);
    free_sampler(&sampler);
    return output;
}

/*
 * Check if two strings match (for output comparison)
 */
static int
strings_match(char *a, char *b)
{
    if (a == nil || b == nil) return 0;
    return strcmp(a, b) == 0;
}

/*
 * Check if a file exists
 */
static int
file_exists(char *path)
{
    Dir *d = dirstat(path);
    if (d == nil) return 0;
    free(d);
    return 1;
}

void
main(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    Transformer t_safetensors, t_gguf;
    Tokenizer tokenizer;
    char *out_safetensors = nil;
    char *out_gguf = nil;
    char *prompt = "Once upon a time";
    int have_safetensors = 0, have_gguf = 0;

    print("=== Format Generation Comparison Test ===\n");
    print("Prompt: \"%s\"\n", prompt);
    print("Steps: %d, Temp: %.1f, Seed: %d\n\n", TEST_STEPS, TEST_TEMP, TEST_SEED);

    /* Check which model files are available */
    have_safetensors = file_exists(SAFETENSORS_MODEL);
    have_gguf = file_exists(GGUF_MODEL);

    print("Model files found:\n");
    print("  Safetensors:       %s\n", have_safetensors ? "YES" : "NO");
    print("  GGUF (Q8_0):       %s\n", have_gguf ? "YES" : "NO");
    print("\n");

    if (!have_safetensors && !have_gguf) {
        print("FAIL: No model files found\n");
        exits("nomodels");
    }

    /* Load tokenizer (same for all formats) */
    if (!file_exists(TOKENIZER_PATH)) {
        print("FAIL: Tokenizer not found: %s\n", TOKENIZER_PATH);
        exits("tokenizer");
    }

    /* We need at least one model to get vocab_size for tokenizer */
    int vocab_size = 32000;  /* Default for stories15M */

    /* Test 1: Safetensors format */
    if (have_safetensors) {
        print("Loading safetensors model...\n");
        build_transformer(&t_safetensors, SAFETENSORS_MODEL);
        vocab_size = t_safetensors.config.vocab_size;
        print("  Config: dim=%d, layers=%d, heads=%d, vocab=%d\n",
              t_safetensors.config.dim, t_safetensors.config.n_layers,
              t_safetensors.config.n_heads, t_safetensors.config.vocab_size);

        build_tokenizer(&tokenizer, TOKENIZER_PATH, vocab_size);

        print("Generating with safetensors model...\n");
        out_safetensors = generate_text(&t_safetensors, &tokenizer, prompt, TEST_STEPS, TEST_TEMP, TEST_SEED);
        if (out_safetensors) {
            print("  Output: %s\n", out_safetensors);
        } else {
            print("  ERROR: Generation failed\n");
        }
        free_transformer(&t_safetensors);
        free_tokenizer(&tokenizer);
        print("\n");
    }

    /* Test 2: GGUF format */
    if (have_gguf) {
        print("Loading GGUF model...\n");
        build_transformer(&t_gguf, GGUF_MODEL);
        vocab_size = t_gguf.config.vocab_size;
        print("  Config: dim=%d, layers=%d, heads=%d, vocab=%d\n",
              t_gguf.config.dim, t_gguf.config.n_layers,
              t_gguf.config.n_heads, t_gguf.config.vocab_size);

        build_tokenizer(&tokenizer, TOKENIZER_PATH, vocab_size);

        print("Generating with GGUF model...\n");
        out_gguf = generate_text(&t_gguf, &tokenizer, prompt, TEST_STEPS, TEST_TEMP, TEST_SEED);
        if (out_gguf) {
            print("  Output: %s\n", out_gguf);
        } else {
            print("  ERROR: Generation failed\n");
        }
        free_transformer(&t_gguf);
        free_tokenizer(&tokenizer);
        print("\n");
    }

    /* Compare outputs */
    print("=== Comparison Results ===\n");

    if (have_safetensors && have_gguf) {
        if (strings_match(out_safetensors, out_gguf)) {
            print("Safetensors vs GGUF: MATCH\n");
            print("\n=== Summary ===\n");
            print("PASS: All format generation tests passed\n");
            if (out_safetensors) free(out_safetensors);
            if (out_gguf) free(out_gguf);
            exits(0);
        } else {
            print("Safetensors vs GGUF: MISMATCH\n");
            print("  Safetensors: %s\n", out_safetensors ? out_safetensors : "(null)");
            print("  GGUF:        %s\n", out_gguf ? out_gguf : "(null)");
            print("\n=== Summary ===\n");
            print("FAIL: Format generation tests failed\n");
            if (out_safetensors) free(out_safetensors);
            if (out_gguf) free(out_gguf);
            exits("fail");
        }
    } else {
        /* Only one format available - can't compare, but we can verify it works */
        print("SKIP: Need both safetensors and GGUF for comparison\n");
        if (out_safetensors) free(out_safetensors);
        if (out_gguf) free(out_gguf);
        exits(0);
    }
}
