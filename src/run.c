/* Inference for Llama-2 Transformer model in pure C */
/* Plan 9 port - CLI with SIMD and parallel optimization support */

#include "model.c"

// ----------------------------------------------------------------------------
// CLI

void error_usage(void) {
    fprint(2, "Usage:   run <checkpoint> [options]\n");
    fprint(2, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprint(2, "Options:\n");
    fprint(2, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprint(2, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprint(2, "  -s <int>    random seed, default time(NULL)\n");
    fprint(2, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprint(2, "  -i <string> input prompt\n");
    fprint(2, "  -z <string> optional path to custom tokenizer\n");
    fprint(2, "  -m <string> mode: generate|chat, default: generate\n");
    fprint(2, "  -y <string> (optional) system prompt in chat mode\n");
    fprint(2, "  -j <int>    number of threads (default: auto-detect, 1 = single-threaded)\n");
    fprint(2, "  --no-simd   disable SIMD optimizations (use scalar code)\n");
    exits("usage");
}

/* Check if argument matches a long option */
int match_long_opt(char *arg, char *opt) {
    return strcmp(arg, opt) == 0;
}

void
threadmain(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = nil;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = nil;
    uvlong rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = nil;

    // parse args
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i++) {
        // Handle long options first
        if (match_long_opt(argv[i], "--no-simd")) {
            opt_config.use_simd = 0;
            continue;
        }

        // Short options require a value
        if (i + 1 >= argc && argv[i][0] == '-' && strlen(argv[i]) == 2) {
            error_usage();
        }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }

        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); i++; }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); i++; }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); i++; }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); i++; }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; i++; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; i++; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; i++; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; i++; }
        else if (argv[i][1] == 'j') { opt_config.nthreads = atoi(argv[i + 1]); i++; }
        else { error_usage(); }
    }

    // parameter validation
    if (rng_seed == 0) rng_seed = nsec();
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // Initialize optimization subsystem
    opt_init();

    // Print optimization settings
    fprint(2, "Optimization: SIMD=%s, threads=%d\n",
           opt_config.use_simd ? "on" : "off",
           opt_config.nthreads);

    // build the Transformer
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    // build the Tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprint(2, "unknown mode: %s\n", mode);
        error_usage();
    }

    // cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    opt_cleanup();
    threadexits(0);
}
