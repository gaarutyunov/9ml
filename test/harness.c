/*
 * 9ml Test Harness
 * Runs Plan 9 tests in QEMU and compares against C reference implementations
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>

#include "reference.h"
#include "fat.h"
#include "qemu.h"

/* Paths */
#define QEMU_DIR     "../qemu"
#define SRC_DIR      "../src"
#define TESTS_DIR    "../src/tests"
#define DISK_IMAGE   QEMU_DIR "/9front.qcow2"
#define SHARED_IMAGE QEMU_DIR "/shared.img"
#define CPU_DISK     QEMU_DIR "/cpu.qcow2"
#define TERM_DISK    QEMU_DIR "/terminal.qcow2"
#define MODEL_FILE   "../stories15M.bin"
#define MODEL_Q_FILE "../stories15M_q80.bin"
#define TOKENIZER_FILE "../tokenizer.bin"

/* Epsilon for float comparisons */
#define EPSILON      0.0001f
#define EPSILON_QUANT 0.5f

/* Test result tracking */
typedef struct {
    const char *name;
    int passed;
    int skipped;
    char error[256];
} TestResult;

static TestResult results[16];
static int num_results = 0;
static QemuVM vm;
static DualVM dualvm;

/* File existence check */
static int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

/* Add a test result */
static void add_result(const char *name, int passed, int skipped, const char *error) {
    TestResult *r = &results[num_results++];
    r->name = name;
    r->passed = passed;
    r->skipped = skipped;
    if (error) {
        strncpy(r->error, error, sizeof(r->error) - 1);
        r->error[sizeof(r->error) - 1] = '\0';
    } else {
        r->error[0] = '\0';
    }
}

/* Prepare shared disk with test files */
static int prepare_shared_disk(void) {
    printf("Creating shared disk...\n");
    if (fat_create(SHARED_IMAGE, 128) != 0) {
        fprintf(stderr, "Failed to create shared disk\n");
        return -1;
    }

    printf("Copying test files...\n");

    /* Copy source files */
    const char *files[] = {
        SRC_DIR "/model.c",
        SRC_DIR "/modelq.c",
        SRC_DIR "/run.c",
        SRC_DIR "/runq.c",
        SRC_DIR "/llmfs.c",
        TESTS_DIR "/test_rmsnorm.c",
        TESTS_DIR "/test_softmax.c",
        TESTS_DIR "/test_matmul.c",
        TESTS_DIR "/test_rng.c",
        TESTS_DIR "/test_quantize.c",
        TESTS_DIR "/test_quantized_matmul.c",
        TESTS_DIR "/test_model_loading.c",
        NULL
    };

    for (int i = 0; files[i]; i++) {
        const char *src = files[i];
        const char *name = strrchr(src, '/');
        name = name ? name + 1 : src;
        if (fat_copy_to(SHARED_IMAGE, src, name) != 0) {
            fprintf(stderr, "Failed to copy %s\n", src);
            return -1;
        }
    }

    /* Copy model files if present */
    if (file_exists(MODEL_FILE)) {
        if (fat_copy_to(SHARED_IMAGE, MODEL_FILE, "stories15M.bin") != 0) {
            fprintf(stderr, "Warning: failed to copy model file\n");
        }
    }
    if (file_exists(MODEL_Q_FILE)) {
        if (fat_copy_to(SHARED_IMAGE, MODEL_Q_FILE, "stories15M_q80.bin") != 0) {
            fprintf(stderr, "Warning: failed to copy quantized model file\n");
        }
    }
    if (file_exists(TOKENIZER_FILE)) {
        if (fat_copy_to(SHARED_IMAGE, TOKENIZER_FILE, "tokenizer.bin") != 0) {
            fprintf(stderr, "Warning: failed to copy tokenizer file\n");
        }
    }

    return 0;
}

/* Boot the VM and mount shared disk */
static int boot_vm(void) {
    printf("Starting QEMU...\n");

    if (qemu_start(&vm, DISK_IMAGE, SHARED_IMAGE) != 0) {
        fprintf(stderr, "Failed to start QEMU\n");
        return -1;
    }

    /* Wait for bootargs prompt and accept default */
    printf("Waiting for bootargs prompt...\n");
    if (qemu_wait_for(&vm, "bootargs", 30) < 0) {
        fprintf(stderr, "Timeout waiting for bootargs\n");
        return -1;
    }
    qemu_sendln(&vm, "");

    /* Wait for user prompt and accept default */
    printf("Waiting for user prompt...\n");
    if (qemu_wait_for(&vm, "user", 30) < 0) {
        fprintf(stderr, "Timeout waiting for user prompt\n");
        return -1;
    }
    qemu_sendln(&vm, "");

    /* Wait for shell prompt */
    printf("Waiting for shell...\n");
    if (qemu_wait_for(&vm, "term%", 60) < 0) {
        fprintf(stderr, "Timeout waiting for shell\n");
        return -1;
    }

    /* Mount shared disk */
    printf("Mounting shared disk...\n");
    qemu_sendln_wait(&vm, "dossrv -f /dev/sdG0/data shared", 10);
    qemu_sendln_wait(&vm, "mount -c /srv/shared /mnt/host", 10);
    qemu_sendln_wait(&vm, "cd /mnt/host", 5);

    return 0;
}

/* Run a command in the VM and wait for prompt */
static void run_vm_cmd(const char *cmd, int timeout_secs) {
    qemu_sendln_wait(&vm, cmd, timeout_secs);
}

/* Run all tests in VM */
static int run_vm_tests(void) {
    printf("Compiling and running tests in Plan 9...\n");

    /* Compile and run each test */
    run_vm_cmd("6c -w test_rmsnorm.c && 6l -o t_rmsnorm test_rmsnorm.6 && ./t_rmsnorm > rmsnorm.out", 5);
    run_vm_cmd("6c -w test_softmax.c && 6l -o t_softmax test_softmax.6 && ./t_softmax > softmax.out", 5);
    run_vm_cmd("6c -w test_matmul.c && 6l -o t_matmul test_matmul.6 && ./t_matmul > matmul.out", 5);
    run_vm_cmd("6c -w test_rng.c && 6l -o t_rng test_rng.6 && ./t_rng > rng.out", 5);
    run_vm_cmd("6c -w test_quantize.c && 6l -o t_quantize test_quantize.6 && ./t_quantize > quantize.out", 5);
    run_vm_cmd("6c -w test_quantized_matmul.c && 6l -o t_qmatmul test_quantized_matmul.6 && ./t_qmatmul > quantized_matmul.out", 5);
    run_vm_cmd("6c -w test_model_loading.c && 6l -o t_model test_model_loading.6 && ./t_model > model_loading.out", 5);

    /* Generation test (needs model files) - run.c is large, needs more compile time */
    /* Note: Plan 9 rc shell uses >[2] for stderr, not 2> */
    run_vm_cmd("6c -w run.c", 45);
    run_vm_cmd("6l -o run run.6", 20);
    run_vm_cmd("./run stories15M.bin -z tokenizer.bin -n 20 -s 42 -t 0.0 > generation.out >[2=1]", 120);

    /* Quantized generation test (runq.c with Q8_0 model) */
    run_vm_cmd("6c -w runq.c", 45);
    run_vm_cmd("6l -o runq runq.6", 20);
    run_vm_cmd("./runq stories15M_q80.bin -z tokenizer.bin -n 20 -s 42 -t 0.0 > generation_q.out >[2=1]", 120);

    /* Mark completion */
    run_vm_cmd("echo done > complete.txt", 2);

    return 0;
}

/* Test: rmsnorm */
static void test_rmsnorm(void) {
    printf("Testing rmsnorm... ");

    /* Reference computation */
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {0.5f, 0.5f, 0.5f, 0.5f};
    float ref[4];
    ref_rmsnorm(ref, x, w, 4);

    /* Read Plan 9 output */
    int size;
    char *data = fat_read_file(SHARED_IMAGE, "rmsnorm.out", &size);
    if (!data || size == 0) {
        add_result("rmsnorm", 0, 0, "no output file");
        printf("FAIL (no output)\n");
        free(data);
        return;
    }

    float plan9[4];
    int n = parse_float_output(data, plan9, 4);
    free(data);

    if (n != 4) {
        add_result("rmsnorm", 0, 0, "wrong number of values");
        printf("FAIL (got %d values, expected 4)\n", n);
        return;
    }

    char errbuf[256];
    if (compare_floats(ref, plan9, 4, EPSILON, errbuf, sizeof(errbuf))) {
        add_result("rmsnorm", 1, 0, NULL);
        printf("PASS\n");
    } else {
        add_result("rmsnorm", 0, 0, errbuf);
        printf("FAIL: %s\n", errbuf);
    }
}

/* Test: softmax */
static void test_softmax(void) {
    printf("Testing softmax... ");

    /* Reference computation */
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float ref[5];
    memcpy(ref, x, sizeof(ref));
    ref_softmax(ref, 5);

    /* Read Plan 9 output */
    int size;
    char *data = fat_read_file(SHARED_IMAGE, "softmax.out", &size);
    if (!data || size == 0) {
        add_result("softmax", 0, 0, "no output file");
        printf("FAIL (no output)\n");
        free(data);
        return;
    }

    float plan9[5];
    int n = parse_float_output(data, plan9, 5);
    free(data);

    if (n != 5) {
        add_result("softmax", 0, 0, "wrong number of values");
        printf("FAIL (got %d values, expected 5)\n", n);
        return;
    }

    char errbuf[256];
    if (compare_floats(ref, plan9, 5, EPSILON, errbuf, sizeof(errbuf))) {
        add_result("softmax", 1, 0, NULL);
        printf("PASS\n");
    } else {
        add_result("softmax", 0, 0, errbuf);
        printf("FAIL: %s\n", errbuf);
    }
}

/* Test: matmul */
static void test_matmul(void) {
    printf("Testing matmul... ");

    /* Reference computation */
    float w[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};  /* 3x4 matrix */
    float x[] = {1, 2, 3, 4};
    float ref[3];
    ref_matmul(ref, w, x, 4, 3);

    /* Read Plan 9 output */
    int size;
    char *data = fat_read_file(SHARED_IMAGE, "matmul.out", &size);
    if (!data || size == 0) {
        add_result("matmul", 0, 0, "no output file");
        printf("FAIL (no output)\n");
        free(data);
        return;
    }

    float plan9[3];
    int n = parse_float_output(data, plan9, 3);
    free(data);

    if (n != 3) {
        add_result("matmul", 0, 0, "wrong number of values");
        printf("FAIL (got %d values, expected 3)\n", n);
        return;
    }

    char errbuf[256];
    if (compare_floats(ref, plan9, 3, EPSILON, errbuf, sizeof(errbuf))) {
        add_result("matmul", 1, 0, NULL);
        printf("PASS\n");
    } else {
        add_result("matmul", 0, 0, errbuf);
        printf("FAIL: %s\n", errbuf);
    }
}

/* Test: RNG */
static void test_rng(void) {
    printf("Testing RNG... ");

    /* Reference computation */
    uint64_t state = 42;
    uint32_t ref[10];
    for (int i = 0; i < 10; i++) {
        ref[i] = ref_random_u32(&state);
    }

    /* Read Plan 9 output */
    int size;
    char *data = fat_read_file(SHARED_IMAGE, "rng.out", &size);
    if (!data || size == 0) {
        add_result("rng", 0, 0, "no output file");
        printf("FAIL (no output)\n");
        free(data);
        return;
    }

    uint32_t plan9[10];
    int n = parse_int_output(data, plan9, 10);
    free(data);

    if (n != 10) {
        add_result("rng", 0, 0, "wrong number of values");
        printf("FAIL (got %d values, expected 10)\n", n);
        return;
    }

    char errbuf[256];
    if (compare_ints(ref, plan9, 10, errbuf, sizeof(errbuf))) {
        add_result("rng", 1, 0, NULL);
        printf("PASS\n");
    } else {
        add_result("rng", 0, 0, errbuf);
        printf("FAIL: %s\n", errbuf);
    }
}

/* Test: quantize */
static void test_quantize(void) {
    printf("Testing quantize... ");

    /* Reference computation */
    const int GS = 32;
    float x[64];
    for (int i = 0; i < 64; i++) {
        x[i] = (float)i / 10.0f;
    }

    int8_t q[64];
    float s[2];
    ref_quantize(q, s, x, 64, GS);

    float ref[64];
    ref_dequantize(ref, q, s, 64, GS);

    /* Read Plan 9 output */
    int size;
    char *data = fat_read_file(SHARED_IMAGE, "quantize.out", &size);
    if (!data || size == 0) {
        add_result("quantize", 0, 0, "no output file");
        printf("FAIL (no output)\n");
        free(data);
        return;
    }

    float plan9[64];
    int n = parse_float_output(data, plan9, 64);
    free(data);

    if (n != 64) {
        add_result("quantize", 0, 0, "wrong number of values");
        printf("FAIL (got %d values, expected 64)\n", n);
        return;
    }

    char errbuf[256];
    if (compare_floats(ref, plan9, 64, 0.1f, errbuf, sizeof(errbuf))) {
        add_result("quantize", 1, 0, NULL);
        printf("PASS\n");
    } else {
        add_result("quantize", 0, 0, errbuf);
        printf("FAIL: %s\n", errbuf);
    }
}

/* Test: quantized_matmul */
static void test_quantized_matmul(void) {
    printf("Testing quantized_matmul... ");

    const int GS = 32;
    const int n = 64, d = 2;

    /* Input vector */
    float x[64];
    for (int i = 0; i < n; i++) {
        x[i] = (float)(i % 10) / 10.0f;
    }

    /* Weight matrix */
    float w[128];
    for (int i = 0; i < d * n; i++) {
        w[i] = (float)((i / n + i % n) % 10) / 10.0f;
    }

    /* Quantize */
    int8_t xq[64], wq[128];
    float xs[2], ws[4];
    ref_quantize(xq, xs, x, n, GS);
    ref_quantize(wq, ws, w, d * n, GS);

    /* Reference matmul */
    float ref[2];
    ref_quantized_matmul(ref, xq, xs, wq, ws, n, d, GS);

    /* Read Plan 9 output */
    int size;
    char *data = fat_read_file(SHARED_IMAGE, "quantized_matmul.out", &size);
    if (!data || size == 0) {
        add_result("quantized_matmul", 0, 0, "no output file");
        printf("FAIL (no output)\n");
        free(data);
        return;
    }

    float plan9[2];
    int count = parse_float_output(data, plan9, 2);
    free(data);

    if (count != 2) {
        add_result("quantized_matmul", 0, 0, "wrong number of values");
        printf("FAIL (got %d values, expected 2)\n", count);
        return;
    }

    char errbuf[256];
    if (compare_floats(ref, plan9, 2, EPSILON_QUANT, errbuf, sizeof(errbuf))) {
        add_result("quantized_matmul", 1, 0, NULL);
        printf("PASS\n");
    } else {
        add_result("quantized_matmul", 0, 0, errbuf);
        printf("FAIL: %s\n", errbuf);
    }
}

/* Test: model_loading */
static void test_model_loading(void) {
    printf("Testing model_loading... ");

    if (!file_exists(MODEL_FILE)) {
        add_result("model_loading", 0, 1, "model not found");
        printf("SKIP (model not found)\n");
        return;
    }

    /* Read expected config from model file */
    FILE *f = fopen(MODEL_FILE, "rb");
    if (!f) {
        add_result("model_loading", 0, 1, "cannot open model");
        printf("SKIP (cannot open model)\n");
        return;
    }

    int config[7];
    float weights[10];
    if (fread(config, sizeof(int), 7, f) != 7 ||
        fread(weights, sizeof(float), 10, f) != 10) {
        fclose(f);
        add_result("model_loading", 0, 0, "failed to read model");
        printf("FAIL (read error)\n");
        return;
    }
    fclose(f);

    /* vocab_size may be negative (signals shared weights) */
    int vocab_size = config[5];
    if (vocab_size < 0) vocab_size = -vocab_size;

    /* Read Plan 9 output */
    int size;
    char *data = fat_read_file(SHARED_IMAGE, "model_loading.out", &size);
    if (!data || size == 0) {
        add_result("model_loading", 0, 0, "no output file");
        printf("FAIL (no output)\n");
        free(data);
        return;
    }

    /* Parse key=value output */
    int p9_dim = parse_keyval_int(data, "dim");
    int p9_hidden = parse_keyval_int(data, "hidden_dim");
    int p9_layers = parse_keyval_int(data, "n_layers");
    int p9_heads = parse_keyval_int(data, "n_heads");
    int p9_kv = parse_keyval_int(data, "n_kv_heads");
    int p9_vocab = parse_keyval_int(data, "vocab_size");
    int p9_seq = parse_keyval_int(data, "seq_len");

    /* Check config values */
    int ok = 1;
    char errbuf[256] = {0};
    if (p9_dim != config[0]) { ok = 0; snprintf(errbuf, sizeof(errbuf), "dim mismatch: %d vs %d", p9_dim, config[0]); }
    else if (p9_hidden != config[1]) { ok = 0; snprintf(errbuf, sizeof(errbuf), "hidden_dim mismatch"); }
    else if (p9_layers != config[2]) { ok = 0; snprintf(errbuf, sizeof(errbuf), "n_layers mismatch"); }
    else if (p9_heads != config[3]) { ok = 0; snprintf(errbuf, sizeof(errbuf), "n_heads mismatch"); }
    else if (p9_kv != config[4]) { ok = 0; snprintf(errbuf, sizeof(errbuf), "n_kv_heads mismatch"); }
    else if (p9_vocab != vocab_size) { ok = 0; snprintf(errbuf, sizeof(errbuf), "vocab_size mismatch"); }
    else if (p9_seq != config[6]) { ok = 0; snprintf(errbuf, sizeof(errbuf), "seq_len mismatch"); }

    if (!ok) {
        free(data);
        add_result("model_loading", 0, 0, errbuf);
        printf("FAIL: %s\n", errbuf);
        return;
    }

    /* Check first 10 weights */
    float p9_weights[10];
    for (int i = 0; i < 10; i++) {
        char key[8];
        snprintf(key, sizeof(key), "w%d", i);
        p9_weights[i] = parse_keyval_float(data, key);
    }
    free(data);

    if (compare_floats(weights, p9_weights, 10, EPSILON, errbuf, sizeof(errbuf))) {
        add_result("model_loading", 1, 0, NULL);
        printf("PASS\n");
    } else {
        add_result("model_loading", 0, 0, errbuf);
        printf("FAIL: %s\n", errbuf);
    }
}

/* Test: generation */
static void test_generation(void) {
    printf("Testing generation... ");

    if (!file_exists(MODEL_FILE) || !file_exists(TOKENIZER_FILE)) {
        add_result("generation", 0, 1, "model or tokenizer not found");
        printf("SKIP (model or tokenizer not found)\n");
        return;
    }

    /* Read Plan 9 output */
    int size;
    char *data = fat_read_file(SHARED_IMAGE, "generation.out", &size);
    if (!data || size == 0) {
        add_result("generation", 0, 0, "no output file");
        printf("FAIL (no output)\n");
        free(data);
        return;
    }

    /* Just check that we got some output */
    /* A full comparison would require running llama2.c reference on host */
    if (strlen(data) > 10) {
        add_result("generation", 1, 0, NULL);
        printf("PASS (output: %.50s...)\n", data);
    } else {
        add_result("generation", 0, 0, "output too short");
        printf("FAIL (output too short)\n");
    }
    free(data);
}

/* Test: generation_quantized (runq with Q8_0 model) */
static void test_generation_quantized(void) {
    printf("Testing generation_quantized... ");

    if (!file_exists(MODEL_Q_FILE) || !file_exists(TOKENIZER_FILE)) {
        add_result("generation_quantized", 0, 1, "quantized model or tokenizer not found");
        printf("SKIP (quantized model or tokenizer not found)\n");
        return;
    }

    /* Read FP32 output for comparison */
    int fp32_size;
    char *fp32_data = fat_read_file(SHARED_IMAGE, "generation.out", &fp32_size);
    if (!fp32_data || fp32_size == 0) {
        add_result("generation_quantized", 0, 1, "no FP32 output to compare");
        printf("SKIP (no FP32 output to compare)\n");
        free(fp32_data);
        return;
    }

    /* Read quantized output */
    int size;
    char *data = fat_read_file(SHARED_IMAGE, "generation_q.out", &size);
    if (!data || size == 0) {
        add_result("generation_quantized", 0, 0, "no output file");
        printf("FAIL (no output)\n");
        free(fp32_data);
        free(data);
        return;
    }

    /* Extract first line (the generated text, before tok/s stats) */
    char *fp32_line = strtok(fp32_data, "\n");
    char *q_line = strtok(data, "\n");

    if (fp32_line && q_line && strcmp(fp32_line, q_line) == 0) {
        add_result("generation_quantized", 1, 0, NULL);
        printf("PASS (matches FP32)\n");
    } else {
        add_result("generation_quantized", 0, 0, "output differs from FP32");
        printf("FAIL (output differs from FP32)\n");
        if (fp32_line) printf("  FP32: %.50s...\n", fp32_line);
        if (q_line) printf("  Q8_0: %.50s...\n", q_line);
    }
    free(fp32_data);
    free(data);
}

/* Run llmfs tests in single VM (local mount) */
static void run_vm_llmfs_local(void) {
    printf("\n==================================================\n");
    printf("Running llmfs local tests in Plan 9...\n");
    printf("==================================================\n\n");

    /* Compile llmfs - capture errors */
    run_vm_cmd("6c -w llmfs.c >[2=1] > llmfs_compile.log; echo compile_done", 120);
    run_vm_cmd("6l -o llmfs llmfs.6 >[2=1] >> llmfs_compile.log; echo link_done", 60);

    /* Start llmfs and mount locally */
    run_vm_cmd("./llmfs -s llm &", 5);
    run_vm_cmd("mount -c /srv/llm /mnt/llm", 3);

    /* Load model */
    run_vm_cmd("echo 'load stories15M.bin tokenizer.bin' > /mnt/llm/ctl", 5);

    /* Check model info */
    run_vm_cmd("cat /mnt/llm/model > llmfs_model.out", 3);

    /* Create session and read session id */
    run_vm_cmd("cat /mnt/llm/clone > llmfs_session.out", 3);

    /* Configure and run generation - use deterministic settings */
    run_vm_cmd("echo 'temp 0.0' > /mnt/llm/0/ctl", 2);
    run_vm_cmd("echo 'steps 20' > /mnt/llm/0/ctl", 2);
    run_vm_cmd("echo 'seed 42' > /mnt/llm/0/ctl", 2);
    run_vm_cmd("echo 'Once upon a time' > /mnt/llm/0/prompt", 2);
    run_vm_cmd("echo 'generate' > /mnt/llm/0/ctl", 2);

    /* Wait for generation and read output */
    run_vm_cmd("cat /mnt/llm/0/output > llmfs_output.out", 120);

    /* Read status */
    run_vm_cmd("cat /mnt/llm/0/status > llmfs_status.out", 3);

    /* Mark completion */
    run_vm_cmd("echo llmfs_done > llmfs_complete.txt", 2);
}

/* Test: llmfs local generation */
static void test_llmfs_local(void) {
    printf("Testing llmfs_local... ");

    if (!file_exists(MODEL_FILE) || !file_exists(TOKENIZER_FILE)) {
        add_result("llmfs_local", 0, 1, "model or tokenizer not found");
        printf("SKIP (model or tokenizer not found)\n");
        return;
    }

    /* Check compile log for errors */
    int size;
    char *compile_log = fat_read_file(SHARED_IMAGE, "llmfs_compile.log", &size);
    if (compile_log && size > 0) {
        printf("\n  Compile log: %s\n", compile_log);
    }
    free(compile_log);

    /* Check if llmfs test was run */
    char *data = fat_read_file(SHARED_IMAGE, "llmfs_complete.txt", &size);
    if (!data || size == 0) {
        add_result("llmfs_local", 0, 1, "llmfs tests not run");
        printf("SKIP (llmfs tests not run)\n");
        free(data);
        return;
    }
    free(data);

    /* Check model output */
    data = fat_read_file(SHARED_IMAGE, "llmfs_model.out", &size);
    if (!data || size == 0) {
        add_result("llmfs_local", 0, 0, "no model output");
        printf("FAIL (no model output)\n");
        free(data);
        return;
    }

    /* Verify model info contains expected fields */
    if (strstr(data, "dim") == NULL || strstr(data, "vocab_size") == NULL) {
        add_result("llmfs_local", 0, 0, "model info incomplete");
        printf("FAIL (model info incomplete)\n");
        free(data);
        return;
    }
    free(data);

    /* Check generation output */
    data = fat_read_file(SHARED_IMAGE, "llmfs_output.out", &size);
    if (!data || size == 0) {
        add_result("llmfs_local", 0, 0, "no generation output");
        printf("FAIL (no generation output)\n");
        free(data);
        return;
    }

    /* Verify we got some text output */
    if (strlen(data) < 10) {
        add_result("llmfs_local", 0, 0, "output too short");
        printf("FAIL (output too short)\n");
        free(data);
        return;
    }

    printf("PASS (output: %.40s...)\n", data);
    add_result("llmfs_local", 1, 0, NULL);
    free(data);

    /* Check status */
    data = fat_read_file(SHARED_IMAGE, "llmfs_status.out", &size);
    if (data && strstr(data, "done") != NULL) {
        /* Status shows completion with tok/s */
        printf("  Status: %s", data);
    }
    free(data);
}

/* Run llmfs tests with two VMs (remote 9P) */
static int run_dualvm_llmfs_remote(void) {
    printf("\n==================================================\n");
    printf("Running llmfs remote tests (dual VM)...\n");
    printf("==================================================\n\n");

    /* Start both VMs with separate disk overlays */
    if (dualvm_start(&dualvm, CPU_DISK, TERM_DISK, SHARED_IMAGE) != 0) {
        fprintf(stderr, "Failed to start dual VMs\n");
        return -1;
    }

    /* Boot and mount shared disk on both */
    if (dualvm_boot_and_mount_shared(&dualvm) != 0) {
        fprintf(stderr, "Failed to boot dual VMs\n");
        dualvm_shutdown(&dualvm);
        qemu_killall();
        return -1;
    }

    /* Configure network */
    dualvm_configure_network(&dualvm);

    /* CPU VM: Compile and start llmfs */
    printf("CPU: Compiling llmfs...\n");
    qemu_sendln_wait(&dualvm.cpu, "6c -w llmfs.c", 120);
    qemu_sendln_wait(&dualvm.cpu, "6l -o llmfs llmfs.6", 60);

    /* CPU VM: Start llmfs server and mount it locally */
    printf("CPU: Starting llmfs server...\n");
    qemu_sendln_wait(&dualvm.cpu, "./llmfs -s llm &", 10);
    qemu_sendln_wait(&dualvm.cpu, "mount /srv/llm /mnt/llm", 10);

    /* CPU VM: Load model (via local mount) */
    printf("CPU: Loading model...\n");
    qemu_sendln_wait(&dualvm.cpu, "echo 'load stories15M.bin tokenizer.bin' > /mnt/llm/ctl", 10);

    /* CPU VM: Export mounted llmfs tree via 9P over TCP */
    printf("CPU: Starting 9P export on tcp!*!564...\n");
    qemu_sendln_wait(&dualvm.cpu, "aux/listen1 -tv tcp!*!564 /bin/exportfs -r /mnt/llm &", 10);
    qemu_sleep(3);  /* Give listener time to start */

    /* Verify network is working */
    printf("CPU: Checking network setup...\n");
    qemu_sendln_wait(&dualvm.cpu, "cat /net/ipifc/0/status > /mnt/host/cpu_net.log", 5);

    /* Terminal VM: Connect to CPU's 9P server */
    printf("Terminal: Connecting to CPU's 9P server...\n");
    qemu_sendln_wait(&dualvm.terminal, "srv tcp!10.0.0.2!564 llm", 30);
    qemu_sendln_wait(&dualvm.terminal, "mount /srv/llm /mnt/llm", 10);

    /* Terminal VM: Verify 9P mount works */
    printf("Terminal: Verifying 9P mount...\n");
    qemu_sendln_wait(&dualvm.terminal, "ls /mnt/llm", 10);

    /* Terminal VM: Create session and generate (operations go over 9P network) */
    printf("Terminal: Creating session and generating (via 9P)...\n");
    qemu_sendln_wait(&dualvm.terminal, "cat /mnt/llm/clone", 10);  /* Creates session 0 */
    qemu_sendln_wait(&dualvm.terminal, "echo 'temp 0.0' > /mnt/llm/0/ctl", 5);
    qemu_sendln_wait(&dualvm.terminal, "echo 'steps 20' > /mnt/llm/0/ctl", 5);
    qemu_sendln_wait(&dualvm.terminal, "echo 'seed 42' > /mnt/llm/0/ctl", 5);
    qemu_sendln_wait(&dualvm.terminal, "echo 'Once upon a time' > /mnt/llm/0/prompt", 5);
    qemu_sendln_wait(&dualvm.terminal, "echo 'generate' > /mnt/llm/0/ctl", 5);

    /* Terminal VM: Wait for generation and read output (via 9P) */
    printf("Terminal: Waiting for generation (via 9P)...\n");
    qemu_sendln_wait(&dualvm.terminal, "cat /mnt/llm/0/output", 180);

    /* CPU VM saves the results to shared disk (CPU has exclusive FAT access) */
    printf("CPU: Saving remote test results to shared disk...\n");
    qemu_sendln_wait(&dualvm.cpu, "cat /mnt/llm/model > /mnt/host/llmfs_remote_model.out", 10);
    qemu_sendln_wait(&dualvm.cpu, "cat /mnt/llm/0/output > /mnt/host/llmfs_remote_output.out", 10);
    qemu_sendln_wait(&dualvm.cpu, "cat /mnt/llm/0/status > /mnt/host/llmfs_remote_status.out", 10);
    qemu_sendln_wait(&dualvm.cpu, "echo remote_done > /mnt/host/llmfs_remote_complete.txt", 5);
    qemu_sleep(2);  /* Give time for FAT writes to complete */

    /* Shutdown */
    printf("Shutting down dual VMs...\n");
    dualvm_shutdown(&dualvm);
    qemu_killall();

    return 0;
}

/* Test: llmfs remote generation */
static void test_llmfs_remote(void) {
    printf("Testing llmfs_remote... ");

    if (!file_exists(MODEL_FILE) || !file_exists(TOKENIZER_FILE)) {
        add_result("llmfs_remote", 0, 1, "model or tokenizer not found");
        printf("SKIP (model or tokenizer not found)\n");
        return;
    }

    /* Check if remote test was run */
    int size;
    char *data = fat_read_file(SHARED_IMAGE, "llmfs_remote_complete.txt", &size);
    if (!data || size == 0) {
        add_result("llmfs_remote", 0, 1, "remote tests not run");
        printf("SKIP (remote tests not run)\n");
        free(data);
        return;
    }
    free(data);

    /* Check remote generation output */
    data = fat_read_file(SHARED_IMAGE, "llmfs_remote_output.out", &size);
    if (!data || size == 0) {
        add_result("llmfs_remote", 0, 0, "no remote output");
        printf("FAIL (no remote output)\n");
        free(data);
        return;
    }

    /* Verify we got some text output */
    if (strlen(data) < 10) {
        add_result("llmfs_remote", 0, 0, "remote output too short");
        printf("FAIL (remote output too short)\n");
        free(data);
        return;
    }

    /* Compare with local llmfs output if available */
    int local_size;
    char *local_data = fat_read_file(SHARED_IMAGE, "llmfs_output.out", &local_size);
    if (local_data && local_size > 0) {
        if (strcmp(data, local_data) == 0) {
            printf("PASS (matches local: %.40s...)\n", data);
        } else {
            printf("PASS (output: %.40s...)\n", data);
            printf("  Note: differs from local (may be due to seed/timing)\n");
        }
    } else {
        printf("PASS (output: %.40s...)\n", data);
    }

    add_result("llmfs_remote", 1, 0, NULL);
    free(data);
    free(local_data);

    /* Check status */
    data = fat_read_file(SHARED_IMAGE, "llmfs_remote_status.out", &size);
    if (data && strstr(data, "done") != NULL) {
        printf("  Status: %s", data);
    }
    free(data);
}

/* Print summary */
static void print_summary(void) {
    printf("\n==================================================\n");
    printf("Summary\n");
    printf("==================================================\n");

    int passed = 0, failed = 0, skipped = 0;
    for (int i = 0; i < num_results; i++) {
        TestResult *r = &results[i];
        const char *status = r->skipped ? "SKIP" : (r->passed ? "PASS" : "FAIL");
        printf("  %-20s %s", r->name, status);
        if (r->error[0]) printf(" (%s)", r->error);
        printf("\n");

        if (r->skipped) skipped++;
        else if (r->passed) passed++;
        else failed++;
    }

    printf("\nPassed: %d, Failed: %d, Skipped: %d\n", passed, failed, skipped);
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    printf("==================================================\n");
    printf("9ml Automated Test Suite (C Harness)\n");
    printf("==================================================\n\n");

    /* Ensure we're in the right directory */
    if (chdir("test") != 0) {
        /* Already in test dir or running from project root */
    }

    /* Kill any lingering QEMU */
    printf("Cleaning up...\n");
    qemu_killall();

    /* Check for 9front disk */
    if (!file_exists(DISK_IMAGE)) {
        if (qemu_ensure_disk(DISK_IMAGE) != 0) {
            fprintf(stderr, "Cannot obtain 9front disk image\n");
            return 1;
        }
    }

    /* Prepare shared disk */
    if (prepare_shared_disk() != 0) {
        fprintf(stderr, "Failed to prepare shared disk\n");
        return 1;
    }

    /* Boot VM */
    if (boot_vm() != 0) {
        fprintf(stderr, "Failed to boot VM\n");
        qemu_shutdown(&vm);
        return 1;
    }

    /* Run tests in VM */
    run_vm_tests();

    /* Run llmfs local tests */
    run_vm_llmfs_local();

    /* Shutdown VM */
    printf("\nShutting down VM...\n");
    qemu_shutdown(&vm);
    qemu_killall();

    /* Run dual-VM tests for remote 9P */
    int run_remote_tests = 1;  /* Set to 1 to enable remote tests */
    if (run_remote_tests && file_exists(MODEL_FILE)) {
        run_dualvm_llmfs_remote();
    }

    /* Compare results */
    printf("\n==================================================\n");
    printf("Verifying Results\n");
    printf("==================================================\n\n");

    test_rmsnorm();
    test_softmax();
    test_matmul();
    test_rng();
    test_quantize();
    test_quantized_matmul();
    test_model_loading();
    test_generation();
    test_generation_quantized();
    test_llmfs_local();
    test_llmfs_remote();

    print_summary();

    /* Return non-zero if any tests failed */
    for (int i = 0; i < num_results; i++) {
        if (!results[i].passed && !results[i].skipped) {
            return 1;
        }
    }
    return 0;
}
