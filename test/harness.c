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

static TestResult results[12];
static int num_results = 0;
static QemuVM vm;

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

    /* Wait for boot prompts */
    printf("Waiting for boot (15s)...\n");
    qemu_sleep(15);
    qemu_sendln(&vm, "");  /* Accept default bootargs */

    qemu_sleep(5);
    qemu_sendln(&vm, "");  /* Accept default user (glenda) */

    printf("Waiting for shell (20s)...\n");
    qemu_sleep(20);

    /* Mount shared disk */
    printf("Mounting shared disk...\n");
    qemu_sendln(&vm, "dossrv -f /dev/sdG0/data shared");
    qemu_sleep(2);
    qemu_sendln(&vm, "mount -c /srv/shared /mnt/host");
    qemu_sleep(2);
    qemu_sendln(&vm, "cd /mnt/host");
    qemu_sleep(1);

    return 0;
}

/* Run a command in the VM and wait */
static void run_vm_cmd(const char *cmd, int wait_secs) {
    qemu_sendln(&vm, cmd);
    qemu_sleep(wait_secs);
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

    /* Shutdown VM */
    printf("\nShutting down VM...\n");
    qemu_shutdown(&vm);
    qemu_killall();

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

    print_summary();

    /* Return non-zero if any tests failed */
    for (int i = 0; i < num_results; i++) {
        if (!results[i].passed && !results[i].skipped) {
            return 1;
        }
    }
    return 0;
}
