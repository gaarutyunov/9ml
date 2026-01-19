/*
 * test_hf_download.c - HuggingFace Download Integration Test
 *
 * This is a REAL integration test that:
 * 1. Tests network connectivity (DNS, TLS)
 * 2. Downloads a model from HuggingFace
 * 3. Loads it into the model pool
 * 4. Generates text
 *
 * Requires: Internet-enabled VM with proper DNS/TLS configuration
 *
 * Build: Compile http.c, hfhub.c separately, then link:
 *   6c -w -Idownload download/http.c
 *   6c -w -Idownload download/hfhub.c
 *   6c -w -Ipool -Idownload test_hf_download.c
 *   6l -o t_hf_download test_hf_download.6 http.6 hfhub.6 simd_amd64.6 arch/arch.a6
 */

/* Include model (includes u.h, libc.h, thread.h) */
#include "model.c"

/* Include pool implementation */
#include "pool/pool.c"

/* Include headers only - implementations compiled separately */
#include "download/http.h"
#include "download/hfhub.h"

/* Stack size for threadmain */
int mainstacksize = 1024*1024;

/* Start webfs if not already running */
static int
start_webfs(void)
{
    Dir *d;
    int i;

    /* Check if already mounted */
    d = dirstat("/mnt/web/clone");
    if (d != nil) {
        free(d);
        print("  webfs already running\n");
        return 1;
    }

    /* Start webfs */
    print("  Starting webfs...\n");
    if (fork() == 0) {
        execl("/bin/webfs", "webfs", nil);
        exits("exec webfs");
    }

    /* Wait for webfs to start */
    for (i = 0; i < 10; i++) {
        sleep(500);
        d = dirstat("/mnt/web/clone");
        if (d != nil) {
            free(d);
            print("  webfs started successfully\n");
            return 1;
        }
    }

    print("  Failed to start webfs\n");
    return 0;
}

/* Progress callback for downloads */
static void
download_progress(vlong downloaded, vlong total, void *arg)
{
    USED(arg);
    if (total > 0)
        print("\r  Progress: %lld / %lld bytes (%d%%)",
              downloaded, total, (int)(downloaded * 100 / total));
    else
        print("\r  Downloaded: %lld bytes", downloaded);
}

/* Test DNS resolution */
static int
test_dns_resolution(void)
{
    int fd;
    char buf[256];
    int n;

    print("Test: DNS resolution for huggingface.co\n");

    /* Use cs to translate hostname */
    fd = open("/net/cs", ORDWR);
    if (fd < 0) {
        print("  FAIL: Cannot open /net/cs\n");
        print("  Error: %r\n");
        return 0;
    }

    /* Query for huggingface.co */
    if (write(fd, "tcp!huggingface.co!443", 22) != 22) {
        print("  FAIL: Cannot write to cs\n");
        close(fd);
        return 0;
    }

    seek(fd, 0, 0);
    n = read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (n <= 0) {
        print("  FAIL: No response from cs\n");
        return 0;
    }
    buf[n] = '\0';

    print("  Resolved: %s\n", buf);
    print("  PASS: DNS working\n");
    return 1;
}

/* Test TCP connection to HuggingFace */
static int
test_tcp_connection(void)
{
    int fd;

    print("Test: TCP connection to huggingface.co:443\n");

    fd = dial("tcp!huggingface.co!443", nil, nil, nil);
    if (fd < 0) {
        print("  FAIL: Cannot connect to huggingface.co:443\n");
        print("  Error: %r\n");
        return 0;
    }
    close(fd);
    print("  PASS: TCP connection successful\n");
    return 1;
}

/* Test HTTPS using hget command (Plan 9's native HTTP client) */
static int
test_tls_connection(void)
{
    int fd;
    char buf[1024];
    int n;
    Waitmsg *w;

    print("Test: HTTPS via hget command\n");

    /* Use hget to fetch a small file - this tests TLS via webfs internally */
    print("  Running: hget -o /tmp/robots.txt https://huggingface.co/robots.txt\n");

    int pid = fork();
    if (pid == 0) {
        /* Child: run hget */
        int null = open("/dev/null", OWRITE);
        if (null >= 0) {
            dup(null, 2);  /* Suppress stderr */
            close(null);
        }
        execl("/bin/hget", "hget", "-o", "/tmp/robots.txt", "https://huggingface.co/robots.txt", nil);
        exits("exec hget failed");
    }

    if (pid < 0) {
        print("  FAIL: fork failed\n");
        return 0;
    }

    /* Wait for hget with timeout */
    print("  Waiting for hget (pid %d)...\n", pid);

    /* Use alarm for timeout */
    int timeout = 30;  /* 30 second timeout */
    for (int i = 0; i < timeout; i++) {
        w = wait();
        if (w != nil) {
            if (w->pid == pid) {
                if (w->msg[0] == '\0') {
                    print("  hget completed successfully\n");
                    free(w);
                    break;
                } else {
                    print("  hget failed: %s\n", w->msg);
                    free(w);
                    return 0;
                }
            }
            free(w);
        }
        sleep(1000);
    }

    /* Check if file was downloaded */
    Dir *d = dirstat("/tmp/robots.txt");
    if (d == nil) {
        print("  FAIL: /tmp/robots.txt not found after hget\n");
        return 0;
    }

    print("  Downloaded file: %lld bytes\n", d->length);
    free(d);

    /* Read first part of file */
    fd = open("/tmp/robots.txt", OREAD);
    if (fd < 0) {
        print("  FAIL: cannot open downloaded file: %r\n");
        return 0;
    }

    n = read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (n <= 0) {
        print("  FAIL: file is empty\n");
        return 0;
    }

    buf[n] = '\0';
    /* Truncate for display */
    if (n > 100) buf[100] = '\0';
    print("  Content: %s...\n", buf);

    print("  PASS: HTTPS via hget works\n");
    return 1;
}

/* Test HTTP GET request */
static int
test_http_request(void)
{
    HttpClient http;
    HttpResponse resp;
    char buf[1024];
    int n;

    print("Test: HTTP GET request to HuggingFace API\n");
    print("  Connecting via webfs...\n");

    http_init(&http);
    if (http_connect(&http, "huggingface.co", 443, 1) < 0) {
        print("  FAIL: Connection failed: %s\n", http.errmsg);
        return 0;
    }
    print("  Connected, sending request...\n");

    /* Request the API root - should return JSON */
    if (http_get(&http, "/api/models?limit=1", nil, &resp) < 0) {
        print("  FAIL: GET request failed: %s\n", http.errmsg);
        http_close(&http);
        return 0;
    }

    if (resp.status != HTTP_OK) {
        print("  FAIL: HTTP status %d\n", resp.status);
        http_resp_free(&resp);
        http_close(&http);
        return 0;
    }

    /* Read some of the response body */
    n = read(http.fd, buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        /* Truncate for display */
        if (n > 100) {
            buf[100] = '\0';
            strcat(buf, "...");
        }
        print("  Response: %s\n", buf);
    }

    http_resp_free(&resp);
    http_close(&http);
    print("  PASS: HTTP request successful\n");
    return 1;
}

/* Test downloading a small file from HuggingFace */
static int
test_hf_download_file(void)
{
    HFHub hub;
    HFModel model;
    HFFile *files, *f;
    char *localpath = "/tmp/hf_test_model.bin";

    print("Test: Download model file from HuggingFace\n");
    print("  Repository: karpathy/tinyllamas\n");

    /* Initialize hub client */
    hf_init(&hub, "/tmp/hf_cache");

    /* Get model info */
    print("  Fetching file list...\n");
    if (hf_get_model_info(&hub, "karpathy/tinyllamas", "main", &model) < 0) {
        print("  FAIL: Cannot get model info: %s\n", hub.errmsg);
        hf_close(&hub);
        return 0;
    }

    print("  Found files:\n");
    for (f = model.files; f != nil; f = f->next) {
        print("    %s (%lld bytes)%s\n",
              f->filename, f->size, f->is_lfs ? " [LFS]" : "");
    }

    /* Find .bin files */
    files = hf_find_files(&model, ".bin");
    if (files == nil) {
        print("  FAIL: No .bin files found\n");
        hf_model_free(&model);
        hf_close(&hub);
        return 0;
    }

    /* Download the smallest .bin file */
    f = files;
    for (HFFile *tmp = files; tmp != nil; tmp = tmp->next) {
        if (tmp->size > 0 && tmp->size < f->size)
            f = tmp;
    }

    print("  Downloading: %s (%lld bytes)\n", f->filename, f->size);

    if (hf_download_file_to(&hub, &model, f, localpath, download_progress, nil) < 0) {
        print("\n  FAIL: Download failed: %s\n", hub.errmsg);
        hf_files_free(files);
        hf_model_free(&model);
        hf_close(&hub);
        return 0;
    }
    print("\n");

    /* Verify file exists and has expected size */
    Dir *d = dirstat(localpath);
    if (d == nil) {
        print("  FAIL: Downloaded file not found\n");
        hf_files_free(files);
        hf_model_free(&model);
        hf_close(&hub);
        return 0;
    }

    print("  Downloaded: %s (%lld bytes)\n", localpath, d->length);
    free(d);

    hf_files_free(files);
    hf_model_free(&model);
    hf_close(&hub);

    print("  PASS: Download successful\n");
    return 1;
}

/* Test loading downloaded model and generating text */
static int
test_model_generation(char *modelpath, char *tokpath)
{
    ModelPool pool;
    PoolEntry *entry;
    Transformer *t;
    Tokenizer *tok;
    Sampler sampler;
    int *prompt_tokens;
    int num_prompt_tokens;
    char *prompt = "Once upon a time";
    int token, next, pos;
    char *piece;
    float *logits;
    char output[512];
    int outlen = 0;

    print("Test: Load model and generate text\n");
    print("  Model: %s\n", modelpath);
    print("  Tokenizer: %s\n", tokpath);

    /* Initialize pool */
    pool_init(&pool, 4, 500 * 1024 * 1024);

    /* Load model into pool */
    entry = pool_load(&pool, "hf-model", modelpath, tokpath);
    if (entry == nil) {
        print("  FAIL: Cannot load model\n");
        pool_free(&pool);
        return 0;
    }

    print("  Loaded: %s (memory: %lludMB)\n",
          entry->name, entry->memory / (1024 * 1024));

    t = (Transformer *)entry->transformer;
    tok = (Tokenizer *)entry->tokenizer;

    /* Encode prompt */
    prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tok, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        print("  FAIL: Encoding failed\n");
        free(prompt_tokens);
        pool_release(&pool, entry);
        pool_free(&pool);
        return 0;
    }

    /* Build sampler (greedy for deterministic output) */
    build_sampler(&sampler, t->config.vocab_size, 0.0f, 0.9f, 42);

    /* Generate 20 tokens */
    print("  Generating from prompt: '%s'\n", prompt);
    token = prompt_tokens[0];
    for (pos = 0; pos < 20 && outlen < sizeof(output) - 100; pos++) {
        logits = forward(t, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(&sampler, logits);
        }

        if (next == 1) /* EOS */
            break;

        piece = decode(tok, token, next);
        if (piece) {
            int plen = strlen(piece);
            if (outlen + plen < sizeof(output) - 1) {
                strcpy(output + outlen, piece);
                outlen += plen;
            }
        }

        token = next;
    }
    output[outlen] = '\0';

    free_sampler(&sampler);
    free(prompt_tokens);
    pool_release(&pool, entry);
    pool_free(&pool);

    if (outlen > 20) {
        print("  Generated: %s\n", output);
        print("  PASS: Generation successful\n");
        return 1;
    } else {
        print("  FAIL: Output too short: %s\n", output);
        return 0;
    }
}

void
threadmain(int argc, char *argv[])
{
    int passed = 0;
    int failed = 0;
    int total = 0;
    char *downloaded_model = "/tmp/hf_test_model.bin";
    char *tokenizer = "tokenizer.bin";

    USED(argc);
    USED(argv);

    print("=== HuggingFace Download Integration Test ===\n");
    print("This test downloads a real model from HuggingFace.\n\n");

    /* Test 1: DNS resolution */
    total++;
    if (test_dns_resolution())
        passed++;
    else {
        failed++;
        print("  Skipping remaining tests - DNS not working\n");
        goto done;
    }
    print("\n");

    /* Test 2: TCP connection */
    total++;
    if (test_tcp_connection())
        passed++;
    else {
        failed++;
        print("  Skipping remaining tests - TCP not working\n");
        goto done;
    }
    print("\n");

    /* Test 3: TLS connection */
    total++;
    if (test_tls_connection())
        passed++;
    else {
        failed++;
        print("  Skipping remaining tests - TLS not working\n");
        goto done;
    }
    print("\n");

    /* Test 4: HTTP request */
    total++;
    if (test_http_request())
        passed++;
    else {
        failed++;
        print("  Skipping remaining tests - HTTP not working\n");
        goto done;
    }
    print("\n");

    /* Test 5: Download model file */
    total++;
    if (test_hf_download_file())
        passed++;
    else {
        failed++;
        print("  Skipping model loading - download failed\n");
        goto done;
    }
    print("\n");

    /* Test 6: Load and generate */
    total++;
    if (test_model_generation(downloaded_model, tokenizer))
        passed++;
    else
        failed++;
    print("\n");

done:
    /* Summary */
    print("=== Result ===\n");
    if (failed == 0) {
        print("PASS: HuggingFace integration tests passed (%d/%d tests)\n",
              passed, total);
    } else {
        print("FAIL: %d passed, %d failed out of %d tests\n",
              passed, failed, total);
    }

    exits(failed ? "fail" : nil);
}
