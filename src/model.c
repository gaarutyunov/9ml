/* Inference for Llama-2 Transformer model in pure C */
/* Plan 9 port - Model implementation with SIMD and parallel optimizations */
/*
 * Define DISABLE_OPTIMIZATIONS before including this file to disable
 * threading and SIMD assembly (for simple tests that don't link with assembly).
 */

#include <u.h>
#include <libc.h>

#ifndef DISABLE_OPTIMIZATIONS
#include <thread.h>
#endif

/* Include arch plugin types and registry */
#include "arch/arch.h"

// ----------------------------------------------------------------------------
// Optimization configuration

typedef struct {
    int nthreads;           /* number of threads to use (0 = auto-detect) */
    int use_simd;           /* whether to use SIMD optimizations */
    int softmax_mode;       /* 0=scalar, 1=partial, 2=schraudolph, 3=poly, 4=lut */
} OptConfig;

OptConfig opt_config = {
    .nthreads = 0,      /* 0 = auto-detect */
#ifdef DISABLE_OPTIMIZATIONS
    .use_simd = 0,      /* SIMD disabled when optimizations disabled (no assembly) */
#else
    .use_simd = 1,      /* SIMD enabled by default */
#endif
    .softmax_mode = 2,  /* default to Schraudolph (57x speedup, ~1e-5 error) */
};

/* Thread pool for parallel execution */
typedef struct ThreadPool ThreadPool;
typedef struct WorkItem WorkItem;

struct WorkItem {
    void (*fn)(void*);
    void *arg;
    WorkItem *next;
};

#ifndef DISABLE_OPTIMIZATIONS
struct ThreadPool {
    int nworkers;
    int active;
    Channel *work;
    Channel *done;
    Channel *shutdown;
    Channel *ready;     /* workers signal ready before entering main loop */
};
#else
struct ThreadPool {
    int nworkers;
    int active;
};
#endif

/* Global thread pool - accessible to arch plugins */
ThreadPool *global_pool = nil;

/* Detect number of CPUs by reading /dev/sysstat */
int
cpu_count(void)
{
#ifdef DISABLE_OPTIMIZATIONS
    return 1;
#else
    int fd, n, count;
    char *buf, *p;

    /* Use heap instead of stack to avoid stack overflow in threadmain */
    buf = malloc(4096);
    if (buf == nil) {
        return 1;
    }

    fd = open("/dev/sysstat", OREAD);
    if (fd < 0) {
        free(buf);
        return 1;
    }

    n = read(fd, buf, 4095);
    close(fd);

    if (n <= 0) {
        free(buf);
        return 1;
    }
    buf[n] = '\0';

    count = 0;
    for (p = buf; *p; p++) {
        if (*p == '\n') {
            count++;
        }
    }

    free(buf);
    return count > 0 ? count : 1;
#endif
}

#ifndef DISABLE_OPTIMIZATIONS
/* Worker thread main loop */
static void
worker_proc(void *arg)
{
    ThreadPool *p = arg;
    WorkItem *item;

    /* Signal that we're ready */
    sendp(p->ready, (void*)1);

    for (;;) {
        item = recvp(p->work);
        if (item == nil) {
            /* Acknowledge shutdown and exit */
            sendp(p->shutdown, (void*)1);
            break;
        }

        if (item->fn != nil) {
            item->fn(item->arg);
        }

        sendp(p->done, (void*)1);
        free(item);
    }

    threadexits(nil);
}

/* Create a thread pool */
ThreadPool*
pool_create(int nworkers)
{
    ThreadPool *p;
    int i;

    if (nworkers <= 0) {
        nworkers = cpu_count();
    }

    p = malloc(sizeof(ThreadPool));
    if (p == nil) {
        return nil;
    }

    p->nworkers = nworkers;
    p->active = 1;
    p->work = chancreate(sizeof(void*), nworkers * 4);
    p->done = chancreate(sizeof(void*), nworkers * 4);
    p->shutdown = chancreate(sizeof(void*), nworkers);
    p->ready = chancreate(sizeof(void*), nworkers);

    if (p->work == nil || p->done == nil || p->shutdown == nil || p->ready == nil) {
        if (p->work) chanfree(p->work);
        if (p->done) chanfree(p->done);
        if (p->shutdown) chanfree(p->shutdown);
        if (p->ready) chanfree(p->ready);
        free(p);
        return nil;
    }

    /* Create worker threads with 8KB stack */
    for (i = 0; i < nworkers; i++) {
        proccreate(worker_proc, p, 8192);
    }

    /* Wait for all workers to be ready before returning */
    for (i = 0; i < nworkers; i++) {
        recvp(p->ready);
    }

    return p;
}

/* Destroy a thread pool */
void
pool_destroy(ThreadPool *p)
{
    int i;

    if (p == nil) {
        return;
    }

    p->active = 0;

    /* Send shutdown signals to all workers */
    for (i = 0; i < p->nworkers; i++) {
        sendp(p->work, nil);
    }

    /* Wait for all workers to acknowledge shutdown */
    for (i = 0; i < p->nworkers; i++) {
        recvp(p->shutdown);
    }

    /* Now safe to free channels */
    chanfree(p->work);
    chanfree(p->done);
    chanfree(p->shutdown);
    chanfree(p->ready);
    free(p);
}

/* Submit work to the thread pool */
void
pool_submit(ThreadPool *p, void (*fn)(void*), void *arg)
{
    WorkItem *item;

    if (p == nil || !p->active) {
        if (fn != nil) {
            fn(arg);
        }
        return;
    }

    item = malloc(sizeof(WorkItem));
    if (item == nil) {
        if (fn != nil) {
            fn(arg);
        }
        return;
    }

    item->fn = fn;
    item->arg = arg;
    item->next = nil;

    sendp(p->work, item);
}

/* Wait for njobs to complete */
void
pool_wait(ThreadPool *p, int njobs)
{
    int i;

    if (p == nil) {
        return;
    }

    for (i = 0; i < njobs; i++) {
        recvp(p->done);
    }
}

#else /* DISABLE_OPTIMIZATIONS - stub implementations */

ThreadPool* pool_create(int nworkers) { USED(nworkers); return nil; }
void pool_destroy(ThreadPool *p) { USED(p); }
void pool_submit(ThreadPool *p, void (*fn)(void*), void *arg) { USED(p); if (fn) fn(arg); }
void pool_wait(ThreadPool *p, int njobs) { USED(p); USED(njobs); }

#endif /* DISABLE_OPTIMIZATIONS */

/* Initialize optimization subsystem */
void
opt_init(void)
{
    if (opt_config.nthreads == 0) {
        opt_config.nthreads = cpu_count();
    }

    /* Create thread pool if using more than 1 thread */
    if (opt_config.nthreads > 1 && global_pool == nil) {
        global_pool = pool_create(opt_config.nthreads);
    }
}

/* Cleanup optimization subsystem */
void
opt_cleanup(void)
{
    if (global_pool != nil) {
        pool_destroy(global_pool);
        global_pool = nil;
    }
}

// ----------------------------------------------------------------------------
// SIMD function declarations
// Note: Plan 9 assembly SIMD support is limited. We provide C fallbacks
// that can be replaced with assembly when available.

#ifndef DISABLE_OPTIMIZATIONS
// Forward declare scalar versions (defined below)
void matmul_scalar(float* xout, float* x, float* w, int n, int d);
void rmsnorm_scalar(float* o, float* x, float* weight, int size);

// ----------------------------------------------------------------------------
// SIMD functions - implemented in simd_amd64.s assembly
// Uses SSE packed instructions for 4x float parallelism.
//
// Plan 9 amd64 calling convention:
// - First integer/pointer arg is in BP (RARG)
// - Subsequent args are on stack at +8(FP), +16(FP), etc.

/* Assembly SIMD functions - defined in simd_amd64.s */
extern void matmul_simd(float *xout, float *x, float *w, int n, int d);
extern float dot_product_simd(float *a, float *b, int n);
extern void rmsnorm_simd(float *o, float *x, float *weight, int size);
extern void vec_add_simd(float *o, float *a, float *b, int n);
extern void vec_scale_simd(float *o, float *x, float scalar, int n);

/* Softmax SIMD helper functions - defined in simd_amd64.s */
extern float softmax_max_simd(float *x, int size);
extern float softmax_sum_simd(float *x, int size);
extern void softmax_scale_simd(float *x, float scale, int size);
extern void softmax_subtract_simd(float *x, float val, int size);
extern void exp_schraudolph_simd(float *x, int size);
extern void exp_poly_simd(float *x, int size);

/*
 * softmax_simd - Optimized softmax (C implementation)
 * x[i] = exp(x[i] - max) / sum(exp(x - max))
 * Note: Kept in C because it requires exp() from the math library.
 * Uses 4x unrolled loops for max, exp, and normalize passes.
 */
void softmax_simd(float *x, int size) {
    int i;

    /* Find max value (4x unrolled) */
    float max0 = x[0];
    float max1 = size > 1 ? x[1] : x[0];
    float max2 = size > 2 ? x[2] : x[0];
    float max3 = size > 3 ? x[3] : x[0];

    for (i = 4; i + 3 < size; i += 4) {
        if (x[i]     > max0) max0 = x[i];
        if (x[i + 1] > max1) max1 = x[i + 1];
        if (x[i + 2] > max2) max2 = x[i + 2];
        if (x[i + 3] > max3) max3 = x[i + 3];
    }

    float max_val = max0;
    if (max1 > max_val) max_val = max1;
    if (max2 > max_val) max_val = max2;
    if (max3 > max_val) max_val = max3;

    /* Handle remaining elements for max */
    for (; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Exp and sum (4x unrolled) */
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (i = 0; i + 3 < size; i += 4) {
        x[i]     = exp(x[i]     - max_val);
        x[i + 1] = exp(x[i + 1] - max_val);
        x[i + 2] = exp(x[i + 2] - max_val);
        x[i + 3] = exp(x[i + 3] - max_val);
        sum0 += x[i];
        sum1 += x[i + 1];
        sum2 += x[i + 2];
        sum3 += x[i + 3];
    }

    float sum = sum0 + sum1 + sum2 + sum3;

    /* Handle remaining elements for exp */
    for (; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }

    /* Normalize (4x unrolled) */
    float inv_sum = 1.0f / sum;
    for (i = 0; i + 3 < size; i += 4) {
        x[i]     *= inv_sum;
        x[i + 1] *= inv_sum;
        x[i + 2] *= inv_sum;
        x[i + 3] *= inv_sum;
    }

    /* Handle remaining elements for normalize */
    for (; i < size; i++) {
        x[i] *= inv_sum;
    }
}
#endif

// ----------------------------------------------------------------------------
// Transformer model
//
// Types (Config, TransformerWeights, RunState, Transformer) are defined in
// arch/arch.h. This ensures binary compatibility with the arch plugins.

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprint(2, "malloc failed!\n");
        exits("malloc");
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    uvlong n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

// ----------------------------------------------------------------------------
// 9ml Model Format
//
// Legacy format (llama2.c compatible): 7 ints (28 bytes)
//   dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
//
// Extended format (9ml v1): magic + explicit architecture info
//   magic (4 bytes): 0x394D4C01 ("9ML\x01")
//   header_size (4 bytes): total header size (60 for v1)
//   arch_id (4 bytes): 1=llama2, 2=llama3, 3=mistral
//   rope_theta (4 bytes): float, e.g., 10000.0 or 500000.0
//   ffn_type (4 bytes): 0=SwiGLU, 1=GeGLU, 2=GELU
//   flags (4 bytes): bit0=attn_bias, bit1=mlp_bias
//   sliding_window (4 bytes): 0=full attention, >0=window size
//   reserved (4 bytes): for future use
//   dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len (28 bytes)
//
// Detection: first 4 bytes. If == 0x394D4C01, it's extended format.
// Otherwise it's legacy (first int is dim, typically 64-8192).

#define FORMAT_MAGIC_9ML_V1   0x394D4C01  /* "9ML\x01" */
#define LEGACY_CONFIG_SIZE    (7 * sizeof(int))   /* 28 bytes */
#define EXTENDED_HEADER_SIZE  (15 * sizeof(int))  /* 60 bytes */

/* Architecture IDs are defined in arch/arch.h */

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     float** data, vlong* file_size) {
    int fd;
    Dir *d;
    vlong n;
    uint magic;
    int header_size;
    int shared_weights;

    // open file and get size
    fd = open(checkpoint, OREAD);
    if (fd < 0) {
        fprint(2, "Couldn't open file %s\n", checkpoint);
        exits("open");
    }
    d = dirfstat(fd);
    if (d == nil) {
        fprint(2, "dirfstat failed\n");
        close(fd);
        exits("dirfstat");
    }
    *file_size = d->length;
    free(d);

    // Read first 4 bytes to detect format
    n = read(fd, &magic, sizeof(uint));
    if (n != sizeof(uint)) {
        fprint(2, "failed to read header\n");
        close(fd);
        exits("read");
    }

    if (magic == FORMAT_MAGIC_9ML_V1) {
        // Extended 9ml format - read full extended header
        int ext_buf[14];  /* header_size + 13 more fields */
        n = read(fd, ext_buf, 14 * sizeof(int));
        if (n != 14 * sizeof(int)) {
            fprint(2, "failed to read extended header\n");
            close(fd);
            exits("read");
        }

        header_size = ext_buf[0];
        config->arch_id = ext_buf[1];

        // rope_theta is stored as float bits in an int slot
        union { int i; float f; } theta_conv;
        theta_conv.i = ext_buf[2];
        config->rope_theta = theta_conv.f;

        // Skip ffn_type (ext_buf[3]), flags (ext_buf[4]),
        // sliding_window (ext_buf[5]), reserved (ext_buf[6])

        config->dim = ext_buf[7];
        config->hidden_dim = ext_buf[8];
        config->n_layers = ext_buf[9];
        config->n_heads = ext_buf[10];
        config->n_kv_heads = ext_buf[11];
        config->vocab_size = ext_buf[12];
        config->seq_len = ext_buf[13];

        // Handle shared weights flag
        shared_weights = config->vocab_size > 0 ? 1 : 0;
        if (config->vocab_size < 0) config->vocab_size = -config->vocab_size;

        // Skip any extra header bytes if header_size > EXTENDED_HEADER_SIZE
        if (header_size > EXTENDED_HEADER_SIZE) {
            seek(fd, header_size - EXTENDED_HEADER_SIZE, 1);
        }

    } else {
        // Legacy llama2.c format - magic was actually dim
        int config_buf[6];  /* remaining 6 fields */
        n = read(fd, config_buf, 6 * sizeof(int));
        if (n != 6 * sizeof(int)) {
            fprint(2, "failed to read legacy config\n");
            close(fd);
            exits("read");
        }

        config->dim = (int)magic;  /* first int was dim */
        config->hidden_dim = config_buf[0];
        config->n_layers = config_buf[1];
        config->n_heads = config_buf[2];
        config->n_kv_heads = config_buf[3];
        config->vocab_size = config_buf[4];
        config->seq_len = config_buf[5];

        // negative vocab size signals unshared weights
        shared_weights = config->vocab_size > 0 ? 1 : 0;
        if (config->vocab_size < 0) config->vocab_size = -config->vocab_size;

        // Auto-detect architecture from vocab size (heuristic for legacy files)
        if (config->vocab_size >= 100000) {
            config->rope_theta = 500000.0f;
            config->arch_id = ARCH_LLAMA3;
        } else {
            config->rope_theta = 10000.0f;
            config->arch_id = ARCH_LLAMA2;
        }

        header_size = LEGACY_CONFIG_SIZE;
    }

    // Allocate memory for weights
    vlong weights_size = *file_size - header_size;
    *data = malloc(weights_size);
    if (*data == nil) {
        fprint(2, "malloc failed for weights\n");
        close(fd);
        exits("malloc");
    }

    // Read weights in chunks (Plan 9 read may have size limits)
    char *buf = (char*)*data;
    vlong remaining = weights_size;
    while (remaining > 0) {
        long chunk = remaining > 8192 ? 8192 : remaining;
        n = read(fd, buf, chunk);
        if (n <= 0) {
            fprint(2, "failed to read weights\n");
            close(fd);
            exits("read");
        }
        buf += n;
        remaining -= n;
    }
    close(fd);

    memory_map_weights(weights, config, *data, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // Initialize architecture subsystem once
    arch_init();

    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);

    // Bind architecture plugin based on detected arch_id
    t->arch = arch_find_by_id(t->config.arch_id);
    // Note: t->arch may be nil if no plugin matches - forward() handles this
}

void free_transformer(Transformer* t) {
    // free the weights data
    if (t->data != nil) { free(t->data); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

/* Scalar C implementation of rmsnorm */
void rmsnorm_scalar(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrt(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

/* Dispatcher for rmsnorm - uses SIMD or scalar based on config */
void rmsnorm(float* o, float* x, float* weight, int size) {
#ifndef DISABLE_OPTIMIZATIONS
    if (opt_config.use_simd) {
        rmsnorm_simd(o, x, weight, size);
    } else {
        rmsnorm_scalar(o, x, weight, size);
    }
#else
    rmsnorm_scalar(o, x, weight, size);
#endif
}

// ----------------------------------------------------------------------------
// Softmax implementations
// Mode 0: scalar (baseline), Mode 1: partial SIMD, Mode 2: Schraudolph,
// Mode 3: polynomial, Mode 4: LUT

/* Mode 0: Scalar softmax (baseline) */
void softmax_scalar(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

#ifndef DISABLE_OPTIMIZATIONS
/* Mode 1: Partial SIMD - SIMD for max/sum/scale, C exp() */
void softmax_partial(float* x, int size) {
    float max_val = softmax_max_simd(x, size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    softmax_scale_simd(x, 1.0f/sum, size);
}

/* Mode 2: Schraudolph fast exp approximation */
void softmax_schraudolph(float* x, int size) {
    float max_val = softmax_max_simd(x, size);
    softmax_subtract_simd(x, max_val, size);
    exp_schraudolph_simd(x, size);
    float sum = softmax_sum_simd(x, size);
    softmax_scale_simd(x, 1.0f/sum, size);
}

/* Mode 3: Polynomial exp approximation */
void softmax_poly(float* x, int size) {
    float max_val = softmax_max_simd(x, size);
    softmax_subtract_simd(x, max_val, size);
    exp_poly_simd(x, size);
    float sum = softmax_sum_simd(x, size);
    softmax_scale_simd(x, 1.0f/sum, size);
}

/* Mode 4: LUT exp approximation with linear interpolation */
#define EXP_LUT_SIZE 512
#define EXP_LUT_MIN -20.0f
#define EXP_LUT_MAX 20.0f
static float exp_lut[EXP_LUT_SIZE];
static int exp_lut_initialized = 0;

void exp_lut_init(void) {
    if (exp_lut_initialized) return;
    float range = EXP_LUT_MAX - EXP_LUT_MIN;
    for (int i = 0; i < EXP_LUT_SIZE; i++) {
        float x = EXP_LUT_MIN + i * (range / (EXP_LUT_SIZE - 1));
        exp_lut[i] = exp(x);
    }
    exp_lut_initialized = 1;
}

void softmax_lut(float* x, int size) {
    exp_lut_init();
    float max_val = softmax_max_simd(x, size);
    float range = EXP_LUT_MAX - EXP_LUT_MIN;
    float scale = (EXP_LUT_SIZE - 1) / range;
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        float v = x[i] - max_val;  // Now in [-inf, 0]
        // Clamp to LUT range
        if (v < EXP_LUT_MIN) v = EXP_LUT_MIN;
        if (v > EXP_LUT_MAX) v = EXP_LUT_MAX;
        float idx_f = (v - EXP_LUT_MIN) * scale;
        int idx = (int)idx_f;
        if (idx >= EXP_LUT_SIZE - 1) idx = EXP_LUT_SIZE - 2;
        float frac = idx_f - idx;
        // Linear interpolation
        x[i] = exp_lut[idx] * (1.0f - frac) + exp_lut[idx + 1] * frac;
        sum += x[i];
    }
    softmax_scale_simd(x, 1.0f/sum, size);
}
#endif

/* Softmax dispatcher - selects implementation based on opt_config.softmax_mode */
void softmax(float* x, int size) {
#ifndef DISABLE_OPTIMIZATIONS
    switch (opt_config.softmax_mode) {
    case 1: softmax_partial(x, size); break;
    case 2: softmax_schraudolph(x, size); break;
    case 3: softmax_poly(x, size); break;
    case 4: softmax_lut(x, size); break;
    default: softmax_scalar(x, size); break;
    }
#else
    softmax_scalar(x, size);
#endif
}

/* Scalar C implementation of matmul */
void matmul_scalar(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

/* Dispatcher for matmul - uses SIMD or scalar based on config */
void matmul(float* xout, float* x, float* w, int n, int d) {
#ifndef DISABLE_OPTIMIZATIONS
    if (opt_config.use_simd) {
        matmul_simd(xout, x, w, n, d);
    } else {
        matmul_scalar(xout, x, w, n, d);
    }
#else
    matmul_scalar(xout, x, w, n, d);
#endif
}

// ----------------------------------------------------------------------------
// Parallel attention

/* Context for parallel attention head computation */
typedef struct {
    int h;              /* head index */
    int head_size;
    int pos;
    int seq_len;
    int kv_dim;
    int kv_mul;
    int loff;
    float *q;
    float *att;
    float *xb;
    float *key_cache;
    float *value_cache;
} HeadContext;

/* Worker function for single attention head */
static void
attention_head_worker(void *arg)
{
    HeadContext *ctx = arg;
    int h = ctx->h;
    int head_size = ctx->head_size;
    int pos = ctx->pos;
    int seq_len = ctx->seq_len;
    int kv_dim = ctx->kv_dim;
    int kv_mul = ctx->kv_mul;
    int loff = ctx->loff;

    /* Get pointers for this head */
    float *q = ctx->q + h * head_size;
    float *att = ctx->att + h * seq_len;
    float *xb = ctx->xb + h * head_size;

    int t, i;
    float score, a;
    float *k, *v;

    /* Compute attention scores */
    for (t = 0; t <= pos; t++) {
        k = ctx->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        score = 0.0f;
        for (i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        att[t] = score / sqrt((float)head_size);
    }

    /* Softmax the scores */
    softmax(att, pos + 1);

    /* Weighted sum of values */
    memset(xb, 0, head_size * sizeof(float));
    for (t = 0; t <= pos; t++) {
        v = ctx->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        a = att[t];
        for (i = 0; i < head_size; i++) {
            xb[i] += a * v[i];
        }
    }
}

/* Parallel multihead attention */
static void
parallel_attention(RunState *s, Config *p, int pos, int loff, int kv_dim, int kv_mul, int head_size) {
    int h;
    HeadContext *contexts;

    /* If single-threaded or no pool, use sequential */
    if (opt_config.nthreads <= 1 || global_pool == nil) {
        for (h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;

            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                att[t] = score / sqrt(head_size);
            }

            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        return;
    }

    /* Allocate contexts for all heads */
    contexts = malloc(p->n_heads * sizeof(HeadContext));
    if (contexts == nil) {
        /* Fallback to sequential */
        for (h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;

            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                att[t] = score / sqrt(head_size);
            }

            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        return;
    }

    /* Submit all heads to thread pool */
    for (h = 0; h < p->n_heads; h++) {
        contexts[h].h = h;
        contexts[h].head_size = head_size;
        contexts[h].pos = pos;
        contexts[h].seq_len = p->seq_len;
        contexts[h].kv_dim = kv_dim;
        contexts[h].kv_mul = kv_mul;
        contexts[h].loff = loff;
        contexts[h].q = s->q;
        contexts[h].att = s->att;
        contexts[h].xb = s->xb;
        contexts[h].key_cache = s->key_cache;
        contexts[h].value_cache = s->value_cache;

        pool_submit(global_pool, attention_head_worker, &contexts[h]);
    }

    /* Wait for all heads to complete */
    pool_wait(global_pool, p->n_heads);

    free(contexts);
}

float* forward(Transformer* transformer, int token, int pos) {

    // Dispatch to architecture plugin if available
    if (transformer->arch != nil && transformer->arch->forward != nil) {
        return transformer->arch->forward(transformer, token, pos);
    }

    // Fallback: built-in forward implementation (LLaMA 2 compatible)
    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(uvlong l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        // Uses rope_theta from config (10000 for LLaMA2, 500000 for LLaMA3)
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / pow(p->rope_theta, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cos(val);
            float fci = sin(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention - use parallel version
        parallel_attention(s, p, pos, loff, kv_dim, kv_mul, head_size);

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + exp(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    uint max_token_length;
    uchar byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(void *a, void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    int fd;
    long n;

    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = nil; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (uchar)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    fd = open(tokenizer_path, OREAD);
    if (fd < 0) {
        fprint(2, "couldn't load %s\n", tokenizer_path);
        exits("open");
    }
    n = read(fd, &t->max_token_length, sizeof(int));
    if (n != sizeof(int)) {
        fprint(2, "failed read\n");
        exits("read");
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (read(fd, t->vocab_scores + i, sizeof(float)) != sizeof(float)) {
            fprint(2, "failed read\n");
            exits("read");
        }
        if (read(fd, &len, sizeof(int)) != sizeof(int)) {
            fprint(2, "failed read\n");
            exits("read");
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (read(fd, t->vocab[i], len) != len) {
            fprint(2, "failed read\n");
            exits("read");
        }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    close(fd);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    uint byte_val;
    char buf[8];
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
        // manual hex parsing since Plan 9 sscanf may differ
        buf[0] = piece[3];
        buf[1] = piece[4];
        buf[2] = '\0';
        byte_val = strtoul(buf, nil, 16);
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == nil) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        uchar byte_val = piece[0];
        // check if printable or whitespace
        if (byte_val >= 0x20 && byte_val < 0x7f) {
            // printable
        } else if (byte_val == '\t' || byte_val == '\n' || byte_val == '\r' || byte_val == ' ') {
            // whitespace
        } else {
            return; // bad byte, don't print it
        }
    }
    print("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    // binary search
    int lo = 0, hi = vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(str, sorted_vocab[mid].str);
        if (cmp == 0) return sorted_vocab[mid].id;
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

void encode(Tokenizer* t, char *text, int bos, int eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == nil) {
        fprint(2, "cannot encode NULL text\n");
        exits("encode");
    }

    if (t->sorted_vocab == nil) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    ulong str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            for (ulong i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (uchar)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprint(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    uvlong rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare_prob(void* a, void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, uvlong rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

uint random_u32(uvlong *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(uvlong *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms(void) {
    // return time in milliseconds, for benchmarking the model speed
    return nsec() / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == nil) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprint(2, "something is wrong, expected at least 1 prompt token\n");
        exits("encode");
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {

        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    print("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprint(2, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(char* guide, char* buffer, int bufsize) {
    print("%s", guide);
    // Plan 9 doesn't have fgets, use read
    int n = read(0, buffer, bufsize - 1);
    if (n > 0) {
        buffer[n] = '\0';
        // strip newline
        if (n > 0 && buffer[n-1] == '\n') {
            buffer[n-1] = '\0';
        }
    } else {
        buffer[0] = '\0';
    }
}

// ----------------------------------------------------------------------------
// chat loop

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    int user_turn = 1;
    int next;
    int token;
    int pos = 0;
    while (pos < steps) {

        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == nil) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            if (pos == 0 && cli_user_prompt != nil) {
                strcpy(user_prompt, cli_user_prompt);
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprint(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprint(rendered_prompt, user_template, user_prompt);
            }
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            print("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) { user_turn = 1; }

        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece);
        }
        if (next == 2) { print("\n"); }
    }
    print("\n");
    free(prompt_tokens);
}
