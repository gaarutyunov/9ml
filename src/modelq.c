/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */
/* Plan 9 port - Model implementation with SIMD and parallel optimizations */
/*
 * Define DISABLE_THREADING before including this file to disable
 * thread support (for simple tests that don't need parallelism).
 */

#include <u.h>
#include <libc.h>

#ifndef DISABLE_THREADING
#include <thread.h>
#endif

// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

// ----------------------------------------------------------------------------
// Optimization configuration

typedef struct {
    int nthreads;           /* number of threads to use (0 = auto-detect) */
    int use_simd;           /* whether to use SIMD optimizations */
} OptConfig;

OptConfig opt_config = {
    .nthreads = 0,      /* 0 = auto-detect */
#ifdef DISABLE_THREADING
    .use_simd = 0,      /* SIMD disabled when threading disabled (no assembly linkage) */
#else
    .use_simd = 1,      /* SIMD enabled by default */
#endif
};

/* Thread pool for parallel execution */
typedef struct ThreadPool ThreadPool;
typedef struct WorkItem WorkItem;

struct WorkItem {
    void (*fn)(void*);
    void *arg;
    WorkItem *next;
};

#ifndef DISABLE_THREADING
struct ThreadPool {
    int nworkers;
    int active;
    Channel *work;
    Channel *done;
    Channel *shutdown;
    Channel *ready;
};
#else
struct ThreadPool {
    int nworkers;
    int active;
};
#endif

static ThreadPool *global_pool = nil;

/* Detect number of CPUs by reading /dev/sysstat */
int
cpu_count(void)
{
#ifdef DISABLE_THREADING
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

#ifndef DISABLE_THREADING
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

    /* 32KB stack - enough for worker with some headroom */
    for (i = 0; i < nworkers; i++) {
        proccreate(worker_proc, p, 32768);
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

#else /* DISABLE_THREADING - stub implementations */

ThreadPool* pool_create(int nworkers) { USED(nworkers); return nil; }
void pool_destroy(ThreadPool *p) { USED(p); }
void pool_submit(ThreadPool *p, void (*fn)(void*), void *arg) { USED(p); if (fn) fn(arg); }
void pool_wait(ThreadPool *p, int njobs) { USED(p); USED(njobs); }

#endif /* DISABLE_THREADING */

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
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    schar* q;    // quantized values (int8_t -> schar in Plan 9)
    float* s; // scaling factors
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // file-based loading (Plan 9: no mmap)
    float* data; // loaded data pointer
    vlong file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq.q = calloc(p->dim, sizeof(schar));
    s->xq.s = calloc(p->dim, sizeof(float));
    s->hq.q = calloc(p->hidden_dim, sizeof(schar));
    s->hq.s = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
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
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            schar quantized = (schar) floor(quant_value + 0.5f); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (schar*)p;
        p = (schar*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uchar shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

// Config is stored as 7 ints (28 bytes) in the file, but Plan 9 may pad the struct
#define CONFIG_FILE_SIZE (7 * sizeof(int))

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     float** data, vlong* file_size) {
    int fd;
    Dir *d;
    vlong n;
    int config_buf[7];
    uint magic_number;
    int version;
    int header_size = 256; // the header size for version 2 in bytes
    uchar shared_classifier;
    int group_size;

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

    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    n = read(fd, &magic_number, sizeof(uint));
    if (n != sizeof(uint)) {
        fprint(2, "failed to read magic number\n");
        close(fd);
        exits("read");
    }
    if (magic_number != 0x616b3432) {
        fprint(2, "Bad magic number\n");
        close(fd);
        exits("magic");
    }

    // read in the version number (uint32), has to be 2
    n = read(fd, &version, sizeof(int));
    if (n != sizeof(int)) {
        fprint(2, "failed to read version\n");
        close(fd);
        exits("read");
    }
    if (version != 2) {
        fprint(2, "Bad version %d, need version 2\n", version);
        close(fd);
        exits("version");
    }

    // read in the Config (as raw ints to avoid struct padding issues)
    n = read(fd, config_buf, CONFIG_FILE_SIZE);
    if (n != CONFIG_FILE_SIZE) {
        fprint(2, "failed to read config\n");
        close(fd);
        exits("read");
    }
    config->dim = config_buf[0];
    config->hidden_dim = config_buf[1];
    config->n_layers = config_buf[2];
    config->n_heads = config_buf[3];
    config->n_kv_heads = config_buf[4];
    config->vocab_size = config_buf[5];
    config->seq_len = config_buf[6];

    // read in flags
    n = read(fd, &shared_classifier, sizeof(uchar));
    if (n != sizeof(uchar)) {
        fprint(2, "failed to read shared_classifier\n");
        close(fd);
        exits("read");
    }

    n = read(fd, &group_size, sizeof(int));
    if (n != sizeof(int)) {
        fprint(2, "failed to read group_size\n");
        close(fd);
        exits("read");
    }
    GS = group_size; // set as global, as it will be used in many places

    // allocate memory for data and read from header offset
    vlong data_size = *file_size - header_size;
    *data = malloc(data_size);
    if (*data == nil) {
        fprint(2, "malloc failed for weights\n");
        close(fd);
        exits("malloc");
    }

    // seek to header offset and read all data
    seek(fd, header_size, 0);

    // read data in chunks (Plan 9 read may have size limits)
    char *buf = (char*)*data;
    vlong remaining = data_size;
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

    memory_map_weights(weights, config, *data, shared_classifier);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // free QuantizedTensors
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    // free the data
    free(t->data);
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
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

void softmax(float* x, int size) {
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

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    for (int i = 0; i < d; i++) {

        float val = 0.0f;
        int ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int) x->q[j + k]) * ((int) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
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
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / pow(10000.0f, head_dim / (float)head_size);
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

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention - use parallel version
        parallel_attention(s, p, pos, loff, kv_dim, kv_mul, head_size);

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

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
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
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

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
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
    int fd = open(tokenizer_path, OREAD);
    if (fd < 0) {
        fprint(2, "couldn't load %s\n", tokenizer_path);
        exits("open");
    }
    if (read(fd, &t->max_token_length, sizeof(int)) != sizeof(int)) {
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
    char* piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    uchar byte_val;
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        // parse hex byte
        char hex[3] = {piece[3], piece[4], '\0'};
        byte_val = (uchar)strtol(hex, nil, 16);
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
        if (!(byte_val >= 32 && byte_val < 127) && byte_val != '\n' && byte_val != '\t') {
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
        fprint(2, "cannot encode nil text\n");
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
    vlong str_len = 0;

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

        // append this byte to the buffer
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
            for (int i=0; i < str_len; i++) {
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

int compare_prob(const void* a, const void* b) {
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
        print("achieved tok/s: %f\n", (double)(pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, vlong bufsize) {
    print("%s", guide);
    if (read(0, buffer, bufsize - 1) <= 0) { buffer[0] = '\0'; return; }
    vlong len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') { buffer[len - 1] = '\0'; }
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
