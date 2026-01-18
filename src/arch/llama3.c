/*
 * arch/llama3.c - LLaMA 3 architecture plugin for 9ml
 *
 * LLaMA 3 (Meta, 2024) characteristics:
 *   - RoPE with theta=500000 (extended context)
 *   - SwiGLU activation (same as LLaMA 2)
 *   - No biases in attention or MLP
 *   - Grouped-Query Attention (GQA) support
 *   - RMSNorm (same as LLaMA 2)
 *   - Larger vocabulary (128K tokens)
 *
 * The main difference from LLaMA 2 is the rope_theta value which
 * allows for much longer context windows.
 */

#include <u.h>
#include <libc.h>

#ifndef DISABLE_THREADING
#include <thread.h>
#endif

#include "arch.h"
#include "../core/kernels.h"

/* LLaMA 3 specific constants */
#define LLAMA3_ROPE_THETA    500000.0f
#define LLAMA3_VOCAB_SIZE    128256    /* Typical LLaMA 3 vocab size */

/* ----------------------------------------------------------------------------
 * Thread pool access (defined in model.c)
 * ---------------------------------------------------------------------------- */

#ifndef DISABLE_THREADING
typedef struct ThreadPool ThreadPool;
extern ThreadPool *global_pool;
extern void pool_submit(ThreadPool *p, void (*fn)(void*), void *arg);
extern void pool_wait(ThreadPool *p, int njobs);
#endif

/* ----------------------------------------------------------------------------
 * Attention head worker context (shared with llama2.c)
 * ---------------------------------------------------------------------------- */

typedef struct {
    int h;
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

static void
attention_head_worker(void *arg)
{
    HeadContext *ctx = arg;
    int h = ctx->h;
    int head_size = ctx->head_size;
    int pos = ctx->pos;
    int kv_dim = ctx->kv_dim;
    int kv_mul = ctx->kv_mul;
    int loff = ctx->loff;
    float *q_ptr, *att_ptr, *xb_ptr, *k, *v;
    int t, i;
    float score, a;

    q_ptr = ctx->q + h * head_size;
    att_ptr = ctx->att + h * ctx->seq_len;
    xb_ptr = ctx->xb + h * head_size;

    /* Compute attention scores */
    for (t = 0; t <= pos; t++) {
        k = ctx->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        score = 0.0f;
        for (i = 0; i < head_size; i++) {
            score += q_ptr[i] * k[i];
        }
        att_ptr[t] = score / sqrt((float)head_size);
    }

    /* Softmax */
    softmax(att_ptr, pos + 1);

    /* Weighted sum of values */
    memset(xb_ptr, 0, head_size * sizeof(float));
    for (t = 0; t <= pos; t++) {
        v = ctx->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        a = att_ptr[t];
        for (i = 0; i < head_size; i++) {
            xb_ptr[i] += a * v[i];
        }
    }
}

/* ----------------------------------------------------------------------------
 * Parallel multihead attention
 * ---------------------------------------------------------------------------- */

static void
parallel_attention(RunState *s, ModelConfig *p, int pos, int loff,
                   int kv_dim, int kv_mul, int head_size)
{
    int h;
    HeadContext *contexts;

#ifndef DISABLE_THREADING
    if (opt_config.nthreads <= 1 || global_pool == nil) {
#endif
        for (h = 0; h < p->n_heads; h++) {
            float *q_ptr = s->q + h * head_size;
            float *att_ptr = s->att + h * p->seq_len;
            float *xb_ptr = s->xb + h * head_size;
            int t, i;
            float score, a;
            float *k, *v;

            for (t = 0; t <= pos; t++) {
                k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                score = 0.0f;
                for (i = 0; i < head_size; i++) {
                    score += q_ptr[i] * k[i];
                }
                att_ptr[t] = score / sqrt((float)head_size);
            }

            softmax(att_ptr, pos + 1);

            memset(xb_ptr, 0, head_size * sizeof(float));
            for (t = 0; t <= pos; t++) {
                v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                a = att_ptr[t];
                for (i = 0; i < head_size; i++) {
                    xb_ptr[i] += a * v[i];
                }
            }
        }
        return;
#ifndef DISABLE_THREADING
    }

    contexts = malloc(p->n_heads * sizeof(HeadContext));
    if (contexts == nil) {
        /* Fallback to sequential */
        for (h = 0; h < p->n_heads; h++) {
            float *q_ptr = s->q + h * head_size;
            float *att_ptr = s->att + h * p->seq_len;
            float *xb_ptr = s->xb + h * head_size;
            int t, i;
            float score, a;
            float *k, *v;

            for (t = 0; t <= pos; t++) {
                k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                score = 0.0f;
                for (i = 0; i < head_size; i++) {
                    score += q_ptr[i] * k[i];
                }
                att_ptr[t] = score / sqrt((float)head_size);
            }

            softmax(att_ptr, pos + 1);

            memset(xb_ptr, 0, head_size * sizeof(float));
            for (t = 0; t <= pos; t++) {
                v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                a = att_ptr[t];
                for (i = 0; i < head_size; i++) {
                    xb_ptr[i] += a * v[i];
                }
            }
        }
        return;
    }

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

    pool_wait(global_pool, p->n_heads);
    free(contexts);
#endif
}

/* ----------------------------------------------------------------------------
 * LLaMA 3 Forward Pass
 *
 * Identical to LLaMA 2 except rope_theta is read from config (500000 for LLaMA 3)
 * ---------------------------------------------------------------------------- */

static float *
llama3_forward(ModelInstance *m, int token, int pos)
{
    ModelConfig *p = &m->config;
    TransformerWeights *w = &m->weights;
    RunState *s = &m->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    uvlong l;
    int loff, i;
    float *content_row;

    /* Copy the token embedding into x */
    content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));

    /* Forward all the layers */
    for (l = 0; l < p->n_layers; l++) {
        /* Attention rmsnorm */
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        /* Key and value point to the kv cache */
        loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        /* QKV matmuls for this position */
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        /* RoPE relative positional encoding - uses config rope_theta */
        rope_apply_standard(s->q, s->k, dim, kv_dim, pos, head_size, p);

        /* Multihead attention */
        parallel_attention(s, p, pos, loff, kv_dim, kv_mul, head_size);

        /* Final matmul to get the output of the attention */
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        /* Residual connection back into x */
        for (i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        /* FFN rmsnorm */
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        /* FFN: self.w2(F.silu(self.w1(x)) * self.w3(x)) */
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        /* SwiGLU non-linearity */
        for (i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + exp(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        /* Final matmul to get the output of the ffn */
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        /* Residual connection */
        for (i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    /* Final rmsnorm */
    rmsnorm(x, x, w->rms_final_weight, dim);

    /* Classifier into logits */
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);

    return s->logits;
}

/* ----------------------------------------------------------------------------
 * LLaMA 3 Architecture Detection
 *
 * LLaMA 3 can be detected by:
 *   1. Large vocabulary size (128K+)
 *   2. Extended header with rope_theta field
 *   3. GGUF metadata (handled by format/gguf.c)
 * ---------------------------------------------------------------------------- */

static int
llama3_detect(void *data, vlong size, ModelConfig *cfg)
{
    int *header;
    int vocab_size;

    USED(size);

    if (size < 28) {
        return 0;
    }

    header = (int *)data;
    vocab_size = header[5];
    if (vocab_size < 0) vocab_size = -vocab_size;

    /* LLaMA 3 typically has 128K+ vocabulary */
    if (vocab_size >= 100000) {
        cfg->rope_theta = LLAMA3_ROPE_THETA;
        cfg->arch_id = ARCH_LLAMA3;
        return 1;
    }

    return 0;
}

/* ----------------------------------------------------------------------------
 * Memory Estimation
 * ---------------------------------------------------------------------------- */

static uvlong
llama3_estimate_memory(ModelConfig *cfg, int quant_type)
{
    uvlong params;
    uvlong head_size;
    uvlong kv_dim;
    int bytes_per_param;

    USED(quant_type);

    head_size = cfg->dim / cfg->n_heads;
    kv_dim = cfg->n_kv_heads * head_size;

    params = 0;
    params += cfg->vocab_size * cfg->dim;
    params += cfg->n_layers * cfg->dim;
    params += cfg->n_layers * cfg->dim;
    params += cfg->n_layers * cfg->dim * cfg->dim;
    params += cfg->n_layers * cfg->dim * kv_dim;
    params += cfg->n_layers * cfg->dim * kv_dim;
    params += cfg->n_layers * cfg->dim * cfg->dim;
    params += cfg->n_layers * cfg->dim * cfg->hidden_dim;
    params += cfg->n_layers * cfg->hidden_dim * cfg->dim;
    params += cfg->n_layers * cfg->dim * cfg->hidden_dim;
    params += cfg->dim;
    params += cfg->dim * cfg->vocab_size;

    bytes_per_param = sizeof(float);

    return params * bytes_per_param;
}

/* ----------------------------------------------------------------------------
 * Architecture Registration
 * ---------------------------------------------------------------------------- */

static ModelArch llama3_arch = {
    .name = "llama3",
    .arch_id = ARCH_LLAMA3,
    .forward = llama3_forward,
    .apply_rope = rope_apply_standard,
    .estimate_memory = llama3_estimate_memory,
    .detect = llama3_detect,
};

void
llama3_register(void)
{
    arch_register(&llama3_arch);
}
