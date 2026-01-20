/*
 * arch/gemma3.c - Gemma 3 architecture plugin for 9ml
 *
 * Gemma 3 (Google, 2024) characteristics:
 *   - Alternating sliding window (local) and full (global) attention
 *   - Pattern: 5 local layers, 1 global layer (repeating)
 *   - GeGLU activation: GELU(x) * W3(x)
 *   - QK normalization before RoPE
 *   - Dual RoPE frequencies: 10000 (local), 1000000 (global)
 *   - Embedding scaling by sqrt(dim)
 *   - Post-attention and post-FFN normalization
 *   - RMSNorm with +1 weight offset
 */

#include <u.h>
#include <libc.h>

#ifndef DISABLE_THREADING
#include <thread.h>
#endif

#include "arch.h"
#include "../core/kernels.h"

/* ----------------------------------------------------------------------------
 * Gemma 3 Constants
 * ---------------------------------------------------------------------------- */

/* Sliding window pattern: 5 local + 1 global */
#define GEMMA3_LOCAL_LAYERS_PER_CYCLE 5
#define GEMMA3_CYCLE_LENGTH 6

/* Default theta values */
#define GEMMA3_LOCAL_THETA  10000.0f
#define GEMMA3_GLOBAL_THETA 1000000.0f

/* Default epsilon for RMSNorm */
#define GEMMA3_RMS_EPS 1e-6f

/* ----------------------------------------------------------------------------
 * GELU Activation (tanh approximation)
 * ---------------------------------------------------------------------------- */

/*
 * gelu_tanh - GELU with tanh approximation
 * Matches PyTorch's gelu_pytorch_tanh:
 *   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
static float
gelu_tanh(float x)
{
    float c = 0.7978845608f;  /* sqrt(2/pi) */
    float x3 = x * x * x;
    float inner = c * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanh(inner));
}

/* ----------------------------------------------------------------------------
 * Gemma-style RMSNorm (with +1 weight offset)
 * ---------------------------------------------------------------------------- */

static void
rmsnorm_gemma(float *o, float *x, float *weight, int size, float eps)
{
    int j;
    float ss = 0.0f;

    /* Calculate sum of squares */
    for (j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += eps;
    ss = 1.0f / sqrt(ss);

    /* Normalize and scale with +1 offset on weight */
    for (j = 0; j < size; j++) {
        o[j] = (1.0f + weight[j]) * (ss * x[j]);
    }
}

/* ----------------------------------------------------------------------------
 * QK Normalization
 * ---------------------------------------------------------------------------- */

/*
 * qk_norm - Normalize query or key vector (per head)
 * Simple L2 normalization: x / sqrt(mean(x^2) + eps)
 */
static void
qk_norm(float *o, float *x, int size, float eps)
{
    int j;
    float ss = 0.0f;

    for (j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += eps;
    ss = 1.0f / sqrt(ss);

    for (j = 0; j < size; j++) {
        o[j] = x[j] * ss;
    }
}

/* ----------------------------------------------------------------------------
 * Attention Layer Type Detection
 * ---------------------------------------------------------------------------- */

/*
 * gemma3_is_local_layer - Returns 1 if layer uses local (sliding window) attention
 * Pattern: 5 local layers, then 1 global layer, repeating
 */
static int
gemma3_is_local_layer(int layer_idx)
{
    return (layer_idx % GEMMA3_CYCLE_LENGTH) < GEMMA3_LOCAL_LAYERS_PER_CYCLE;
}

/*
 * gemma3_get_rope_theta - Get RoPE theta for a given layer
 */
static float
gemma3_get_rope_theta(int layer_idx, ModelConfig *cfg)
{
    if (gemma3_is_local_layer(layer_idx)) {
        return cfg->rope_local_theta > 0 ? cfg->rope_local_theta : GEMMA3_LOCAL_THETA;
    } else {
        return cfg->rope_global_theta > 0 ? cfg->rope_global_theta : GEMMA3_GLOBAL_THETA;
    }
}

/* ----------------------------------------------------------------------------
 * RoPE with configurable theta
 * ---------------------------------------------------------------------------- */

static void
rope_apply_gemma3(float *q, float *k, int dim, int kv_dim, int pos,
                  int head_size, float theta)
{
    int i, head_dim, rotn, v;
    float freq, val, fcr, fci, v0, v1;
    float *vec;

    for (i = 0; i < dim; i += 2) {
        head_dim = i % head_size;
        freq = 1.0f / pow(theta, head_dim / (float)head_size);
        val = pos * freq;
        fcr = cos(val);
        fci = sin(val);

        /* How many vectors? 2 = q & k, 1 = q only (when i >= kv_dim) */
        rotn = i < kv_dim ? 2 : 1;

        for (v = 0; v < rotn; v++) {
            vec = v == 0 ? q : k;
            v0 = vec[i];
            v1 = vec[i + 1];
            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

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
 * Attention head worker context
 * ---------------------------------------------------------------------------- */

typedef struct {
    int h;
    int head_size;
    int pos;
    int seq_len;
    int kv_dim;
    int kv_mul;
    int loff;
    int sliding_window;
    int is_local;
    float attn_scale;
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
    int t, i, start;
    float score, a;

    q_ptr = ctx->q + h * head_size;
    att_ptr = ctx->att + h * ctx->seq_len;
    xb_ptr = ctx->xb + h * head_size;

    /* Determine attention window start */
    if (ctx->is_local && ctx->sliding_window > 0) {
        start = pos > ctx->sliding_window - 1 ? pos - ctx->sliding_window + 1 : 0;
    } else {
        start = 0;
    }

    /* Clear attention scores outside window */
    for (t = 0; t < start; t++) {
        att_ptr[t] = 0.0f;
    }

    /* Compute attention scores within window */
    for (t = start; t <= pos; t++) {
        k = ctx->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        score = 0.0f;
        for (i = 0; i < head_size; i++) {
            score += q_ptr[i] * k[i];
        }
        att_ptr[t] = score * ctx->attn_scale;
    }

    /* Softmax over valid positions */
    softmax(att_ptr + start, pos - start + 1);

    /* Clear softmax output for positions before window */
    for (t = 0; t < start; t++) {
        att_ptr[t] = 0.0f;
    }

    /* Weighted sum of values */
    memset(xb_ptr, 0, head_size * sizeof(float));
    for (t = start; t <= pos; t++) {
        v = ctx->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        a = att_ptr[t];
        for (i = 0; i < head_size; i++) {
            xb_ptr[i] += a * v[i];
        }
    }
}

/* ----------------------------------------------------------------------------
 * Parallel multihead attention for Gemma 3
 * ---------------------------------------------------------------------------- */

static void
parallel_attention_gemma3(RunState *s, ModelConfig *p, int pos, int loff,
                          int kv_dim, int kv_mul, int head_size, int is_local,
                          float attn_scale)
{
    int h;
    HeadContext *contexts;

#ifndef DISABLE_THREADING
    /* If single-threaded or no pool, use sequential */
    if (opt_config.nthreads <= 1 || global_pool == nil) {
#endif
        for (h = 0; h < p->n_heads; h++) {
            float *q_ptr = s->q + h * head_size;
            float *att_ptr = s->att + h * p->seq_len;
            float *xb_ptr = s->xb + h * head_size;
            int t, i, start;
            float score, a;
            float *k, *v;

            /* Determine attention window start */
            if (is_local && p->sliding_window > 0) {
                start = pos > p->sliding_window - 1 ? pos - p->sliding_window + 1 : 0;
            } else {
                start = 0;
            }

            /* Clear attention scores outside window */
            for (t = 0; t < start; t++) {
                att_ptr[t] = 0.0f;
            }

            /* Compute attention scores */
            for (t = start; t <= pos; t++) {
                k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                score = 0.0f;
                for (i = 0; i < head_size; i++) {
                    score += q_ptr[i] * k[i];
                }
                att_ptr[t] = score * attn_scale;
            }

            /* Softmax over valid positions only */
            softmax(att_ptr + start, pos - start + 1);

            /* Clear softmax output for positions before window */
            for (t = 0; t < start; t++) {
                att_ptr[t] = 0.0f;
            }

            /* Weighted sum of values */
            memset(xb_ptr, 0, head_size * sizeof(float));
            for (t = start; t <= pos; t++) {
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

    /* Allocate contexts and submit to thread pool */
    contexts = malloc(p->n_heads * sizeof(HeadContext));
    if (contexts == nil) {
        /* Fallback to sequential on allocation failure - recursive call with threading disabled */
        for (h = 0; h < p->n_heads; h++) {
            float *q_ptr = s->q + h * head_size;
            float *att_ptr = s->att + h * p->seq_len;
            float *xb_ptr = s->xb + h * head_size;
            int t, i, start;
            float score, a;
            float *k, *v;

            if (is_local && p->sliding_window > 0) {
                start = pos > p->sliding_window - 1 ? pos - p->sliding_window + 1 : 0;
            } else {
                start = 0;
            }

            for (t = 0; t < start; t++) {
                att_ptr[t] = 0.0f;
            }

            for (t = start; t <= pos; t++) {
                k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                score = 0.0f;
                for (i = 0; i < head_size; i++) {
                    score += q_ptr[i] * k[i];
                }
                att_ptr[t] = score * attn_scale;
            }

            softmax(att_ptr + start, pos - start + 1);

            for (t = 0; t < start; t++) {
                att_ptr[t] = 0.0f;
            }

            memset(xb_ptr, 0, head_size * sizeof(float));
            for (t = start; t <= pos; t++) {
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
        contexts[h].sliding_window = p->sliding_window;
        contexts[h].is_local = is_local;
        contexts[h].attn_scale = attn_scale;
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
 * Gemma 3 Forward Pass
 * ---------------------------------------------------------------------------- */

static float *
gemma3_forward(ModelInstance *m, int token, int pos)
{
    ModelConfig *p = &m->config;
    TransformerWeights *w = &m->weights;
    RunState *s = &m->state;
    float *x = s->x;
    int dim = p->dim;
    int head_size = p->head_dim > 0 ? p->head_dim : dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    float eps = p->rms_norm_eps > 0 ? p->rms_norm_eps : GEMMA3_RMS_EPS;
    uvlong l;
    int loff, i, h;
    float *content_row;
    float embed_scale;
    int is_local;
    float rope_theta, attn_scale;

    /* Copy the token embedding into x with scaling */
    content_row = w->token_embedding_table + token * dim;
    embed_scale = sqrt((float)dim);
    for (i = 0; i < dim; i++) {
        x[i] = content_row[i] * embed_scale;
    }

    /* Forward all the layers */
    for (l = 0; l < p->n_layers; l++) {
        is_local = gemma3_is_local_layer(l);
        rope_theta = gemma3_get_rope_theta(l, p);

        /* Compute attention scale */
        if (p->query_pre_attn_scalar > 0) {
            attn_scale = 1.0f / sqrt((float)p->query_pre_attn_scalar);
        } else {
            attn_scale = 1.0f / sqrt((float)head_size);
        }

        /* Pre-attention RMSNorm (Gemma style with +1 offset) */
        rmsnorm_gemma(s->xb, x, w->rms_att_weight + l * dim, dim, eps);

        /* Key and value point to the kv cache */
        loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        /* QKV matmuls for this position */
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        /* QK normalization (if enabled) */
        if (p->use_qk_norm && w->q_norm_weight != nil && w->k_norm_weight != nil) {
            /* Normalize each Q head */
            for (h = 0; h < p->n_heads; h++) {
                qk_norm(s->q + h * head_size, s->q + h * head_size, head_size, eps);
            }
            /* Normalize each K head */
            for (h = 0; h < p->n_kv_heads; h++) {
                qk_norm(s->k + h * head_size, s->k + h * head_size, head_size, eps);
            }
        }

        /* RoPE with layer-specific theta */
        rope_apply_gemma3(s->q, s->k, dim, kv_dim, pos, head_size, rope_theta);

        /* Multihead attention with sliding window for local layers */
        parallel_attention_gemma3(s, p, pos, loff, kv_dim, kv_mul, head_size,
                                  is_local, attn_scale);

        /* Final matmul to get the output of the attention */
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        /* Post-attention normalization (if weights present) */
        if (w->post_att_norm_weight != nil) {
            rmsnorm_gemma(s->xb2, s->xb2, w->post_att_norm_weight + l * dim, dim, eps);
        }

        /* Residual connection back into x */
        for (i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        /* Pre-FFN RMSNorm (Gemma style with +1 offset) */
        rmsnorm_gemma(s->xb, x, w->rms_ffn_weight + l * dim, dim, eps);

        /* FFN: GeGLU activation - self.w2(gelu(self.w1(x)) * self.w3(x)) */
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        /* GeGLU non-linearity */
        for (i = 0; i < hidden_dim; i++) {
            float gelu_val = gelu_tanh(s->hb[i]);
            s->hb[i] = gelu_val * s->hb2[i];
        }

        /* Final matmul to get the output of the FFN */
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        /* Post-FFN normalization (if weights present) */
        if (w->post_ffn_norm_weight != nil) {
            rmsnorm_gemma(s->xb, s->xb, w->post_ffn_norm_weight + l * dim, dim, eps);
        }

        /* Residual connection */
        for (i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    /* Final RMSNorm (Gemma style) */
    rmsnorm_gemma(x, x, w->rms_final_weight, dim, eps);

    /* Classifier into logits */
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);

    return s->logits;
}

/* ----------------------------------------------------------------------------
 * Memory Estimation
 * ---------------------------------------------------------------------------- */

static uvlong
gemma3_estimate_memory(ModelConfig *cfg, int quant_type)
{
    uvlong params;
    uvlong head_size;
    uvlong kv_dim;
    int bytes_per_param;

    USED(quant_type);  /* TODO: support different quant types */

    head_size = cfg->head_dim > 0 ? cfg->head_dim : cfg->dim / cfg->n_heads;
    kv_dim = cfg->n_kv_heads * head_size;

    /* Count parameters */
    params = 0;
    params += cfg->vocab_size * cfg->dim;           /* token embeddings */
    params += cfg->n_layers * cfg->dim;             /* rms_att_weight */
    params += cfg->n_layers * cfg->dim;             /* rms_ffn_weight */
    params += cfg->n_layers * cfg->dim * cfg->dim;  /* wq */
    params += cfg->n_layers * cfg->dim * kv_dim;    /* wk */
    params += cfg->n_layers * cfg->dim * kv_dim;    /* wv */
    params += cfg->n_layers * cfg->dim * cfg->dim;  /* wo */
    params += cfg->n_layers * cfg->dim * cfg->hidden_dim;  /* w1 */
    params += cfg->n_layers * cfg->hidden_dim * cfg->dim;  /* w2 */
    params += cfg->n_layers * cfg->dim * cfg->hidden_dim;  /* w3 */
    params += cfg->dim;                             /* rms_final_weight */
    params += cfg->dim * cfg->vocab_size;           /* wcls (may be shared) */

    /* Gemma 3 specific */
    if (cfg->use_qk_norm) {
        params += head_size;                        /* q_norm_weight */
        params += head_size;                        /* k_norm_weight */
    }
    params += cfg->n_layers * cfg->dim;             /* post_att_norm_weight */
    params += cfg->n_layers * cfg->dim;             /* post_ffn_norm_weight */

    bytes_per_param = sizeof(float);  /* FP32 */

    return params * bytes_per_param;
}

/* ----------------------------------------------------------------------------
 * Wrapper for RoPE to conform to interface
 * ---------------------------------------------------------------------------- */

static void
gemma3_rope_wrapper(float *q, float *k, int dim, int kv_dim, int pos,
                    int head_size, ModelConfig *cfg)
{
    /* Use local theta by default (most layers are local) */
    float theta = cfg->rope_local_theta > 0 ? cfg->rope_local_theta : GEMMA3_LOCAL_THETA;
    rope_apply_gemma3(q, k, dim, kv_dim, pos, head_size, theta);
}

/* ----------------------------------------------------------------------------
 * Architecture Registration
 * ---------------------------------------------------------------------------- */

static ModelArch gemma3_arch = {
    .name = "gemma3",
    .arch_id = ARCH_GEMMA3,
    .forward = gemma3_forward,
    .apply_rope = gemma3_rope_wrapper,
    .estimate_memory = gemma3_estimate_memory,
};

/* Registration function - call at startup */
void
gemma3_register(void)
{
    arch_register(&gemma3_arch);
}
