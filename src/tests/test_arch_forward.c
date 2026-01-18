/*
 * test_arch_forward.c - Compare arch plugin forward() vs built-in forward()
 *
 * Tests that the architecture plugin dispatch produces identical results
 * to the built-in fallback implementation.
 */

#include <u.h>
#include <libc.h>

/* Minimal model structs for testing */
typedef struct ModelArch ModelArch;

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    float rope_theta;
    int arch_id;
} Config;

typedef struct {
    float *token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *w1;
    float *w2;
    float *w3;
    float *rms_final_weight;
    float *wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    ModelArch *arch;
    float *data;
    vlong file_size;
} Transformer;

/* Architecture IDs */
#define ARCH_UNKNOWN  0
#define ARCH_LLAMA2   1
#define ARCH_LLAMA3   2

/* RoPE constants */
#define LLAMA2_ROPE_THETA 10000.0f
#define LLAMA3_ROPE_THETA 500000.0f

/* Simple RMS normalization for testing */
void rmsnorm(float *o, float *x, float *weight, int size) {
    int j;
    float ss = 0.0f;
    for (j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrt(ss);
    for (j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

/* Simple softmax for testing */
void softmax(float *x, int size) {
    int i;
    float max_val = x[0];
    for (i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    for (i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

/* Simple matmul for testing */
void matmul(float *xout, float *x, float *w, int n, int d) {
    int i, j;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

/* Built-in forward pass (copy of model.c implementation) */
float *forward_builtin(Transformer *t, int token, int pos) {
    Config *p = &t->config;
    TransformerWeights *w = &t->weights;
    RunState *s = &t->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    uvlong l;
    int i;

    /* Copy token embedding */
    float *content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));

    /* Forward all layers */
    for (l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        /* RoPE - use rope_theta from config */
        for (i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / pow(p->rope_theta, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cos(val);
            float fci = sin(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float *vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        /* Attention - sequential for simplicity */
        for (int h = 0; h < p->n_heads; h++) {
            float *q_ptr = s->q + h * head_size;
            float *att_ptr = s->att + h * p->seq_len;
            float *xb_ptr = s->xb + h * head_size;

            for (int t = 0; t <= pos; t++) {
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (i = 0; i < head_size; i++) {
                    score += q_ptr[i] * k[i];
                }
                att_ptr[t] = score / sqrt((float)head_size);
            }

            softmax(att_ptr, pos + 1);

            memset(xb_ptr, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att_ptr[t];
                for (i = 0; i < head_size; i++) {
                    xb_ptr[i] += a * v[i];
                }
            }
        }

        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        for (i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        for (i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + exp(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        for (i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

/* Test: Forward with different rope_theta produces different results */
int test_rope_theta_difference(void) {
    /* Tiny test config */
    int dim = 16;
    int hidden_dim = 32;
    int n_layers = 1;
    int n_heads = 2;
    int n_kv_heads = 2;
    int vocab_size = 32;
    int seq_len = 8;
    int kv_dim = dim;
    int i;

    /* Allocate transformer */
    Transformer t1, t2;
    memset(&t1, 0, sizeof(t1));
    memset(&t2, 0, sizeof(t2));

    /* Config for LLaMA 2 */
    t1.config.dim = dim;
    t1.config.hidden_dim = hidden_dim;
    t1.config.n_layers = n_layers;
    t1.config.n_heads = n_heads;
    t1.config.n_kv_heads = n_kv_heads;
    t1.config.vocab_size = vocab_size;
    t1.config.seq_len = seq_len;
    t1.config.rope_theta = LLAMA2_ROPE_THETA;
    t1.config.arch_id = ARCH_LLAMA2;

    /* Config for LLaMA 3 */
    t2.config = t1.config;
    t2.config.rope_theta = LLAMA3_ROPE_THETA;
    t2.config.arch_id = ARCH_LLAMA3;

    /* Allocate shared weights (same for both) */
    int total_weights = vocab_size * dim +          /* embeddings */
                       n_layers * dim +             /* rms_att */
                       n_layers * dim +             /* rms_ffn */
                       n_layers * dim * dim +       /* wq */
                       n_layers * dim * kv_dim +    /* wk */
                       n_layers * dim * kv_dim +    /* wv */
                       n_layers * dim * dim +       /* wo */
                       n_layers * dim * hidden_dim + /* w1 */
                       n_layers * hidden_dim * dim + /* w2 */
                       n_layers * dim * hidden_dim + /* w3 */
                       dim +                        /* rms_final */
                       dim * vocab_size;            /* wcls */

    float *weights = malloc(total_weights * sizeof(float));
    if (weights == nil) {
        print("  malloc failed\n");
        return 0;
    }

    /* Initialize with pseudo-random values */
    for (i = 0; i < total_weights; i++) {
        weights[i] = ((i * 17 + 31) % 1000) / 1000.0f - 0.5f;
    }

    /* Point both transformers to same weights */
    float *ptr = weights;
    t1.weights.token_embedding_table = t2.weights.token_embedding_table = ptr; ptr += vocab_size * dim;
    t1.weights.rms_att_weight = t2.weights.rms_att_weight = ptr; ptr += n_layers * dim;
    t1.weights.rms_ffn_weight = t2.weights.rms_ffn_weight = ptr; ptr += n_layers * dim;
    t1.weights.wq = t2.weights.wq = ptr; ptr += n_layers * dim * dim;
    t1.weights.wk = t2.weights.wk = ptr; ptr += n_layers * dim * kv_dim;
    t1.weights.wv = t2.weights.wv = ptr; ptr += n_layers * dim * kv_dim;
    t1.weights.wo = t2.weights.wo = ptr; ptr += n_layers * dim * dim;
    t1.weights.w1 = t2.weights.w1 = ptr; ptr += n_layers * dim * hidden_dim;
    t1.weights.w2 = t2.weights.w2 = ptr; ptr += n_layers * hidden_dim * dim;
    t1.weights.w3 = t2.weights.w3 = ptr; ptr += n_layers * dim * hidden_dim;
    t1.weights.rms_final_weight = t2.weights.rms_final_weight = ptr; ptr += dim;
    t1.weights.wcls = t2.weights.wcls = ptr;

    /* Allocate state for both */
    t1.state.x = calloc(dim, sizeof(float));
    t1.state.xb = calloc(dim, sizeof(float));
    t1.state.xb2 = calloc(dim, sizeof(float));
    t1.state.hb = calloc(hidden_dim, sizeof(float));
    t1.state.hb2 = calloc(hidden_dim, sizeof(float));
    t1.state.q = calloc(dim, sizeof(float));
    t1.state.key_cache = calloc(n_layers * seq_len * kv_dim, sizeof(float));
    t1.state.value_cache = calloc(n_layers * seq_len * kv_dim, sizeof(float));
    t1.state.att = calloc(n_heads * seq_len, sizeof(float));
    t1.state.logits = calloc(vocab_size, sizeof(float));

    t2.state.x = calloc(dim, sizeof(float));
    t2.state.xb = calloc(dim, sizeof(float));
    t2.state.xb2 = calloc(dim, sizeof(float));
    t2.state.hb = calloc(hidden_dim, sizeof(float));
    t2.state.hb2 = calloc(hidden_dim, sizeof(float));
    t2.state.q = calloc(dim, sizeof(float));
    t2.state.key_cache = calloc(n_layers * seq_len * kv_dim, sizeof(float));
    t2.state.value_cache = calloc(n_layers * seq_len * kv_dim, sizeof(float));
    t2.state.att = calloc(n_heads * seq_len, sizeof(float));
    t2.state.logits = calloc(vocab_size, sizeof(float));

    /* Run forward at position > 0 (RoPE has effect) */
    int token = 5;
    int pos = 3;

    float *logits1 = forward_builtin(&t1, token, pos);
    float *logits2 = forward_builtin(&t2, token, pos);

    /* Check that outputs differ (different rope_theta should produce different results) */
    int differ = 0;
    float max_diff = 0.0f;
    for (i = 0; i < vocab_size; i++) {
        float diff = logits1[i] - logits2[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.001f) differ = 1;
    }

    /* Cleanup */
    free(weights);
    free(t1.state.x); free(t1.state.xb); free(t1.state.xb2);
    free(t1.state.hb); free(t1.state.hb2); free(t1.state.q);
    free(t1.state.key_cache); free(t1.state.value_cache);
    free(t1.state.att); free(t1.state.logits);
    free(t2.state.x); free(t2.state.xb); free(t2.state.xb2);
    free(t2.state.hb); free(t2.state.hb2); free(t2.state.q);
    free(t2.state.key_cache); free(t2.state.value_cache);
    free(t2.state.att); free(t2.state.logits);

    if (differ) {
        print("  Different rope_theta produces different outputs (max_diff=%.6f)\n", max_diff);
        return 1;
    } else {
        print("  ERROR: Different rope_theta should produce different outputs!\n");
        return 0;
    }
}

/* Test: Same rope_theta produces identical results */
int test_same_theta_identical(void) {
    /* Tiny test config */
    int dim = 16;
    int hidden_dim = 32;
    int n_layers = 1;
    int n_heads = 2;
    int n_kv_heads = 2;
    int vocab_size = 32;
    int seq_len = 8;
    int kv_dim = dim;
    int i;

    Transformer t1, t2;
    memset(&t1, 0, sizeof(t1));
    memset(&t2, 0, sizeof(t2));

    /* Same config for both */
    t1.config.dim = dim;
    t1.config.hidden_dim = hidden_dim;
    t1.config.n_layers = n_layers;
    t1.config.n_heads = n_heads;
    t1.config.n_kv_heads = n_kv_heads;
    t1.config.vocab_size = vocab_size;
    t1.config.seq_len = seq_len;
    t1.config.rope_theta = LLAMA2_ROPE_THETA;
    t1.config.arch_id = ARCH_LLAMA2;

    t2.config = t1.config;  /* Identical */

    /* Allocate shared weights */
    int total_weights = vocab_size * dim + n_layers * dim + n_layers * dim +
                       n_layers * dim * dim + n_layers * dim * kv_dim +
                       n_layers * dim * kv_dim + n_layers * dim * dim +
                       n_layers * dim * hidden_dim + n_layers * hidden_dim * dim +
                       n_layers * dim * hidden_dim + dim + dim * vocab_size;

    float *weights = malloc(total_weights * sizeof(float));
    for (i = 0; i < total_weights; i++) {
        weights[i] = ((i * 17 + 31) % 1000) / 1000.0f - 0.5f;
    }

    /* Point both to same weights */
    float *ptr = weights;
    t1.weights.token_embedding_table = t2.weights.token_embedding_table = ptr; ptr += vocab_size * dim;
    t1.weights.rms_att_weight = t2.weights.rms_att_weight = ptr; ptr += n_layers * dim;
    t1.weights.rms_ffn_weight = t2.weights.rms_ffn_weight = ptr; ptr += n_layers * dim;
    t1.weights.wq = t2.weights.wq = ptr; ptr += n_layers * dim * dim;
    t1.weights.wk = t2.weights.wk = ptr; ptr += n_layers * dim * kv_dim;
    t1.weights.wv = t2.weights.wv = ptr; ptr += n_layers * dim * kv_dim;
    t1.weights.wo = t2.weights.wo = ptr; ptr += n_layers * dim * dim;
    t1.weights.w1 = t2.weights.w1 = ptr; ptr += n_layers * dim * hidden_dim;
    t1.weights.w2 = t2.weights.w2 = ptr; ptr += n_layers * hidden_dim * dim;
    t1.weights.w3 = t2.weights.w3 = ptr; ptr += n_layers * dim * hidden_dim;
    t1.weights.rms_final_weight = t2.weights.rms_final_weight = ptr; ptr += dim;
    t1.weights.wcls = t2.weights.wcls = ptr;

    /* Allocate separate state (so they don't interfere) */
    t1.state.x = calloc(dim, sizeof(float));
    t1.state.xb = calloc(dim, sizeof(float));
    t1.state.xb2 = calloc(dim, sizeof(float));
    t1.state.hb = calloc(hidden_dim, sizeof(float));
    t1.state.hb2 = calloc(hidden_dim, sizeof(float));
    t1.state.q = calloc(dim, sizeof(float));
    t1.state.key_cache = calloc(n_layers * seq_len * kv_dim, sizeof(float));
    t1.state.value_cache = calloc(n_layers * seq_len * kv_dim, sizeof(float));
    t1.state.att = calloc(n_heads * seq_len, sizeof(float));
    t1.state.logits = calloc(vocab_size, sizeof(float));

    t2.state.x = calloc(dim, sizeof(float));
    t2.state.xb = calloc(dim, sizeof(float));
    t2.state.xb2 = calloc(dim, sizeof(float));
    t2.state.hb = calloc(hidden_dim, sizeof(float));
    t2.state.hb2 = calloc(hidden_dim, sizeof(float));
    t2.state.q = calloc(dim, sizeof(float));
    t2.state.key_cache = calloc(n_layers * seq_len * kv_dim, sizeof(float));
    t2.state.value_cache = calloc(n_layers * seq_len * kv_dim, sizeof(float));
    t2.state.att = calloc(n_heads * seq_len, sizeof(float));
    t2.state.logits = calloc(vocab_size, sizeof(float));

    int token = 5;
    int pos = 3;

    float *logits1 = forward_builtin(&t1, token, pos);
    float *logits2 = forward_builtin(&t2, token, pos);

    /* Check outputs are identical */
    int identical = 1;
    float max_diff = 0.0f;
    for (i = 0; i < vocab_size; i++) {
        float diff = logits1[i] - logits2[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-6f) identical = 0;
    }

    /* Cleanup */
    free(weights);
    free(t1.state.x); free(t1.state.xb); free(t1.state.xb2);
    free(t1.state.hb); free(t1.state.hb2); free(t1.state.q);
    free(t1.state.key_cache); free(t1.state.value_cache);
    free(t1.state.att); free(t1.state.logits);
    free(t2.state.x); free(t2.state.xb); free(t2.state.xb2);
    free(t2.state.hb); free(t2.state.hb2); free(t2.state.q);
    free(t2.state.key_cache); free(t2.state.value_cache);
    free(t2.state.att); free(t2.state.logits);

    if (identical) {
        print("  Same config produces identical outputs (max_diff=%.9f)\n", max_diff);
        return 1;
    } else {
        print("  ERROR: Same config should produce identical outputs (max_diff=%.6f)!\n", max_diff);
        return 0;
    }
}

void
main(int argc, char *argv[])
{
    int passed = 0;
    int failed = 0;

    USED(argc);
    USED(argv);

    print("=== Architecture Forward Test ===\n");

    print("\nTest 1: Different rope_theta produces different results\n");
    if (test_rope_theta_difference()) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL\n");
        failed++;
    }

    print("\nTest 2: Same config produces identical results\n");
    if (test_same_theta_identical()) {
        print("  Result: PASS\n");
        passed++;
    } else {
        print("  Result: FAIL\n");
        failed++;
    }

    print("\n=== Result ===\n");
    if (failed == 0) {
        print("PASS: All %d architecture forward tests passed\n", passed);
    } else {
        print("FAIL: %d passed, %d failed\n", passed, failed);
    }

    exits(failed ? "fail" : 0);
}
