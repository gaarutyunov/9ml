/*
 * arch/arch.h - Model architecture plugin interface for 9ml
 *
 * This module defines the interface for model architecture plugins.
 * Currently only LLaMA 2 architecture is supported.
 *
 * NOTE: Types are designed to match model.c's existing types for compatibility.
 * model.c includes this header and uses these types via typedefs.
 */

/* Forward declarations */
typedef struct ModelArch ModelArch;

/* ----------------------------------------------------------------------------
 * Model Configuration
 *
 * Matches model.c Config struct layout for binary compatibility.
 * ---------------------------------------------------------------------------- */

/* FFN activation types */
enum {
    FFN_SWIGLU = 0,    /* SiLU(x) * W3(x) - LLaMA, Mistral */
    FFN_GEGLU  = 1,    /* GELU(x) * W3(x) */
    FFN_GELU   = 2,    /* GELU(x) */
};

/* Architecture IDs */
enum {
    ARCH_UNKNOWN = 0,
    ARCH_LLAMA2  = 1,
};

typedef struct {
    /* Original 7 fields (28 bytes) - matches file format */
    int dim;            /* transformer dimension */
    int hidden_dim;     /* for ffn layers */
    int n_layers;       /* number of layers */
    int n_heads;        /* number of query heads */
    int n_kv_heads;     /* number of key/value heads (GQA support) */
    int vocab_size;     /* vocabulary size */
    int seq_len;        /* max sequence length */

    /* Extended fields */
    float rope_theta;   /* RoPE base frequency: 10000 (LLaMA2), 500000 (LLaMA3) */
    int arch_id;        /* architecture ID for plugin dispatch */
} ModelConfig;

/* Backward compatibility typedef */
typedef ModelConfig Config;

/* Default values for extended fields */
#define DEFAULT_ROPE_THETA    10000.0f
#define DEFAULT_FFN_TYPE      FFN_SWIGLU
#define DEFAULT_ATTN_BIAS     0
#define DEFAULT_MLP_BIAS      0
#define DEFAULT_SLIDING_WINDOW 0

/* ----------------------------------------------------------------------------
 * Transformer Weights and State
 *
 * These match model.c definitions for binary compatibility.
 * ---------------------------------------------------------------------------- */

/* Transformer weights (FP32) */
typedef struct {
    float *token_embedding_table;  /* (vocab_size, dim) */
    float *rms_att_weight;         /* (layer, dim) */
    float *rms_ffn_weight;         /* (layer, dim) */
    float *wq;                     /* (layer, dim, n_heads * head_size) */
    float *wk;                     /* (layer, dim, n_kv_heads * head_size) */
    float *wv;                     /* (layer, dim, n_kv_heads * head_size) */
    float *wo;                     /* (layer, n_heads * head_size, dim) */
    float *w1;                     /* (layer, hidden_dim, dim) */
    float *w2;                     /* (layer, dim, hidden_dim) */
    float *w3;                     /* (layer, hidden_dim, dim) */
    float *rms_final_weight;       /* (dim,) */
    float *wcls;                   /* classifier weights (may alias token_embedding_table) */
} TransformerWeights;

/* Run-time state (activation buffers) */
typedef struct {
    float *x;            /* activation at current time stamp (dim,) */
    float *xb;           /* inside residual branch (dim,) */
    float *xb2;          /* additional buffer (dim,) */
    float *hb;           /* hidden dimension ffn (hidden_dim,) */
    float *hb2;          /* hidden dimension ffn (hidden_dim,) */
    float *q;            /* query (dim,) */
    float *k;            /* key (kv_dim,) - may be less than dim for GQA */
    float *v;            /* value (kv_dim,) */
    float *att;          /* attention scores (n_heads, seq_len) */
    float *logits;       /* output logits (vocab_size,) */
    float *key_cache;    /* (layer, seq_len, kv_dim) */
    float *value_cache;  /* (layer, seq_len, kv_dim) */
} RunState;

/* ----------------------------------------------------------------------------
 * Model Instance (Transformer)
 *
 * A loaded model with weights, state, and architecture binding.
 * Layout matches model.c's Transformer struct.
 * ---------------------------------------------------------------------------- */

typedef struct Transformer Transformer;
struct Transformer {
    ModelConfig config;
    TransformerWeights weights;
    RunState state;
    ModelArch *arch;       /* bound architecture plugin */
    float *data;           /* loaded weights data (to free) */
    vlong file_size;
};

/* Alias for plugin code that prefers ModelInstance name */
typedef Transformer ModelInstance;

/* ----------------------------------------------------------------------------
 * Model Architecture Plugin Interface
 *
 * Each architecture (LLaMA 2, LLaMA 3, Mistral, etc.) implements this
 * interface. The plugin handles architecture-specific details like RoPE
 * frequency computation and attention patterns.
 * ---------------------------------------------------------------------------- */

struct ModelArch {
    char *name;                /* "llama2" */
    int arch_id;               /* ARCH_LLAMA2 */

    /* Forward pass - returns logits */
    float *(*forward)(Transformer *t, int token, int pos);

    /* Apply RoPE to query and key vectors */
    void (*apply_rope)(float *q, float *k, int dim, int kv_dim, int pos,
                       int head_size, ModelConfig *cfg);

    /* Estimate memory usage in bytes */
    uvlong (*estimate_memory)(ModelConfig *cfg, int quant_type);
};

/* ----------------------------------------------------------------------------
 * Architecture Registry
 * ---------------------------------------------------------------------------- */

/* Initialize architecture subsystem - registers all built-in architectures */
void arch_init(void);

/* Register an architecture plugin */
void arch_register(ModelArch *arch);

/* Find architecture by name */
ModelArch *arch_find(char *name);

/* Find architecture by ID */
ModelArch *arch_find_by_id(int arch_id);

/* List registered architectures */
int arch_list(ModelArch **out, int max);

/* ----------------------------------------------------------------------------
 * Standard RoPE Implementation
 *
 * Shared implementation used by LLaMA 2, LLaMA 3, Mistral.
 * ---------------------------------------------------------------------------- */

void rope_apply_standard(float *q, float *k, int dim, int kv_dim, int pos,
                         int head_size, ModelConfig *cfg);
