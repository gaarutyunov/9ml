/*
 * safetensors.c - Safetensors file format parser
 *
 * Parses HuggingFace safetensors format for model weights.
 */

#include <u.h>
#include <libc.h>

#include "safetensors.h"

/* JSON parsing helpers */

static char *
json_skip_ws(char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
        p++;
    return p;
}

static char *
json_parse_string(char *p, char **out)
{
    char *start, *end;
    int len;

    p = json_skip_ws(p);
    if (*p != '"')
        return nil;

    start = ++p;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1])
            p++;
        p++;
    }
    if (*p != '"')
        return nil;

    end = p++;
    len = end - start;
    *out = malloc(len + 1);
    if (*out) {
        memmove(*out, start, len);
        (*out)[len] = '\0';
    }
    return p;
}

static char *
json_skip_value(char *p)
{
    int depth;

    p = json_skip_ws(p);

    switch (*p) {
    case '"':
        p++;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1])
                p++;
            p++;
        }
        if (*p == '"')
            p++;
        break;

    case '[':
    case '{':
        depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '[' || *p == '{')
                depth++;
            else if (*p == ']' || *p == '}')
                depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\' && p[1])
                        p++;
                    p++;
                }
            }
            if (*p)
                p++;
        }
        break;

    default:
        while (*p && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            p++;
        break;
    }

    return p;
}

/* Parse dtype string */
static int
parse_dtype(char *s)
{
    if (strcmp(s, "F32") == 0) return ST_DTYPE_F32;
    if (strcmp(s, "F16") == 0) return ST_DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return ST_DTYPE_BF16;
    if (strcmp(s, "I8") == 0) return ST_DTYPE_I8;
    if (strcmp(s, "I16") == 0) return ST_DTYPE_I16;
    if (strcmp(s, "I32") == 0) return ST_DTYPE_I32;
    if (strcmp(s, "I64") == 0) return ST_DTYPE_I64;
    if (strcmp(s, "U8") == 0) return ST_DTYPE_U8;
    if (strcmp(s, "U16") == 0) return ST_DTYPE_U16;
    if (strcmp(s, "U32") == 0) return ST_DTYPE_U32;
    if (strcmp(s, "U64") == 0) return ST_DTYPE_U64;
    if (strcmp(s, "BOOL") == 0) return ST_DTYPE_BOOL;
    return -1;
}

/* Parse tensor info from JSON object value */
static STTensor *
parse_tensor_info(char *name, char *json)
{
    STTensor *t;
    char *p, *key, *val;

    t = malloc(sizeof(*t));
    if (t == nil)
        return nil;
    memset(t, 0, sizeof(*t));
    t->name = strdup(name);

    p = json_skip_ws(json);
    if (*p != '{') {
        free(t->name);
        free(t);
        return nil;
    }
    p++;

    while (*p && *p != '}') {
        p = json_skip_ws(p);
        if (*p != '"')
            break;

        p = json_parse_string(p, &key);
        if (p == nil)
            break;

        p = json_skip_ws(p);
        if (*p != ':') {
            free(key);
            break;
        }
        p++;
        p = json_skip_ws(p);

        if (strcmp(key, "dtype") == 0) {
            p = json_parse_string(p, &val);
            if (p && val) {
                t->dtype = parse_dtype(val);
                free(val);
            }
        } else if (strcmp(key, "shape") == 0) {
            /* Parse array of dimensions */
            if (*p == '[') {
                p++;
                t->n_dims = 0;
                while (*p && *p != ']' && t->n_dims < 4) {
                    p = json_skip_ws(p);
                    t->dims[t->n_dims++] = strtoull(p, &p, 10);
                    p = json_skip_ws(p);
                    if (*p == ',')
                        p++;
                }
                if (*p == ']')
                    p++;
            }
        } else if (strcmp(key, "data_offsets") == 0) {
            /* Parse [start, end] array */
            if (*p == '[') {
                p++;
                p = json_skip_ws(p);
                t->offset_start = strtoull(p, &p, 10);
                p = json_skip_ws(p);
                if (*p == ',')
                    p++;
                p = json_skip_ws(p);
                t->offset_end = strtoull(p, &p, 10);
                p = json_skip_ws(p);
                if (*p == ']')
                    p++;
            }
        } else {
            p = json_skip_value(p);
        }

        free(key);
        p = json_skip_ws(p);
        if (*p == ',')
            p++;
    }

    return t;
}

int
st_open(STFile *sf, char *path)
{
    uchar hdr_buf[8];
    char *p, *key;
    STTensor *t, *tail = nil;

    memset(sf, 0, sizeof(*sf));

    sf->fd = open(path, OREAD);
    if (sf->fd < 0)
        return -1;

    /* Read header size (8 bytes little-endian) */
    if (read(sf->fd, hdr_buf, 8) != 8) {
        close(sf->fd);
        return -1;
    }

    sf->header_size = (uvlong)hdr_buf[0] |
                      ((uvlong)hdr_buf[1] << 8) |
                      ((uvlong)hdr_buf[2] << 16) |
                      ((uvlong)hdr_buf[3] << 24) |
                      ((uvlong)hdr_buf[4] << 32) |
                      ((uvlong)hdr_buf[5] << 40) |
                      ((uvlong)hdr_buf[6] << 48) |
                      ((uvlong)hdr_buf[7] << 56);

    if (sf->header_size > 100*1024*1024) {
        /* Sanity check - header shouldn't be > 100MB */
        close(sf->fd);
        return -1;
    }

    /* Read JSON header */
    sf->header_json = malloc(sf->header_size + 1);
    if (sf->header_json == nil) {
        close(sf->fd);
        return -1;
    }

    if (read(sf->fd, sf->header_json, sf->header_size) != sf->header_size) {
        free(sf->header_json);
        close(sf->fd);
        return -1;
    }
    sf->header_json[sf->header_size] = '\0';

    sf->data_offset = 8 + sf->header_size;

    /* Parse JSON header */
    p = json_skip_ws(sf->header_json);
    if (*p != '{') {
        st_close(sf);
        return -1;
    }
    p++;

    while (*p && *p != '}') {
        p = json_skip_ws(p);
        if (*p != '"')
            break;

        p = json_parse_string(p, &key);
        if (p == nil)
            break;

        p = json_skip_ws(p);
        if (*p != ':') {
            free(key);
            break;
        }
        p++;
        p = json_skip_ws(p);

        /* Skip __metadata__ key */
        if (strcmp(key, "__metadata__") == 0) {
            free(key);
            p = json_skip_value(p);
            p = json_skip_ws(p);
            if (*p == ',')
                p++;
            continue;
        }

        /* Parse tensor info */
        if (*p == '{') {
            char *obj_start = p;
            p = json_skip_value(p);

            /* Extract object for parsing */
            int obj_len = p - obj_start;
            char *obj = malloc(obj_len + 1);
            if (obj) {
                memmove(obj, obj_start, obj_len);
                obj[obj_len] = '\0';

                t = parse_tensor_info(key, obj);
                if (t) {
                    if (tail) {
                        tail->next = t;
                        tail = t;
                    } else {
                        sf->tensors = tail = t;
                    }
                    sf->n_tensors++;
                }
                free(obj);
            }
        } else {
            p = json_skip_value(p);
        }

        free(key);
        p = json_skip_ws(p);
        if (*p == ',')
            p++;
    }

    return 0;
}

void
st_close(STFile *sf)
{
    STTensor *t, *next;

    if (sf->fd >= 0)
        close(sf->fd);

    free(sf->header_json);

    for (t = sf->tensors; t != nil; t = next) {
        next = t->next;
        free(t->name);
        free(t);
    }

    memset(sf, 0, sizeof(*sf));
    sf->fd = -1;
}

STTensor *
st_find_tensor(STFile *sf, char *name)
{
    STTensor *t;

    for (t = sf->tensors; t != nil; t = t->next) {
        if (strcmp(t->name, name) == 0)
            return t;
    }
    return nil;
}

STTensor *
st_find_tensors(STFile *sf, char *pattern)
{
    STTensor *t, *match, *head = nil, *tail = nil;

    for (t = sf->tensors; t != nil; t = t->next) {
        if (strstr(t->name, pattern) != nil) {
            match = malloc(sizeof(*match));
            if (match) {
                *match = *t;
                match->name = strdup(t->name);
                match->next = nil;

                if (tail) {
                    tail->next = match;
                    tail = match;
                } else {
                    head = tail = match;
                }
            }
        }
    }

    return head;
}

uvlong
st_tensor_nelements(STTensor *t)
{
    uvlong n = 1;
    int i;

    for (i = 0; i < t->n_dims; i++)
        n *= t->dims[i];
    return n;
}

uvlong
st_tensor_size(STTensor *t)
{
    return t->offset_end - t->offset_start;
}

int
st_dtype_size(int dtype)
{
    switch (dtype) {
    case ST_DTYPE_F32:
    case ST_DTYPE_I32:
    case ST_DTYPE_U32:
        return 4;
    case ST_DTYPE_F16:
    case ST_DTYPE_BF16:
    case ST_DTYPE_I16:
    case ST_DTYPE_U16:
        return 2;
    case ST_DTYPE_I8:
    case ST_DTYPE_U8:
    case ST_DTYPE_BOOL:
        return 1;
    case ST_DTYPE_I64:
    case ST_DTYPE_U64:
        return 8;
    default:
        return 0;
    }
}

char *
st_dtype_name(int dtype)
{
    switch (dtype) {
    case ST_DTYPE_F32: return "F32";
    case ST_DTYPE_F16: return "F16";
    case ST_DTYPE_BF16: return "BF16";
    case ST_DTYPE_I8: return "I8";
    case ST_DTYPE_I16: return "I16";
    case ST_DTYPE_I32: return "I32";
    case ST_DTYPE_I64: return "I64";
    case ST_DTYPE_U8: return "U8";
    case ST_DTYPE_U16: return "U16";
    case ST_DTYPE_U32: return "U32";
    case ST_DTYPE_U64: return "U64";
    case ST_DTYPE_BOOL: return "BOOL";
    default: return "UNKNOWN";
    }
}

/* FP16 to FP32 conversion */
static float
fp16_to_fp32(ushort h)
{
    uint sign = (h >> 15) & 1;
    uint exp = (h >> 10) & 0x1f;
    uint mant = h & 0x3ff;
    uint f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            /* Denormal */
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= 0x3ff;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        /* Inf/NaN */
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    union { uint i; float f; } u;
    u.i = f;
    return u.f;
}

/* BF16 to FP32 conversion */
static float
bf16_to_fp32(ushort h)
{
    union { uint i; float f; } u;
    u.i = (uint)h << 16;
    return u.f;
}

vlong
st_read_tensor(STFile *sf, STTensor *t, float *out, vlong max_floats)
{
    uvlong size = st_tensor_size(t);
    uvlong n = st_tensor_nelements(t);
    void *buf;
    vlong nread;
    uvlong i;

    if ((vlong)n > max_floats)
        n = max_floats;

    /* Seek to tensor data */
    if (seek(sf->fd, sf->data_offset + t->offset_start, 0) < 0)
        return -1;

    switch (t->dtype) {
    case ST_DTYPE_F32:
        /* Direct read */
        nread = read(sf->fd, out, n * 4);
        return nread / 4;

    case ST_DTYPE_F16:
        buf = malloc(n * 2);
        if (buf == nil)
            return -1;
        nread = read(sf->fd, buf, n * 2);
        if (nread > 0) {
            nread /= 2;
            ushort *fp16 = (ushort *)buf;
            for (i = 0; i < (uvlong)nread; i++)
                out[i] = fp16_to_fp32(fp16[i]);
        }
        free(buf);
        return nread;

    case ST_DTYPE_BF16:
        buf = malloc(n * 2);
        if (buf == nil)
            return -1;
        nread = read(sf->fd, buf, n * 2);
        if (nread > 0) {
            nread /= 2;
            ushort *bf16 = (ushort *)buf;
            for (i = 0; i < (uvlong)nread; i++)
                out[i] = bf16_to_fp32(bf16[i]);
        }
        free(buf);
        return nread;

    default:
        /* Unsupported dtype for float conversion */
        return -1;
    }
}

vlong
st_read_tensor_raw(STFile *sf, STTensor *t, void *out, vlong max_bytes)
{
    uvlong size = st_tensor_size(t);

    if ((vlong)size > max_bytes)
        size = max_bytes;

    if (seek(sf->fd, sf->data_offset + t->offset_start, 0) < 0)
        return -1;

    return read(sf->fd, out, size);
}

void
st_tensor_list_free(STTensor *t)
{
    STTensor *next;

    while (t) {
        next = t->next;
        free(t->name);
        free(t);
        t = next;
    }
}

/* ----------------------------------------------------------------------------
 * config.json Parsing
 *
 * Attempts to load model configuration from config.json in the same directory
 * as the safetensors file. This provides model_type and rope_theta values.
 * ---------------------------------------------------------------------------- */

/*
 * Try to load config.json from same directory as model file.
 * Extracts model_type, rope_theta, num_hidden_layers, num_attention_heads,
 * num_key_value_heads, hidden_size, intermediate_size, and max_position_embeddings.
 * Returns 0 on success, -1 if config.json not found or parse error.
 */
static int
load_config_json(char *model_path, int *arch_id, float *rope_theta,
                 int *n_layers, int *n_heads, int *n_kv_heads,
                 int *dim, int *hidden_dim, int *seq_len)
{
    char config_path[256];
    char *slash;
    int fd, n;
    char buf[8192];
    char *p;

    /* Build path to config.json in same directory */
    strncpy(config_path, model_path, sizeof(config_path)-1);
    config_path[sizeof(config_path)-1] = '\0';

    slash = strrchr(config_path, '/');
    if (slash)
        strcpy(slash + 1, "config.json");
    else
        strcpy(config_path, "config.json");

    fd = open(config_path, OREAD);
    if (fd < 0)
        return -1;

    n = read(fd, buf, sizeof(buf)-1);
    close(fd);
    if (n <= 0)
        return -1;
    buf[n] = '\0';

    /* Parse model_type - map to arch_id */
    p = strstr(buf, "\"model_type\"");
    if (p) {
        p = strchr(p + 12, ':');
        if (p) {
            p = strchr(p, '"');
            if (p) {
                p++;
                if (strncmp(p, "llama", 5) == 0)
                    *arch_id = 1;  /* ARCH_LLAMA2 */
            }
        }
    }

    /* Parse rope_theta */
    p = strstr(buf, "\"rope_theta\"");
    if (p) {
        p = strchr(p + 12, ':');
        if (p) {
            while (*p && (*p == ':' || *p == ' ' || *p == '\t'))
                p++;
            *rope_theta = strtod(p, nil);
        }
    }

    /* Parse num_hidden_layers */
    p = strstr(buf, "\"num_hidden_layers\"");
    if (p) {
        p = strchr(p + 19, ':');
        if (p) {
            while (*p && (*p == ':' || *p == ' ' || *p == '\t'))
                p++;
            *n_layers = strtol(p, nil, 10);
        }
    }

    /* Parse num_attention_heads */
    p = strstr(buf, "\"num_attention_heads\"");
    if (p) {
        p = strchr(p + 21, ':');
        if (p) {
            while (*p && (*p == ':' || *p == ' ' || *p == '\t'))
                p++;
            *n_heads = strtol(p, nil, 10);
        }
    }

    /* Parse num_key_value_heads */
    p = strstr(buf, "\"num_key_value_heads\"");
    if (p) {
        p = strchr(p + 21, ':');
        if (p) {
            while (*p && (*p == ':' || *p == ' ' || *p == '\t'))
                p++;
            *n_kv_heads = strtol(p, nil, 10);
        }
    }

    /* Parse hidden_size (dim) */
    p = strstr(buf, "\"hidden_size\"");
    if (p) {
        p = strchr(p + 13, ':');
        if (p) {
            while (*p && (*p == ':' || *p == ' ' || *p == '\t'))
                p++;
            *dim = strtol(p, nil, 10);
        }
    }

    /* Parse intermediate_size (hidden_dim) */
    p = strstr(buf, "\"intermediate_size\"");
    if (p) {
        p = strchr(p + 19, ':');
        if (p) {
            while (*p && (*p == ':' || *p == ' ' || *p == '\t'))
                p++;
            *hidden_dim = strtol(p, nil, 10);
        }
    }

    /* Parse max_position_embeddings (seq_len) */
    p = strstr(buf, "\"max_position_embeddings\"");
    if (p) {
        p = strchr(p + 25, ':');
        if (p) {
            while (*p && (*p == ':' || *p == ' ' || *p == '\t'))
                p++;
            *seq_len = strtol(p, nil, 10);
        }
    }

    return 0;
}

/* ----------------------------------------------------------------------------
 * Safetensors to Transformer Loading
 * ---------------------------------------------------------------------------- */

/*
 * Helper to load a single tensor by name into a float buffer.
 * Returns number of floats loaded, or -1 on error.
 */
static vlong
st_load_tensor_by_name(STFile *sf, char *name, float *out, vlong max_floats)
{
    STTensor *t = st_find_tensor(sf, name);
    if (t == nil) {
        fprint(2, "safetensors: tensor not found: %s\n", name);
        return -1;
    }
    return st_read_tensor(sf, t, out, max_floats);
}

/*
 * Load safetensors file into 9ml Transformer structure.
 * Allocates memory and converts weights to FP32.
 *
 * Note: Uses void* for transformer to keep safetensors.h decoupled from arch/arch.h.
 */
int
st_load_transformer(char *path, void *transformer)
{
    /* Local type definitions matching Transformer layout in arch/arch.h */
    typedef struct {
        int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
        float rope_theta;
        int arch_id;
    } TConfig;

    typedef struct {
        float *token_embedding_table;
        float *rms_att_weight;
        float *rms_ffn_weight;
        float *wq, *wk, *wv, *wo;
        float *w1, *w2, *w3;
        float *rms_final_weight;
        float *wcls;
    } TWeights;

    /* RunState has 12 float pointers for activation buffers */
    typedef struct {
        float *x, *xb, *xb2, *hb, *hb2;
        float *q, *k, *v;
        float *att, *logits;
        float *key_cache, *value_cache;
    } TRunState;

    typedef struct {
        TConfig config;
        TWeights weights;
        TRunState state;     /* Full struct, not pointer! */
        void *arch;
        float *data;
        vlong file_size;
    } TTransformer;

    TTransformer *t = (TTransformer *)transformer;
    STFile sf;
    STTensor *emb_tensor;
    vlong weights_size;
    float *weights_data;
    float *ptr;
    int l;
    char name[128];

    /* Initialize config with defaults */
    int vocab_size = 0, dim = 0, hidden_dim = 0, n_layers = 0;
    int n_heads = 0, n_kv_heads = 0, seq_len = 256;
    float rope_theta = 10000.0f;
    int arch_id = 1;  /* ARCH_LLAMA2 */

    /* Try loading config.json first for model parameters */
    if (load_config_json(path, &arch_id, &rope_theta,
                         &n_layers, &n_heads, &n_kv_heads,
                         &dim, &hidden_dim, &seq_len) == 0) {
        /* Config loaded - values that remain 0 will be inferred from tensors */
    }

    if (st_open(&sf, path) < 0) {
        return -1;
    }

    /* Extract model configuration from tensor shapes for missing values */
    /* lm_head.weight shape is [vocab_size, dim] */
    emb_tensor = st_find_tensor(&sf, "lm_head.weight");
    if (emb_tensor == nil) {
        fprint(2, "safetensors: missing lm_head.weight\n");
        st_close(&sf);
        return -1;
    }

    if (emb_tensor->n_dims < 2) {
        fprint(2, "safetensors: invalid lm_head.weight dimensions\n");
        st_close(&sf);
        return -1;
    }

    /* Always get vocab_size from tensor (not in config.json) */
    vocab_size = (int)emb_tensor->dims[0];

    /* Use tensor dim if not from config.json */
    if (dim == 0)
        dim = (int)emb_tensor->dims[1];

    /* Count layers if not from config.json */
    if (n_layers == 0) {
        for (l = 0; l < 256; l++) {
            snprint(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
            if (st_find_tensor(&sf, name) == nil)
                break;
            n_layers++;
        }
    }

    if (n_layers == 0) {
        fprint(2, "safetensors: no layers found\n");
        st_close(&sf);
        return -1;
    }

    /* Get hidden_dim from tensor if not from config.json */
    if (hidden_dim == 0) {
        snprint(name, sizeof(name), "model.layers.0.mlp.gate_proj.weight");
        STTensor *w1_tensor = st_find_tensor(&sf, name);
        if (w1_tensor == nil || w1_tensor->n_dims < 2) {
            fprint(2, "safetensors: missing or invalid gate_proj.weight\n");
            st_close(&sf);
            return -1;
        }
        hidden_dim = (int)w1_tensor->dims[0];
    }

    /* Verify q_proj exists */
    snprint(name, sizeof(name), "model.layers.0.self_attn.q_proj.weight");
    STTensor *wq_tensor = st_find_tensor(&sf, name);
    if (wq_tensor == nil || wq_tensor->n_dims < 2) {
        fprint(2, "safetensors: missing q_proj.weight\n");
        st_close(&sf);
        return -1;
    }

    /* If n_heads not from config.json, infer from dim */
    if (n_heads == 0) {
        int hs = 48;  /* Common default */
        if (dim == 288) hs = 48;
        else if (dim == 4096) hs = 128;  /* LLaMA 7B/13B */
        else hs = dim / 8;  /* Guess 8 heads */
        n_heads = dim / hs;
    }

    /* If n_kv_heads not set, assume MHA (same as n_heads) */
    if (n_kv_heads == 0)
        n_kv_heads = n_heads;

    /* Calculate head_size from n_heads (needed for weight sizes) */
    int head_size = dim / n_heads;

    /* Set config */
    t->config.dim = dim;
    t->config.hidden_dim = hidden_dim;
    t->config.n_layers = n_layers;
    t->config.n_heads = n_heads;
    t->config.n_kv_heads = n_kv_heads;
    t->config.vocab_size = vocab_size;
    t->config.seq_len = seq_len;
    t->config.rope_theta = rope_theta;
    t->config.arch_id = arch_id;

    /* Calculate total weights size (all FP32) */
    int kv_dim = (dim * n_kv_heads) / n_heads;
    uvlong ul_layers = n_layers;

    weights_size = 0;
    weights_size += (uvlong)vocab_size * dim;           /* token_embedding_table */
    weights_size += ul_layers * dim;                    /* rms_att_weight */
    weights_size += ul_layers * dim * (n_heads * head_size);  /* wq */
    weights_size += ul_layers * dim * kv_dim;           /* wk */
    weights_size += ul_layers * dim * kv_dim;           /* wv */
    weights_size += ul_layers * (n_heads * head_size) * dim;  /* wo */
    weights_size += ul_layers * dim;                    /* rms_ffn_weight */
    weights_size += ul_layers * dim * hidden_dim;       /* w1 */
    weights_size += ul_layers * hidden_dim * dim;       /* w2 */
    weights_size += ul_layers * dim * hidden_dim;       /* w3 */
    weights_size += dim;                                /* rms_final_weight */

    /* Allocate weights buffer */
    weights_data = malloc(weights_size * sizeof(float));
    if (weights_data == nil) {
        fprint(2, "safetensors: failed to allocate %lld bytes\n",
               weights_size * sizeof(float));
        st_close(&sf);
        return -1;
    }
    t->data = weights_data;
    t->file_size = weights_size * sizeof(float);

    /* Map weight pointers into the buffer */
    ptr = weights_data;

    /* token_embedding_table - use lm_head.weight (tied embeddings) */
    t->weights.token_embedding_table = ptr;
    if (st_load_tensor_by_name(&sf, "lm_head.weight", ptr,
                               (vlong)vocab_size * dim) < 0) {
        free(weights_data);
        st_close(&sf);
        return -1;
    }
    ptr += (uvlong)vocab_size * dim;

    /* rms_att_weight (per layer) */
    t->weights.rms_att_weight = ptr;
    for (l = 0; l < n_layers; l++) {
        snprint(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        if (st_load_tensor_by_name(&sf, name, ptr, dim) < 0) {
            free(weights_data);
            st_close(&sf);
            return -1;
        }
        ptr += dim;
    }

    /* wq (per layer) */
    t->weights.wq = ptr;
    for (l = 0; l < n_layers; l++) {
        snprint(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
        if (st_load_tensor_by_name(&sf, name, ptr,
                                   (vlong)dim * (n_heads * head_size)) < 0) {
            free(weights_data);
            st_close(&sf);
            return -1;
        }
        ptr += (uvlong)dim * (n_heads * head_size);
    }

    /* wk (per layer) */
    t->weights.wk = ptr;
    for (l = 0; l < n_layers; l++) {
        snprint(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", l);
        if (st_load_tensor_by_name(&sf, name, ptr, (vlong)dim * kv_dim) < 0) {
            free(weights_data);
            st_close(&sf);
            return -1;
        }
        ptr += (uvlong)dim * kv_dim;
    }

    /* wv (per layer) */
    t->weights.wv = ptr;
    for (l = 0; l < n_layers; l++) {
        snprint(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", l);
        if (st_load_tensor_by_name(&sf, name, ptr, (vlong)dim * kv_dim) < 0) {
            free(weights_data);
            st_close(&sf);
            return -1;
        }
        ptr += (uvlong)dim * kv_dim;
    }

    /* wo (per layer) */
    t->weights.wo = ptr;
    for (l = 0; l < n_layers; l++) {
        snprint(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
        if (st_load_tensor_by_name(&sf, name, ptr,
                                   (vlong)(n_heads * head_size) * dim) < 0) {
            free(weights_data);
            st_close(&sf);
            return -1;
        }
        ptr += (uvlong)(n_heads * head_size) * dim;
    }

    /* rms_ffn_weight (per layer) */
    t->weights.rms_ffn_weight = ptr;
    for (l = 0; l < n_layers; l++) {
        snprint(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        if (st_load_tensor_by_name(&sf, name, ptr, dim) < 0) {
            free(weights_data);
            st_close(&sf);
            return -1;
        }
        ptr += dim;
    }

    /* w1 - gate_proj (per layer) */
    t->weights.w1 = ptr;
    for (l = 0; l < n_layers; l++) {
        snprint(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", l);
        if (st_load_tensor_by_name(&sf, name, ptr,
                                   (vlong)dim * hidden_dim) < 0) {
            free(weights_data);
            st_close(&sf);
            return -1;
        }
        ptr += (uvlong)dim * hidden_dim;
    }

    /* w2 - down_proj (per layer) */
    t->weights.w2 = ptr;
    for (l = 0; l < n_layers; l++) {
        snprint(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", l);
        if (st_load_tensor_by_name(&sf, name, ptr,
                                   (vlong)hidden_dim * dim) < 0) {
            free(weights_data);
            st_close(&sf);
            return -1;
        }
        ptr += (uvlong)hidden_dim * dim;
    }

    /* w3 - up_proj (per layer) */
    t->weights.w3 = ptr;
    for (l = 0; l < n_layers; l++) {
        snprint(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", l);
        if (st_load_tensor_by_name(&sf, name, ptr,
                                   (vlong)dim * hidden_dim) < 0) {
            free(weights_data);
            st_close(&sf);
            return -1;
        }
        ptr += (uvlong)dim * hidden_dim;
    }

    /* rms_final_weight */
    t->weights.rms_final_weight = ptr;
    if (st_load_tensor_by_name(&sf, "model.norm.weight", ptr, dim) < 0) {
        free(weights_data);
        st_close(&sf);
        return -1;
    }
    ptr += dim;

    /* wcls - shared with token_embedding_table (tied embeddings) */
    t->weights.wcls = t->weights.token_embedding_table;

    st_close(&sf);
    return 0;
}
