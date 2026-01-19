/*
 * format/gguf.c - GGUF file format parser for 9ml
 *
 * Parses GGUF (GGML Universal Format) files used by llama.cpp.
 * Supports Q4_0 and Q8_0 quantization formats.
 */

#include <u.h>
#include <libc.h>
#include "gguf.h"

/* ----------------------------------------------------------------------------
 * FP16 Conversion
 * ---------------------------------------------------------------------------- */

float
fp16_to_fp32(ushort h)
{
    uint sign = (h >> 15) & 1;
    uint exp = (h >> 10) & 0x1f;
    uint mant = h & 0x3ff;
    uint f;

    if (exp == 0) {
        if (mant == 0) {
            /* Zero */
            f = sign << 31;
        } else {
            /* Denormalized */
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x400;
            exp += 127 - 15;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        /* Inf or NaN */
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        /* Normalized */
        exp += 127 - 15;
        f = (sign << 31) | (exp << 23) | (mant << 13);
    }

    union { uint i; float f; } u;
    u.i = f;
    return u.f;
}

/* ----------------------------------------------------------------------------
 * Dequantization
 * ---------------------------------------------------------------------------- */

void
dequant_q4_0(float *out, BlockQ4_0 *block)
{
    float d = fp16_to_fp32(block->d);
    int i;

    for (i = 0; i < QK4_0/2; i++) {
        /* Each byte contains two 4-bit values */
        int x0 = (block->qs[i] & 0x0f) - 8;  /* low nibble, subtract 8 for signed */
        int x1 = (block->qs[i] >> 4) - 8;    /* high nibble, subtract 8 for signed */
        out[i]          = x0 * d;
        out[i + QK4_0/2] = x1 * d;
    }
}

void
dequant_q8_0(float *out, BlockQ8_0 *block)
{
    float d = fp16_to_fp32(block->d);
    int i;

    for (i = 0; i < QK8_0; i++) {
        out[i] = block->qs[i] * d;
    }
}

/* ----------------------------------------------------------------------------
 * GGUF String Reading
 * ---------------------------------------------------------------------------- */

static int
read_string(int fd, GGUFString *s)
{
    vlong n;

    n = read(fd, &s->len, sizeof(uvlong));
    if (n != sizeof(uvlong)) return -1;

    s->data = malloc(s->len + 1);
    if (s->data == nil) return -1;

    if (s->len > 0) {
        n = read(fd, s->data, s->len);
        if (n != (vlong)s->len) {
            free(s->data);
            s->data = nil;
            return -1;
        }
    }
    s->data[s->len] = '\0';  /* null-terminate for convenience */
    return 0;
}

static void
free_string(GGUFString *s)
{
    if (s->data) {
        free(s->data);
        s->data = nil;
    }
    s->len = 0;
}

/* ----------------------------------------------------------------------------
 * GGUF Metadata Reading
 * ---------------------------------------------------------------------------- */

static int
read_metadata_value(int fd, GGUFMetadata *m)
{
    vlong n;
    uvlong i;

    n = read(fd, &m->type, sizeof(uint));
    if (n != sizeof(uint)) return -1;

    switch (m->type) {
    case GGUF_TYPE_UINT8:
        n = read(fd, &m->value.u8, 1);
        if (n != 1) return -1;
        break;
    case GGUF_TYPE_INT8:
        n = read(fd, &m->value.i8, 1);
        if (n != 1) return -1;
        break;
    case GGUF_TYPE_UINT16:
        n = read(fd, &m->value.u16, 2);
        if (n != 2) return -1;
        break;
    case GGUF_TYPE_INT16:
        n = read(fd, &m->value.i16, 2);
        if (n != 2) return -1;
        break;
    case GGUF_TYPE_UINT32:
        n = read(fd, &m->value.u32, 4);
        if (n != 4) return -1;
        break;
    case GGUF_TYPE_INT32:
        n = read(fd, &m->value.i32, 4);
        if (n != 4) return -1;
        break;
    case GGUF_TYPE_FLOAT32:
        n = read(fd, &m->value.f32, 4);
        if (n != 4) return -1;
        break;
    case GGUF_TYPE_BOOL:
        n = read(fd, &m->value.b, 1);
        if (n != 1) return -1;
        break;
    case GGUF_TYPE_STRING:
        if (read_string(fd, &m->value.str) < 0) return -1;
        break;
    case GGUF_TYPE_UINT64:
        n = read(fd, &m->value.u64, 8);
        if (n != 8) return -1;
        break;
    case GGUF_TYPE_INT64:
        n = read(fd, &m->value.i64, 8);
        if (n != 8) return -1;
        break;
    case GGUF_TYPE_FLOAT64:
        n = read(fd, &m->value.f64, 8);
        if (n != 8) return -1;
        break;
    case GGUF_TYPE_ARRAY:
        n = read(fd, &m->value.arr.type, sizeof(uint));
        if (n != sizeof(uint)) return -1;
        n = read(fd, &m->value.arr.len, sizeof(uvlong));
        if (n != sizeof(uvlong)) return -1;
        /* For simplicity, skip array data for now (we don't need it for model loading) */
        /* TODO: implement array reading if needed */
        m->value.arr.data = nil;
        for (i = 0; i < m->value.arr.len; i++) {
            GGUFMetadata dummy;
            dummy.type = m->value.arr.type;
            /* Read and discard array elements */
            switch (m->value.arr.type) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_BOOL:
                seek(fd, 1, 1);
                break;
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
                seek(fd, 2, 1);
                break;
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
                seek(fd, 4, 1);
                break;
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64:
                seek(fd, 8, 1);
                break;
            case GGUF_TYPE_STRING:
                {
                    GGUFString skip_str;
                    if (read_string(fd, &skip_str) < 0) return -1;
                    free_string(&skip_str);
                }
                break;
            default:
                fprint(2, "gguf: unsupported array element type %d\n", m->value.arr.type);
                return -1;
            }
        }
        break;
    default:
        fprint(2, "gguf: unknown metadata type %d\n", m->type);
        return -1;
    }

    return 0;
}

static void
free_metadata(GGUFMetadata *m)
{
    free_string(&m->key);
    if (m->type == GGUF_TYPE_STRING) {
        free_string(&m->value.str);
    }
    /* Note: array data is not currently allocated */
}

/* ----------------------------------------------------------------------------
 * GGUF Tensor Info Reading
 * ---------------------------------------------------------------------------- */

static int
read_tensor_info(int fd, GGUFTensorInfo *t)
{
    vlong n;
    uint i;

    if (read_string(fd, &t->name) < 0) return -1;

    n = read(fd, &t->n_dims, sizeof(uint));
    if (n != sizeof(uint)) return -1;

    if (t->n_dims > 4) {
        fprint(2, "gguf: tensor has too many dimensions: %d\n", t->n_dims);
        return -1;
    }

    for (i = 0; i < t->n_dims; i++) {
        n = read(fd, &t->dims[i], sizeof(uvlong));
        if (n != sizeof(uvlong)) return -1;
    }
    for (; i < 4; i++) {
        t->dims[i] = 1;
    }

    n = read(fd, &t->type, sizeof(uint));
    if (n != sizeof(uint)) return -1;

    n = read(fd, &t->offset, sizeof(uvlong));
    if (n != sizeof(uvlong)) return -1;

    return 0;
}

static void
free_tensor_info(GGUFTensorInfo *t)
{
    free_string(&t->name);
}

/* ----------------------------------------------------------------------------
 * GGUF File Operations
 * ---------------------------------------------------------------------------- */

int
gguf_open(GGUFFile *gf, char *path)
{
    Dir *d;
    vlong n;
    uvlong i;

    memset(gf, 0, sizeof(GGUFFile));

    gf->fd = open(path, OREAD);
    if (gf->fd < 0) {
        fprint(2, "gguf: cannot open %s\n", path);
        return -1;
    }

    d = dirfstat(gf->fd);
    if (d == nil) {
        close(gf->fd);
        return -1;
    }
    gf->file_size = d->length;
    free(d);

    /* Read header */
    n = read(gf->fd, &gf->magic, sizeof(uint));
    if (n != sizeof(uint) || gf->magic != GGUF_MAGIC) {
        fprint(2, "gguf: invalid magic number (expected GGUF)\n");
        close(gf->fd);
        return -1;
    }

    n = read(gf->fd, &gf->version, sizeof(uint));
    if (n != sizeof(uint)) {
        fprint(2, "gguf: failed to read version\n");
        close(gf->fd);
        return -1;
    }

    if (gf->version < GGUF_VERSION_MIN || gf->version > GGUF_VERSION_MAX) {
        fprint(2, "gguf: unsupported version %d (supported: %d-%d)\n",
               gf->version, GGUF_VERSION_MIN, GGUF_VERSION_MAX);
        close(gf->fd);
        return -1;
    }

    n = read(gf->fd, &gf->n_tensors, sizeof(uvlong));
    if (n != sizeof(uvlong)) {
        fprint(2, "gguf: failed to read tensor count\n");
        close(gf->fd);
        return -1;
    }

    n = read(gf->fd, &gf->n_kv, sizeof(uvlong));
    if (n != sizeof(uvlong)) {
        fprint(2, "gguf: failed to read metadata count\n");
        close(gf->fd);
        return -1;
    }

    /* Allocate and read metadata */
    if (gf->n_kv > 0) {
        gf->metadata = malloc(gf->n_kv * sizeof(GGUFMetadata));
        if (gf->metadata == nil) {
            close(gf->fd);
            return -1;
        }
        memset(gf->metadata, 0, gf->n_kv * sizeof(GGUFMetadata));

        for (i = 0; i < gf->n_kv; i++) {
            if (read_string(gf->fd, &gf->metadata[i].key) < 0) {
                fprint(2, "gguf: failed to read metadata key %ulld\n", i);
                gguf_close(gf);
                return -1;
            }
            if (read_metadata_value(gf->fd, &gf->metadata[i]) < 0) {
                fprint(2, "gguf: failed to read metadata value for %s\n",
                       gf->metadata[i].key.data);
                gguf_close(gf);
                return -1;
            }
        }
    }

    /* Allocate and read tensor info */
    if (gf->n_tensors > 0) {
        gf->tensors = malloc(gf->n_tensors * sizeof(GGUFTensorInfo));
        if (gf->tensors == nil) {
            gguf_close(gf);
            return -1;
        }
        memset(gf->tensors, 0, gf->n_tensors * sizeof(GGUFTensorInfo));

        for (i = 0; i < gf->n_tensors; i++) {
            if (read_tensor_info(gf->fd, &gf->tensors[i]) < 0) {
                fprint(2, "gguf: failed to read tensor info %ulld\n", i);
                gguf_close(gf);
                return -1;
            }
        }
    }

    /* Record current position as end of header/start of tensor data
     * GGUF requires alignment to GGUF_DEFAULT_ALIGNMENT (32 bytes) */
    gf->header_size = seek(gf->fd, 0, 1);  /* current position */

    /* Align to 32 bytes */
    uvlong alignment = 32;
    uvlong aligned = (gf->header_size + alignment - 1) & ~(alignment - 1);
    gf->tensor_data_offset = aligned;

    return 0;
}

void
gguf_close(GGUFFile *gf)
{
    uvlong i;

    if (gf->metadata) {
        for (i = 0; i < gf->n_kv; i++) {
            free_metadata(&gf->metadata[i]);
        }
        free(gf->metadata);
        gf->metadata = nil;
    }

    if (gf->tensors) {
        for (i = 0; i < gf->n_tensors; i++) {
            free_tensor_info(&gf->tensors[i]);
        }
        free(gf->tensors);
        gf->tensors = nil;
    }

    if (gf->fd >= 0) {
        close(gf->fd);
        gf->fd = -1;
    }
}

/* ----------------------------------------------------------------------------
 * Metadata Access
 * ---------------------------------------------------------------------------- */

GGUFMetadata *
gguf_find_metadata(GGUFFile *gf, char *key)
{
    uvlong i;

    for (i = 0; i < gf->n_kv; i++) {
        if (strcmp(gf->metadata[i].key.data, key) == 0) {
            return &gf->metadata[i];
        }
    }
    return nil;
}

int
gguf_get_int(GGUFFile *gf, char *key, int def)
{
    GGUFMetadata *m = gguf_find_metadata(gf, key);
    if (m == nil) return def;

    switch (m->type) {
    case GGUF_TYPE_UINT8:  return (int)m->value.u8;
    case GGUF_TYPE_INT8:   return (int)m->value.i8;
    case GGUF_TYPE_UINT16: return (int)m->value.u16;
    case GGUF_TYPE_INT16:  return (int)m->value.i16;
    case GGUF_TYPE_UINT32: return (int)m->value.u32;
    case GGUF_TYPE_INT32:  return m->value.i32;
    case GGUF_TYPE_UINT64: return (int)m->value.u64;
    case GGUF_TYPE_INT64:  return (int)m->value.i64;
    default: return def;
    }
}

float
gguf_get_float(GGUFFile *gf, char *key, float def)
{
    GGUFMetadata *m = gguf_find_metadata(gf, key);
    if (m == nil) return def;

    switch (m->type) {
    case GGUF_TYPE_FLOAT32: return m->value.f32;
    case GGUF_TYPE_FLOAT64: return (float)m->value.f64;
    case GGUF_TYPE_UINT32:  return (float)m->value.u32;
    case GGUF_TYPE_INT32:   return (float)m->value.i32;
    default: return def;
    }
}

char *
gguf_get_string(GGUFFile *gf, char *key)
{
    GGUFMetadata *m = gguf_find_metadata(gf, key);
    if (m == nil || m->type != GGUF_TYPE_STRING) return nil;

    char *copy = malloc(m->value.str.len + 1);
    if (copy == nil) return nil;
    memmove(copy, m->value.str.data, m->value.str.len + 1);
    return copy;
}

/* ----------------------------------------------------------------------------
 * Tensor Access
 * ---------------------------------------------------------------------------- */

GGUFTensorInfo *
gguf_find_tensor(GGUFFile *gf, char *name)
{
    uvlong i;

    for (i = 0; i < gf->n_tensors; i++) {
        if (strcmp(gf->tensors[i].name.data, name) == 0) {
            return &gf->tensors[i];
        }
    }
    return nil;
}

/* Get tensor element count */
static uvlong
tensor_nelements(GGUFTensorInfo *t)
{
    uvlong n = 1;
    uint i;
    for (i = 0; i < t->n_dims; i++) {
        n *= t->dims[i];
    }
    return n;
}

/* Get tensor size in bytes based on type and element count */
static uvlong
tensor_size_bytes(GGUFTensorInfo *t)
{
    uvlong n = tensor_nelements(t);

    switch (t->type) {
    case GGML_TYPE_F32:
        return n * 4;
    case GGML_TYPE_F16:
        return n * 2;
    case GGML_TYPE_Q4_0:
        /* Q4_0: 32 elements per block, 18 bytes per block (2 scale + 16 data) */
        return (n / QK4_0) * BLOCK_Q4_0_SIZE;
    case GGML_TYPE_Q8_0:
        /* Q8_0: 32 elements per block, 34 bytes per block (2 scale + 32 data) */
        return (n / QK8_0) * BLOCK_Q8_0_SIZE;
    default:
        fprint(2, "gguf: unsupported tensor type %d\n", t->type);
        return 0;
    }
}

vlong
gguf_dequant_tensor(GGUFFile *gf, GGUFTensorInfo *tensor, float *out, vlong max_floats)
{
    uvlong n = tensor_nelements(tensor);
    uvlong offset = gf->tensor_data_offset + tensor->offset;
    uvlong to_read, read_size;
    vlong nread;
    void *buf;
    uvlong i;

    /* Limit output to max_floats */
    if ((vlong)n > max_floats) {
        n = (uvlong)max_floats;
    }

    /* Calculate bytes to read based on type and element count
     * For quantized types, round up to whole blocks */
    switch (tensor->type) {
    case GGML_TYPE_F32:
        read_size = n * 4;
        break;
    case GGML_TYPE_F16:
        read_size = n * 2;
        break;
    case GGML_TYPE_Q4_0:
        /* Round up to whole blocks */
        to_read = ((n + QK4_0 - 1) / QK4_0) * QK4_0;
        read_size = (to_read / QK4_0) * BLOCK_Q4_0_SIZE;
        break;
    case GGML_TYPE_Q8_0:
        /* Round up to whole blocks */
        to_read = ((n + QK8_0 - 1) / QK8_0) * QK8_0;
        read_size = (to_read / QK8_0) * BLOCK_Q8_0_SIZE;
        break;
    default:
        fprint(2, "gguf: unsupported tensor type %d for dequantization\n", tensor->type);
        return -1;
    }

    if (read_size == 0) {
        return 0;
    }

    /* Seek to tensor data */
    if (seek(gf->fd, offset, 0) != (vlong)offset) {
        fprint(2, "gguf: failed to seek to tensor at %ulld\n", offset);
        return -1;
    }

    /* Allocate buffer for quantized data */
    buf = malloc(read_size);
    if (buf == nil) return -1;

    /* Read quantized data */
    nread = read(gf->fd, buf, read_size);
    if (nread != (vlong)read_size) {
        fprint(2, "gguf: failed to read tensor data (wanted %ulld, got %lld)\n", read_size, nread);
        free(buf);
        return -1;
    }

    /* Dequantize based on type */
    switch (tensor->type) {
    case GGML_TYPE_F32:
        /* Already FP32, just copy */
        memmove(out, buf, n * sizeof(float));
        break;

    case GGML_TYPE_F16:
        /* Convert FP16 to FP32 */
        {
            ushort *fp16 = (ushort*)buf;
            for (i = 0; i < n; i++) {
                out[i] = fp16_to_fp32(fp16[i]);
            }
        }
        break;

    case GGML_TYPE_Q4_0:
        {
            uvlong nblocks = (n + QK4_0 - 1) / QK4_0;
            uchar *ptr = (uchar*)buf;
            float *p = out;
            for (i = 0; i < nblocks; i++) {
                /* Parse packed block: 2 bytes scale + 16 bytes data */
                BlockQ4_0 block;
                block.d = ptr[0] | (ptr[1] << 8);  /* little-endian */
                memmove(block.qs, ptr + 2, QK4_0/2);
                ptr += BLOCK_Q4_0_SIZE;

                float tmp[QK4_0];
                dequant_q4_0(tmp, &block);
                uvlong copy = QK4_0;
                if (p - out + copy > n) copy = n - (p - out);
                memmove(p, tmp, copy * sizeof(float));
                p += copy;
            }
        }
        break;

    case GGML_TYPE_Q8_0:
        {
            uvlong nblocks = (n + QK8_0 - 1) / QK8_0;
            uchar *ptr = (uchar*)buf;
            float *p = out;
            for (i = 0; i < nblocks; i++) {
                /* Parse packed block: 2 bytes scale + 32 bytes data */
                BlockQ8_0 block;
                block.d = ptr[0] | (ptr[1] << 8);  /* little-endian */
                memmove(block.qs, ptr + 2, QK8_0);
                ptr += BLOCK_Q8_0_SIZE;

                float tmp[QK8_0];
                dequant_q8_0(tmp, &block);
                uvlong copy = QK8_0;
                if (p - out + copy > n) copy = n - (p - out);
                memmove(p, tmp, copy * sizeof(float));
                p += copy;
            }
        }
        break;

    default:
        /* Should not reach here */
        free(buf);
        return -1;
    }

    free(buf);
    return (vlong)n;
}

/* ----------------------------------------------------------------------------
 * Model Configuration Extraction
 * ---------------------------------------------------------------------------- */

int
gguf_get_model_config(GGUFFile *gf, GGUFModelConfig *cfg)
{
    char *arch;

    memset(cfg, 0, sizeof(GGUFModelConfig));

    /* Get architecture name */
    arch = gguf_get_string(gf, "general.architecture");
    if (arch == nil) {
        fprint(2, "gguf: missing general.architecture\n");
        return -1;
    }
    strncpy(cfg->arch_name, arch, sizeof(cfg->arch_name) - 1);
    free(arch);

    /* Build key prefix based on architecture */
    char key[128];

    /* Embedding dimension */
    snprint(key, sizeof(key), "%s.embedding_length", cfg->arch_name);
    cfg->dim = gguf_get_int(gf, key, 0);
    if (cfg->dim == 0) {
        fprint(2, "gguf: missing %s\n", key);
        return -1;
    }

    /* Feed-forward hidden dimension */
    snprint(key, sizeof(key), "%s.feed_forward_length", cfg->arch_name);
    cfg->hidden_dim = gguf_get_int(gf, key, 0);
    if (cfg->hidden_dim == 0) {
        /* Some models use intermediate_size instead */
        snprint(key, sizeof(key), "%s.intermediate_size", cfg->arch_name);
        cfg->hidden_dim = gguf_get_int(gf, key, 0);
    }

    /* Number of layers */
    snprint(key, sizeof(key), "%s.block_count", cfg->arch_name);
    cfg->n_layers = gguf_get_int(gf, key, 0);
    if (cfg->n_layers == 0) {
        fprint(2, "gguf: missing %s\n", key);
        return -1;
    }

    /* Number of attention heads */
    snprint(key, sizeof(key), "%s.attention.head_count", cfg->arch_name);
    cfg->n_heads = gguf_get_int(gf, key, 0);
    if (cfg->n_heads == 0) {
        fprint(2, "gguf: missing %s\n", key);
        return -1;
    }

    /* Number of KV heads (for GQA) */
    snprint(key, sizeof(key), "%s.attention.head_count_kv", cfg->arch_name);
    cfg->n_kv_heads = gguf_get_int(gf, key, cfg->n_heads);  /* default to n_heads */

    /* Vocab size - from model metadata or tensor dimensions */
    /* Try architecture-specific key first (e.g., "llama.vocab_size") */
    snprint(key, sizeof(key), "%s.vocab_size", cfg->arch_name);
    cfg->vocab_size = gguf_get_int(gf, key, 0);
    if (cfg->vocab_size == 0) {
        /* Try tokenizer metadata */
        cfg->vocab_size = gguf_get_int(gf, "tokenizer.ggml.vocab_size", 0);
    }
    if (cfg->vocab_size == 0) {
        /* Try to infer from token_embd tensor.
         * GGUF stores dimensions in Fortran order (column-major), so
         * dims[1] is vocab_size and dims[0] is embedding_dim */
        GGUFTensorInfo *embd = gguf_find_tensor(gf, "token_embd.weight");
        if (embd && embd->n_dims >= 2) {
            cfg->vocab_size = (int)embd->dims[1];
        } else if (embd && embd->n_dims >= 1) {
            /* Fallback for 1D tensors */
            cfg->vocab_size = (int)embd->dims[0];
        }
    }

    /* Context length */
    snprint(key, sizeof(key), "%s.context_length", cfg->arch_name);
    cfg->seq_len = gguf_get_int(gf, key, 2048);

    /* RoPE theta */
    snprint(key, sizeof(key), "%s.rope.freq_base", cfg->arch_name);
    cfg->rope_theta = gguf_get_float(gf, key, 10000.0f);

    /* Determine architecture ID */
    if (strcmp(cfg->arch_name, "llama") == 0) {
        if (cfg->rope_theta > 100000.0f) {
            cfg->arch_id = 2;  /* ARCH_LLAMA3 */
        } else {
            cfg->arch_id = 1;  /* ARCH_LLAMA2 */
        }
    } else if (strcmp(cfg->arch_name, "mistral") == 0) {
        cfg->arch_id = 3;  /* ARCH_MISTRAL */
    } else {
        cfg->arch_id = 0;  /* ARCH_UNKNOWN */
    }

    return 0;
}

/* ----------------------------------------------------------------------------
 * GGUF to Transformer Loading
 * ---------------------------------------------------------------------------- */

/*
 * De-interleave Q/K attention weights.
 *
 * llama.cpp's converter applies an interleaving transformation to Q and K
 * weights for rotary position embeddings. Within each head of head_dim rows:
 *   - GGUF rows 0,2,4,... contain original rows 0,1,2,...
 *   - GGUF rows 1,3,5,... contain original rows head_dim/2, head_dim/2+1,...
 *
 * This function reverses that transformation in-place.
 *
 * Parameters:
 *   weights - pointer to weight matrix (n_heads * head_dim * dim)
 *   n_heads - number of attention heads
 *   head_dim - dimension per head (dim / n_heads)
 *   dim - embedding dimension (row stride)
 */
static void
deinterleave_qk_weights(float *weights, int n_heads, int head_dim, int dim)
{
    float *tmp;
    int h, i;
    int half = head_dim / 2;

    /* Allocate temp buffer for one head's worth of rows */
    tmp = malloc(head_dim * dim * sizeof(float));
    if (tmp == nil) {
        fprint(2, "gguf: failed to allocate deinterleave buffer\n");
        return;
    }

    /* Process each head */
    for (h = 0; h < n_heads; h++) {
        float *head_weights = weights + h * head_dim * dim;

        /* Copy interleaved data to temp buffer */
        memmove(tmp, head_weights, head_dim * dim * sizeof(float));

        /* De-interleave back to original order:
         * Output row i (for i < half) comes from GGUF row 2*i
         * Output row i (for i >= half) comes from GGUF row 2*(i-half)+1 */
        for (i = 0; i < half; i++) {
            /* First half: output row i from interleaved row 2*i */
            memmove(head_weights + i * dim, tmp + (2 * i) * dim, dim * sizeof(float));
        }
        for (i = half; i < head_dim; i++) {
            /* Second half: output row i from interleaved row 2*(i-half)+1 */
            memmove(head_weights + i * dim, tmp + (2 * (i - half) + 1) * dim, dim * sizeof(float));
        }
    }

    free(tmp);
}

/*
 * Helper to load a single tensor by name into a float buffer.
 * Returns number of floats loaded, or -1 on error.
 * Set quiet=1 to suppress "tensor not found" message (for fallback lookups).
 */
static vlong
load_tensor_by_name_q(GGUFFile *gf, char *name, float *out, vlong max_floats, int quiet)
{
    GGUFTensorInfo *t = gguf_find_tensor(gf, name);
    if (t == nil) {
        if (!quiet)
            fprint(2, "gguf: tensor not found: %s\n", name);
        return -1;
    }
    return gguf_dequant_tensor(gf, t, out, max_floats);
}

/* Convenience wrapper - always prints error on failure */
static vlong
load_tensor_by_name(GGUFFile *gf, char *name, float *out, vlong max_floats)
{
    return load_tensor_by_name_q(gf, name, out, max_floats, 0);
}

/*
 * Load GGUF file into 9ml Transformer structure.
 * This allocates memory and dequantizes all weights to FP32.
 *
 * Note: We use void* for the transformer parameter to keep gguf.h
 * decoupled from arch/arch.h. The caller passes a Transformer*.
 */
int
gguf_load_transformer(char *path, void *transformer)
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
    GGUFFile gf;
    GGUFModelConfig cfg;
    vlong weights_size;
    float *weights_data;
    float *ptr;
    int l;
    char name[128];

    /*
     * Disable Plan 9 FP exceptions during dequantization.
     * GGUF files may contain denormalized FP16 scale values that
     * trigger FPINVAL when used in arithmetic operations.
     */
    setfcr(getfcr() & ~(FPINVAL|FPZDIV|FPOVFL|FPUNFL|FPINEX));

    if (gguf_open(&gf, path) < 0) {
        return -1;
    }

    /* Extract model configuration */
    if (gguf_get_model_config(&gf, &cfg) < 0) {
        gguf_close(&gf);
        return -1;
    }

    /* Copy config to transformer */
    t->config.dim = cfg.dim;
    t->config.hidden_dim = cfg.hidden_dim;
    t->config.n_layers = cfg.n_layers;
    t->config.n_heads = cfg.n_heads;
    t->config.n_kv_heads = cfg.n_kv_heads;
    t->config.vocab_size = cfg.vocab_size;
    t->config.seq_len = cfg.seq_len;
    t->config.rope_theta = cfg.rope_theta;
    t->config.arch_id = cfg.arch_id;

    /* Calculate total weights size (all FP32) */
    int head_size = cfg.dim / cfg.n_heads;
    int kv_dim = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;
    uvlong n_layers = cfg.n_layers;

    weights_size = 0;
    weights_size += (uvlong)cfg.vocab_size * cfg.dim;           /* token_embedding_table */
    weights_size += n_layers * cfg.dim;                          /* rms_att_weight */
    weights_size += n_layers * cfg.dim * (cfg.n_heads * head_size);  /* wq */
    weights_size += n_layers * cfg.dim * kv_dim;                 /* wk */
    weights_size += n_layers * cfg.dim * kv_dim;                 /* wv */
    weights_size += n_layers * (cfg.n_heads * head_size) * cfg.dim;  /* wo */
    weights_size += n_layers * cfg.dim;                          /* rms_ffn_weight */
    weights_size += n_layers * cfg.dim * cfg.hidden_dim;         /* w1 */
    weights_size += n_layers * cfg.hidden_dim * cfg.dim;         /* w2 */
    weights_size += n_layers * cfg.dim * cfg.hidden_dim;         /* w3 */
    weights_size += cfg.dim;                                     /* rms_final_weight */
    /* wcls may be shared with token_embedding_table */

    /* Allocate weights buffer */
    weights_data = malloc(weights_size * sizeof(float));
    if (weights_data == nil) {
        fprint(2, "gguf: failed to allocate %lld bytes for weights\n",
               weights_size * sizeof(float));
        gguf_close(&gf);
        return -1;
    }
    t->data = weights_data;
    t->file_size = weights_size * sizeof(float);

    /* Map weight pointers into the buffer */
    ptr = weights_data;

    /* token_embedding_table
     * Try token_embd.weight first, fall back to output.weight for tied embeddings */
    t->weights.token_embedding_table = ptr;
    if (load_tensor_by_name_q(&gf, "token_embd.weight", ptr,
                              (vlong)cfg.vocab_size * cfg.dim, 1) < 0) {
        /* Try output.weight as fallback (tied embeddings case) */
        if (load_tensor_by_name(&gf, "output.weight", ptr,
                                (vlong)cfg.vocab_size * cfg.dim) < 0) {
            fprint(2, "gguf: neither token_embd.weight nor output.weight found\n");
            free(weights_data);
            gguf_close(&gf);
            return -1;
        }
    }
    ptr += (uvlong)cfg.vocab_size * cfg.dim;

    /* rms_att_weight (per layer) */
    t->weights.rms_att_weight = ptr;
    for (l = 0; l < cfg.n_layers; l++) {
        snprint(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        if (load_tensor_by_name(&gf, name, ptr, cfg.dim) < 0) {
            free(weights_data);
            gguf_close(&gf);
            return -1;
        }
        ptr += cfg.dim;
    }

    /* wq (per layer) */
    t->weights.wq = ptr;
    for (l = 0; l < cfg.n_layers; l++) {
        snprint(name, sizeof(name), "blk.%d.attn_q.weight", l);
        if (load_tensor_by_name(&gf, name, ptr,
                                (vlong)cfg.dim * (cfg.n_heads * head_size)) < 0) {
            free(weights_data);
            gguf_close(&gf);
            return -1;
        }
        /* De-interleave Q weights (llama.cpp interleaves for RoPE) */
        deinterleave_qk_weights(ptr, cfg.n_heads, head_size, cfg.dim);
        ptr += (uvlong)cfg.dim * (cfg.n_heads * head_size);
    }

    /* wk (per layer) */
    t->weights.wk = ptr;
    for (l = 0; l < cfg.n_layers; l++) {
        snprint(name, sizeof(name), "blk.%d.attn_k.weight", l);
        if (load_tensor_by_name(&gf, name, ptr, (vlong)cfg.dim * kv_dim) < 0) {
            free(weights_data);
            gguf_close(&gf);
            return -1;
        }
        /* De-interleave K weights (llama.cpp interleaves for RoPE) */
        deinterleave_qk_weights(ptr, cfg.n_kv_heads, head_size, cfg.dim);
        ptr += (uvlong)cfg.dim * kv_dim;
    }

    /* wv (per layer) */
    t->weights.wv = ptr;
    for (l = 0; l < cfg.n_layers; l++) {
        snprint(name, sizeof(name), "blk.%d.attn_v.weight", l);
        if (load_tensor_by_name(&gf, name, ptr, (vlong)cfg.dim * kv_dim) < 0) {
            free(weights_data);
            gguf_close(&gf);
            return -1;
        }
        ptr += (uvlong)cfg.dim * kv_dim;
    }

    /* wo (per layer) */
    t->weights.wo = ptr;
    for (l = 0; l < cfg.n_layers; l++) {
        snprint(name, sizeof(name), "blk.%d.attn_output.weight", l);
        if (load_tensor_by_name(&gf, name, ptr,
                                (vlong)(cfg.n_heads * head_size) * cfg.dim) < 0) {
            free(weights_data);
            gguf_close(&gf);
            return -1;
        }
        ptr += (uvlong)(cfg.n_heads * head_size) * cfg.dim;
    }

    /* rms_ffn_weight (per layer) */
    t->weights.rms_ffn_weight = ptr;
    for (l = 0; l < cfg.n_layers; l++) {
        snprint(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        if (load_tensor_by_name(&gf, name, ptr, cfg.dim) < 0) {
            free(weights_data);
            gguf_close(&gf);
            return -1;
        }
        ptr += cfg.dim;
    }

    /* w1 - gate_proj (per layer) */
    t->weights.w1 = ptr;
    for (l = 0; l < cfg.n_layers; l++) {
        snprint(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        if (load_tensor_by_name(&gf, name, ptr,
                                (vlong)cfg.dim * cfg.hidden_dim) < 0) {
            free(weights_data);
            gguf_close(&gf);
            return -1;
        }
        ptr += (uvlong)cfg.dim * cfg.hidden_dim;
    }

    /* w2 - down_proj (per layer) */
    t->weights.w2 = ptr;
    for (l = 0; l < cfg.n_layers; l++) {
        snprint(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        if (load_tensor_by_name(&gf, name, ptr,
                                (vlong)cfg.hidden_dim * cfg.dim) < 0) {
            free(weights_data);
            gguf_close(&gf);
            return -1;
        }
        ptr += (uvlong)cfg.hidden_dim * cfg.dim;
    }

    /* w3 - up_proj (per layer) */
    t->weights.w3 = ptr;
    for (l = 0; l < cfg.n_layers; l++) {
        snprint(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        if (load_tensor_by_name(&gf, name, ptr,
                                (vlong)cfg.dim * cfg.hidden_dim) < 0) {
            free(weights_data);
            gguf_close(&gf);
            return -1;
        }
        ptr += (uvlong)cfg.dim * cfg.hidden_dim;
    }

    /* rms_final_weight */
    t->weights.rms_final_weight = ptr;
    if (load_tensor_by_name(&gf, "output_norm.weight", ptr, cfg.dim) < 0) {
        free(weights_data);
        gguf_close(&gf);
        return -1;
    }
    ptr += cfg.dim;

    /* wcls - output projection (may be shared with token_embd) */
    /* Check if output.weight exists; if not, share with token_embedding_table */
    GGUFTensorInfo *output_tensor = gguf_find_tensor(&gf, "output.weight");
    if (output_tensor != nil) {
        /* Separate output weights - need to allocate more space */
        /* For now, we'll just share with token_embedding_table */
        /* TODO: handle non-tied embeddings if needed */
        t->weights.wcls = t->weights.token_embedding_table;
    } else {
        /* Tied embeddings */
        t->weights.wcls = t->weights.token_embedding_table;
    }

    gguf_close(&gf);
    return 0;
}
