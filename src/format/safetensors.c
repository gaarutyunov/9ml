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
