/*
 * format/gguf.h - GGUF file format parser for 9ml
 *
 * GGUF (GGML Universal Format) is the model format used by llama.cpp.
 * This module parses GGUF files and converts them to 9ml's internal format.
 *
 * Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */

#ifndef GGUF_H
#define GGUF_H

/* GGUF magic number: "GGUF" in little-endian */
#define GGUF_MAGIC 0x46554747  /* "GGUF" */

/* GGUF versions we support */
#define GGUF_VERSION_MIN 2
#define GGUF_VERSION_MAX 3

/* GGUF metadata value types */
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* GGML tensor types (quantization formats) */
enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_BF16    = 29,
};

/* Q4_0 block: 32 weights quantized to 4 bits each = 16 bytes + 2 byte scale */
#define QK4_0 32
#define BLOCK_Q4_0_SIZE 18  /* On-disk size: 2 (FP16 scale) + 16 (packed 4-bit) */
typedef struct {
    ushort d;           /* delta (FP16 scale) */
    uchar qs[QK4_0/2];  /* 4-bit quantized values (packed) */
} BlockQ4_0;

/* Q8_0 block: 32 weights quantized to 8 bits each = 32 bytes + 2 byte scale */
#define QK8_0 32
#define BLOCK_Q8_0_SIZE 34  /* On-disk size: 2 (FP16 scale) + 32 (8-bit values) */
typedef struct {
    ushort d;           /* delta (FP16 scale) */
    schar qs[QK8_0];    /* 8-bit quantized values */
} BlockQ8_0;

/* GGUF string (length-prefixed) */
typedef struct {
    uvlong len;
    char *data;         /* NOT null-terminated in file, we null-terminate when reading */
} GGUFString;

/* GGUF metadata key-value pair */
typedef struct {
    GGUFString key;
    uint type;
    union {
        uchar   u8;
        schar   i8;
        ushort  u16;
        short   i16;
        uint    u32;
        int     i32;
        float   f32;
        uchar   b;      /* bool */
        GGUFString str;
        uvlong  u64;
        vlong   i64;
        double  f64;
        struct {
            uint type;
            uvlong len;
            void *data;
        } arr;
    } value;
} GGUFMetadata;

/* GGUF tensor info */
typedef struct {
    GGUFString name;
    uint n_dims;
    uvlong dims[4];     /* max 4 dimensions */
    uint type;          /* GGML_TYPE_* */
    uvlong offset;      /* offset from start of tensor data */
} GGUFTensorInfo;

/* GGUF file header and parsed content */
typedef struct {
    uint magic;
    uint version;
    uvlong n_tensors;
    uvlong n_kv;

    /* Parsed metadata */
    GGUFMetadata *metadata;

    /* Parsed tensor info */
    GGUFTensorInfo *tensors;

    /* Offsets for data access */
    uvlong header_size;      /* size of header + metadata + tensor info */
    uvlong tensor_data_offset;  /* offset to start of tensor data */

    /* File handle for reading tensor data */
    int fd;
    uvlong file_size;
} GGUFFile;

/* Model config extracted from GGUF metadata */
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
    char arch_name[64];
} GGUFModelConfig;

/* ----------------------------------------------------------------------------
 * GGUF Parsing API
 * ---------------------------------------------------------------------------- */

/* Open and parse GGUF file header, metadata, and tensor info */
int gguf_open(GGUFFile *gf, char *path);

/* Close GGUF file and free resources */
void gguf_close(GGUFFile *gf);

/* Find metadata by key (returns NULL if not found) */
GGUFMetadata *gguf_find_metadata(GGUFFile *gf, char *key);

/* Get metadata value as specific type (returns default if not found or wrong type) */
int gguf_get_int(GGUFFile *gf, char *key, int def);
float gguf_get_float(GGUFFile *gf, char *key, float def);
char *gguf_get_string(GGUFFile *gf, char *key);  /* returns malloc'd copy */

/* Find tensor by name (returns NULL if not found) */
GGUFTensorInfo *gguf_find_tensor(GGUFFile *gf, char *name);

/* Extract model configuration from GGUF metadata */
int gguf_get_model_config(GGUFFile *gf, GGUFModelConfig *cfg);

/* ----------------------------------------------------------------------------
 * Dequantization API
 * ---------------------------------------------------------------------------- */

/* Convert FP16 to FP32 */
float fp16_to_fp32(ushort h);

/* Dequantize Q4_0 block to FP32 */
void dequant_q4_0(float *out, BlockQ4_0 *block);

/* Dequantize Q8_0 block to FP32 */
void dequant_q8_0(float *out, BlockQ8_0 *block);

/* Dequantize tensor data to FP32 buffer
 * Returns number of floats written, or -1 on error */
vlong gguf_dequant_tensor(GGUFFile *gf, GGUFTensorInfo *tensor, float *out, vlong max_floats);

/* ----------------------------------------------------------------------------
 * GGUF to 9ml Conversion
 * ---------------------------------------------------------------------------- */

/* GGUF tensor name to 9ml weight name mapping */
typedef struct {
    char *gguf_name;    /* GGUF tensor name pattern (e.g., "blk.%d.attn_q.weight") */
    char *field;        /* 9ml weight field name */
    int per_layer;      /* 1 if this is a per-layer weight */
} GGUFTensorMap;

/* Load GGUF file into 9ml Transformer structure
 * This allocates memory and dequantizes all weights to FP32 */
int gguf_load_transformer(char *path, void *transformer);

#endif /* GGUF_H */
