/*
 * safetensors.h - Safetensors file format parser
 *
 * Safetensors is HuggingFace's format for storing model weights safely.
 * Format:
 *   - 8 bytes: header size (little-endian uint64)
 *   - header_size bytes: JSON metadata
 *   - remaining bytes: tensor data
 */

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

/* Tensor data types */
#define ST_DTYPE_F32    0
#define ST_DTYPE_F16    1
#define ST_DTYPE_BF16   2
#define ST_DTYPE_I8     3
#define ST_DTYPE_I16    4
#define ST_DTYPE_I32    5
#define ST_DTYPE_I64    6
#define ST_DTYPE_U8     7
#define ST_DTYPE_U16    8
#define ST_DTYPE_U32    9
#define ST_DTYPE_U64    10
#define ST_DTYPE_BOOL   11

/* Tensor info from header */
typedef struct STTensor STTensor;
struct STTensor {
    char *name;             /* Tensor name (e.g., "model.embed_tokens.weight") */
    int dtype;              /* Data type (ST_DTYPE_*) */
    int n_dims;             /* Number of dimensions */
    uvlong dims[4];         /* Dimension sizes (max 4D) */
    uvlong offset_start;    /* Start offset in data section */
    uvlong offset_end;      /* End offset in data section */
    STTensor *next;
};

/* Safetensors file handle */
typedef struct STFile STFile;
struct STFile {
    int fd;                 /* File descriptor */
    uvlong header_size;     /* Size of JSON header */
    uvlong data_offset;     /* Offset where tensor data starts */
    char *header_json;      /* Raw JSON header */
    STTensor *tensors;      /* Linked list of tensors */
    int n_tensors;          /* Number of tensors */
};

/* Open a safetensors file */
int st_open(STFile *sf, char *path);

/* Close and free resources */
void st_close(STFile *sf);

/* Find a tensor by name */
STTensor *st_find_tensor(STFile *sf, char *name);

/* Find tensors matching a pattern (returns linked list) */
STTensor *st_find_tensors(STFile *sf, char *pattern);

/* Get number of elements in tensor */
uvlong st_tensor_nelements(STTensor *t);

/* Get size in bytes of tensor data */
uvlong st_tensor_size(STTensor *t);

/* Read tensor data into buffer (dequantizes to float32 if needed) */
vlong st_read_tensor(STFile *sf, STTensor *t, float *out, vlong max_floats);

/* Read raw tensor data (no conversion) */
vlong st_read_tensor_raw(STFile *sf, STTensor *t, void *out, vlong max_bytes);

/* Get dtype size in bytes */
int st_dtype_size(int dtype);

/* Get dtype name */
char *st_dtype_name(int dtype);

/* Free tensor list (for st_find_tensors results) */
void st_tensor_list_free(STTensor *t);

#endif /* SAFETENSORS_H */
