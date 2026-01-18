/*
 * core/quantize.h - Quantization functions for 9ml
 *
 * This module provides Q8_0 quantization/dequantization and quantized
 * matrix multiplication kernels.
 */

/* Group size for quantization (set by model loader) */
extern int GS;

/* Quantized tensor representation */
typedef struct {
    schar *q;    /* quantized values (int8) */
    float *s;    /* scaling factors */
} QuantizedTensor;

/* Quantize/dequantize functions */
void quantize(QuantizedTensor *qx, float *x, int n);
void dequantize(QuantizedTensor *qx, float *x, int n);

/* Initialize quantized tensors from contiguous memory */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each);

/* Quantized matrix multiplication */
void matmul_q8(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);

/* Scalar and optimized variants (for benchmarking) */
void matmul_q8_scalar(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);
void matmul_q8_unrolled(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);
