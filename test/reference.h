/* Reference implementations for test comparison */
#ifndef REFERENCE_H
#define REFERENCE_H

#include <stdint.h>

/* RMS normalization */
void ref_rmsnorm(float *out, const float *x, const float *weight, int size);

/* Softmax */
void ref_softmax(float *x, int size);

/* Matrix multiplication: W (d,n) @ x (n,) -> out (d,) */
void ref_matmul(float *out, const float *w, const float *x, int n, int d);

/* Xorshift64* RNG - returns 32-bit value */
uint32_t ref_random_u32(uint64_t *state);

/* Quantization (Q8_0 format with group_size=32) */
void ref_quantize(int8_t *q, float *s, const float *x, int n, int group_size);

/* Dequantization */
void ref_dequantize(float *out, const int8_t *q, const float *s, int n, int group_size);

/* Quantized matrix multiplication */
void ref_quantized_matmul(float *out, const int8_t *xq, const float *xs,
                          const int8_t *wq, const float *ws,
                          int n, int d, int group_size);

/* Float comparison with epsilon */
int compare_floats(const float *ref, const float *actual, int n, float eps, char *errbuf, int errbuf_size);

/* Integer comparison */
int compare_ints(const uint32_t *ref, const uint32_t *actual, int n, char *errbuf, int errbuf_size);

/* Parse output file containing floats, one per line */
int parse_float_output(const char *data, float *out, int max_n);

/* Parse output file containing integers, one per line */
int parse_int_output(const char *data, uint32_t *out, int max_n);

/* Parse key=value output into arrays */
int parse_keyval_int(const char *data, const char *key);
float parse_keyval_float(const char *data, const char *key);

#endif /* REFERENCE_H */
