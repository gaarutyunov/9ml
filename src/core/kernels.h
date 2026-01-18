/*
 * core/kernels.h - Shared compute kernels for 9ml
 *
 * This module provides optimized kernel functions (matmul, rmsnorm, softmax)
 * that can be used by any model architecture plugin.
 */

/* Optimization configuration */
typedef struct {
    int nthreads;           /* number of threads to use (0 = auto-detect) */
    int use_simd;           /* whether to use SIMD optimizations */
} OptConfig;

extern OptConfig opt_config;

/* Core kernel functions - dispatchers that choose SIMD or scalar */
void matmul(float *xout, float *x, float *w, int n, int d);
void rmsnorm(float *o, float *x, float *weight, int size);
void softmax(float *x, int size);

/* Scalar implementations (for testing/fallback) */
void matmul_scalar(float *xout, float *x, float *w, int n, int d);
void rmsnorm_scalar(float *o, float *x, float *weight, int size);
void softmax_scalar(float *x, int size);

#ifndef DISABLE_THREADING
/* SIMD implementations (defined in simd_amd64.s) */
extern void matmul_simd(float *xout, float *x, float *w, int n, int d);
extern float dot_product_simd(float *a, float *b, int n);
extern void rmsnorm_simd(float *o, float *x, float *weight, int size);
extern void vec_add_simd(float *o, float *a, float *b, int n);
extern void vec_scale_simd(float *o, float *x, float scalar, int n);

/* Optimized softmax (C implementation with 4x unrolling) */
void softmax_simd(float *x, int size);
#endif
