/* SIMD vectorized routines for Plan 9 */

/*
 * All SIMD functions have both assembly implementations (in simd_amd64.s)
 * and C scalar fallbacks. The dispatcher selects based on opt_config.use_simd.
 */

/* FP32 SIMD routines */

/* Matrix-vector multiplication: W (d,n) @ x (n,) -> xout (d,)
 * This is the primary compute bottleneck.
 * SSE2: processes 4 floats per cycle
 * AVX: processes 8 floats per cycle (if available)
 */
void matmul_simd(float *xout, float *x, float *w, int n, int d);

/* Dot product of two vectors: sum(a[i] * b[i]) for i in 0..n */
float dot_product_simd(float *a, float *b, int n);

/* RMS normalization: o[i] = weight[i] * (x[i] / rms(x))
 * where rms(x) = sqrt(sum(x[i]^2) / size + eps)
 */
void rmsnorm_simd(float *o, float *x, float *weight, int size);

/* Softmax: x[i] = exp(x[i] - max) / sum(exp(x - max))
 * In-place operation on x.
 * Uses 4x unrolled loops for max, exp, and normalize passes.
 */
void softmax_simd(float *x, int size);

/* Vector addition: o[i] = a[i] + b[i] for i in 0..n */
void vec_add_simd(float *o, float *a, float *b, int n);

/* Scalar-vector multiplication: o[i] = scalar * x[i] for i in 0..n */
void vec_scale_simd(float *o, float *x, float scalar, int n);

/* INT8 quantized SIMD routines */

/* Quantized matrix-vector multiplication
 * W (d,n) @ x (n,) -> xout (d,)
 * Both W and x are quantized with group scales.
 * gs = group size (typically 32)
 */
void matmul_q8_simd(float *xout, schar *xq, float *xs, schar *wq, float *ws,
                    int n, int d, int gs);

/* Quantized dot product with group scaling */
float dot_product_q8_simd(schar *aq, float *as, schar *bq, float *bs, int n, int gs);

/* Function pointer types for dispatch */
typedef void (*matmul_fn_t)(float*, float*, float*, int, int);
typedef void (*rmsnorm_fn_t)(float*, float*, float*, int);
typedef void (*softmax_fn_t)(float*, int);

/* Get the appropriate function based on SIMD availability */
matmul_fn_t get_matmul_fn(void);
rmsnorm_fn_t get_rmsnorm_fn(void);
softmax_fn_t get_softmax_fn(void);

/* C scalar fallback implementations */
void matmul_scalar(float *xout, float *x, float *w, int n, int d);
void rmsnorm_scalar(float *o, float *x, float *weight, int size);
void softmax_scalar(float *x, int size);
void matmul_q8_scalar(float *xout, schar *xq, float *xs, schar *wq, float *ws,
                      int n, int d, int gs);
