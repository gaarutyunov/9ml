/*
 * core/kernels.c - Shared compute kernels for 9ml
 *
 * Provides optimized kernel functions that can be used by any model
 * architecture plugin. Includes SIMD-accelerated versions using SSE2.
 */

#include <u.h>
#include <libc.h>

#include "kernels.h"

/* Global optimization configuration */
OptConfig opt_config = {
    .nthreads = 0,      /* 0 = auto-detect */
#ifdef DISABLE_THREADING
    .use_simd = 0,      /* SIMD disabled when threading disabled */
#else
    .use_simd = 1,      /* SIMD enabled by default */
#endif
};

/* ----------------------------------------------------------------------------
 * Scalar implementations
 * ---------------------------------------------------------------------------- */

void
rmsnorm_scalar(float *o, float *x, float *weight, int size)
{
    int j;
    float ss;

    /* calculate sum of squares */
    ss = 0.0f;
    for (j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrt(ss);

    /* normalize and scale */
    for (j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void
softmax_scalar(float *x, int size)
{
    int i;
    float max_val, sum;

    /* find max value (for numerical stability) */
    max_val = x[0];
    for (i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    /* exp and sum */
    sum = 0.0f;
    for (i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }

    /* normalize */
    for (i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void
matmul_scalar(float *xout, float *x, float *w, int n, int d)
{
    int i, j;
    float val;

    /* W (d,n) @ x (n,) -> xout (d,) */
    for (i = 0; i < d; i++) {
        val = 0.0f;
        for (j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

/* ----------------------------------------------------------------------------
 * SIMD implementations (optimized C version for softmax)
 * ---------------------------------------------------------------------------- */

#ifndef DISABLE_THREADING
void
softmax_simd(float *x, int size)
{
    int i;
    float max0, max1, max2, max3, max_val;
    float sum0, sum1, sum2, sum3, sum, inv_sum;

    /* Find max value (4x unrolled) */
    max0 = x[0];
    max1 = size > 1 ? x[1] : x[0];
    max2 = size > 2 ? x[2] : x[0];
    max3 = size > 3 ? x[3] : x[0];

    for (i = 4; i + 3 < size; i += 4) {
        if (x[i]     > max0) max0 = x[i];
        if (x[i + 1] > max1) max1 = x[i + 1];
        if (x[i + 2] > max2) max2 = x[i + 2];
        if (x[i + 3] > max3) max3 = x[i + 3];
    }

    max_val = max0;
    if (max1 > max_val) max_val = max1;
    if (max2 > max_val) max_val = max2;
    if (max3 > max_val) max_val = max3;

    /* Handle remaining elements for max */
    for (; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Exp and sum (4x unrolled) */
    sum0 = 0.0f;
    sum1 = 0.0f;
    sum2 = 0.0f;
    sum3 = 0.0f;

    for (i = 0; i + 3 < size; i += 4) {
        x[i]     = exp(x[i]     - max_val);
        x[i + 1] = exp(x[i + 1] - max_val);
        x[i + 2] = exp(x[i + 2] - max_val);
        x[i + 3] = exp(x[i + 3] - max_val);
        sum0 += x[i];
        sum1 += x[i + 1];
        sum2 += x[i + 2];
        sum3 += x[i + 3];
    }

    sum = sum0 + sum1 + sum2 + sum3;

    /* Handle remaining elements for exp */
    for (; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }

    /* Normalize (4x unrolled) */
    inv_sum = 1.0f / sum;
    for (i = 0; i + 3 < size; i += 4) {
        x[i]     *= inv_sum;
        x[i + 1] *= inv_sum;
        x[i + 2] *= inv_sum;
        x[i + 3] *= inv_sum;
    }

    /* Handle remaining elements for normalize */
    for (; i < size; i++) {
        x[i] *= inv_sum;
    }
}
#endif

/* ----------------------------------------------------------------------------
 * Dispatchers - choose SIMD or scalar based on config
 * ---------------------------------------------------------------------------- */

void
rmsnorm(float *o, float *x, float *weight, int size)
{
#ifndef DISABLE_THREADING
    if (opt_config.use_simd) {
        rmsnorm_simd(o, x, weight, size);
    } else {
        rmsnorm_scalar(o, x, weight, size);
    }
#else
    rmsnorm_scalar(o, x, weight, size);
#endif
}

void
softmax(float *x, int size)
{
#ifndef DISABLE_THREADING
    if (opt_config.use_simd) {
        softmax_simd(x, size);
    } else {
        softmax_scalar(x, size);
    }
#else
    softmax_scalar(x, size);
#endif
}

void
matmul(float *xout, float *x, float *w, int n, int d)
{
#ifndef DISABLE_THREADING
    if (opt_config.use_simd) {
        matmul_simd(xout, x, w, n, d);
    } else {
        matmul_scalar(xout, x, w, n, d);
    }
#else
    matmul_scalar(xout, x, w, n, d);
#endif
}
