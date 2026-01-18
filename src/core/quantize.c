/*
 * core/quantize.c - Quantization functions for 9ml
 *
 * Provides Q8_0 quantization/dequantization and quantized matrix
 * multiplication kernels with optimized variants.
 */

#include <u.h>
#include <libc.h>

#include "quantize.h"
#include "kernels.h"

/* Group size global for quantization of the weights */
int GS = 0;

void
dequantize(QuantizedTensor *qx, float *x, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void
quantize(QuantizedTensor *qx, float *x, int n)
{
    int num_groups;
    float Q_MAX;
    int group, i;
    float wmax, val, scale, quant_value;
    schar quantized;

    num_groups = n / GS;
    Q_MAX = 127.0f;

    for (group = 0; group < num_groups; group++) {
        /* find the max absolute value in the current group */
        wmax = 0.0;
        for (i = 0; i < GS; i++) {
            val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        /* calculate and write the scaling factor */
        scale = wmax / Q_MAX;
        qx->s[group] = scale;

        /* calculate and write the quantized values */
        for (i = 0; i < GS; i++) {
            quant_value = x[group * GS + i] / scale;
            quantized = (schar)floor(quant_value + 0.5f);
            qx->q[group * GS + i] = quantized;
        }
    }
}

/* Initialize n quantized tensors (with size_each elements), starting from *ptr */
QuantizedTensor *
init_quantized_tensors(void **ptr, int n, int size_each)
{
    void *p;
    QuantizedTensor *res;
    int i;

    p = *ptr;
    res = malloc(n * sizeof(QuantizedTensor));

    for (i = 0; i < n; i++) {
        /* map quantized int8 values */
        res[i].q = (schar *)p;
        p = (schar *)p + size_each;
        /* map scale factors */
        res[i].s = (float *)p;
        p = (float *)p + size_each / GS;
    }

    *ptr = p;  /* advance ptr to current position */
    return res;
}

/* Scalar quantized matmul - baseline implementation */
void
matmul_q8_scalar(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d)
{
    int i, j, k, in, ival;
    float val;

    for (i = 0; i < d; i++) {
        val = 0.0f;
        in = i * n;

        for (j = 0; j <= n - GS; j += GS) {
            ival = 0;
            for (k = 0; k < GS; k++) {
                ival += ((int)x->q[j + k]) * ((int)w->q[in + j + k]);
            }
            val += ((float)ival) * w->s[(in + j) / GS] * x->s[j / GS];
        }

        xout[i] = val;
    }
}

/* Optimized quantized matmul with 4x loop unrolling */
void
matmul_q8_unrolled(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d)
{
    int i, j, k, base;
    float val;
    schar *wq, *xq;
    float *ws, *xs;
    int ival0, ival1, ival2, ival3, ival;

    for (i = 0; i < d; i++) {
        val = 0.0f;
        wq = w->q + i * n;
        xq = x->q;
        ws = w->s + (i * n) / GS;
        xs = x->s;

        for (j = 0; j <= n - GS; j += GS) {
            /* Unroll inner loop 4x for better ILP */
            ival0 = 0;
            ival1 = 0;
            ival2 = 0;
            ival3 = 0;
            base = j;

            /* GS is typically 32, so 8 iterations of 4 */
            for (k = 0; k < GS; k += 4) {
                ival0 += (int)xq[base + k] * (int)wq[base + k];
                ival1 += (int)xq[base + k + 1] * (int)wq[base + k + 1];
                ival2 += (int)xq[base + k + 2] * (int)wq[base + k + 2];
                ival3 += (int)xq[base + k + 3] * (int)wq[base + k + 3];
            }

            ival = ival0 + ival1 + ival2 + ival3;
            val += (float)ival * ws[j / GS] * xs[j / GS];
        }

        xout[i] = val;
    }
}

/* Dispatch function - uses optimized version when SIMD is enabled */
void
matmul_q8(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d)
{
    /* W (d,n) @ x (n,) -> xout (d,)
     * by far the most amount of time is spent inside this little function
     * inputs to this function are both quantized
     */
    if (opt_config.use_simd) {
        matmul_q8_unrolled(xout, x, w, n, d);
    } else {
        matmul_q8_scalar(xout, x, w, n, d);
    }
}
