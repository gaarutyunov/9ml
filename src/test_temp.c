#include <u.h>
#include <libc.h>

int GS = 32;

typedef struct {
    schar* q;
    float* s;
} QuantizedTensor;

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / Q_MAX;
        if (scale == 0) scale = 1.0f;
        qx->s[group] = scale;
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale;
            schar quantized = (schar) floor(quant_value + 0.5f);
            qx->q[group * GS + i] = quantized;
        }
    }
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int ival = 0;
        int in = i * n;
        for (int j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int) x->q[j + k]) * ((int) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        xout[i] = val;
    }
}

void main(int, char**) {
    int n = 64, d = 2;
    float x[64], w[128];

    for (int i = 0; i < n; i++) x[i] = (float)(i % 10) / 10.0f;
    for (int i = 0; i < d * n; i++) w[i] = (float)((i / n + i % n) % 10) / 10.0f;

    QuantizedTensor xq, wq;
    xq.q = malloc(n * sizeof(schar));
    xq.s = malloc((n / GS) * sizeof(float));
    wq.q = malloc(d * n * sizeof(schar));
    wq.s = malloc((d * n / GS) * sizeof(float));

    quantize(&xq, x, n);
    quantize(&wq, w, d * n);

    float out[2];
    matmul(out, &xq, &wq, n, d);

    for (int i = 0; i < d; i++) print("%.6f\n", out[i]);
    exits(0);
}
