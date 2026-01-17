/* Reference implementations for test comparison */
#include "reference.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void ref_rmsnorm(float *out, const float *x, const float *weight, int size) {
    /* Calculate sum of squares */
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    /* Normalize and scale */
    for (int i = 0; i < size; i++) {
        out[i] = weight[i] * (ss * x[i]);
    }
}

void ref_softmax(float *x, int size) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    /* Exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    /* Normalize */
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void ref_matmul(float *out, const float *w, const float *x, int n, int d) {
    /* W (d,n) @ x (n,) -> out (d,) */
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        out[i] = val;
    }
}

uint32_t ref_random_u32(uint64_t *state) {
    /* xorshift64* */
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * 0x2545F4914F6CDD1DULL) >> 32);
}

void ref_quantize(int8_t *q, float *s, const float *x, int n, int group_size) {
    int num_groups = n / group_size;
    const float Q_MAX = 127.0f;

    for (int g = 0; g < num_groups; g++) {
        /* Find max absolute value in group */
        float wmax = 0.0f;
        for (int i = 0; i < group_size; i++) {
            float val = fabsf(x[g * group_size + i]);
            if (val > wmax) wmax = val;
        }
        /* Calculate scale */
        float scale = wmax / Q_MAX;
        s[g] = scale;
        /* Quantize */
        for (int i = 0; i < group_size; i++) {
            float quant_value = (scale > 0) ? x[g * group_size + i] / scale : 0;
            q[g * group_size + i] = (int8_t)roundf(quant_value);
        }
    }
}

void ref_dequantize(float *out, const int8_t *q, const float *s, int n, int group_size) {
    for (int i = 0; i < n; i++) {
        out[i] = q[i] * s[i / group_size];
    }
}

void ref_quantized_matmul(float *out, const int8_t *xq, const float *xs,
                          const int8_t *wq, const float *ws,
                          int n, int d, int group_size) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j += group_size) {
            int ival = 0;
            for (int k = 0; k < group_size; k++) {
                ival += (int)xq[j + k] * (int)wq[i * n + j + k];
            }
            val += (float)ival * ws[(i * n + j) / group_size] * xs[j / group_size];
        }
        out[i] = val;
    }
}

int compare_floats(const float *ref, const float *actual, int n, float eps, char *errbuf, int errbuf_size) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - actual[i]);
        if (diff > eps) {
            snprintf(errbuf, errbuf_size, "mismatch at [%d]: ref=%.6f, actual=%.6f, diff=%.6f",
                     i, ref[i], actual[i], diff);
            return 0;
        }
    }
    return 1;
}

int compare_ints(const uint32_t *ref, const uint32_t *actual, int n, char *errbuf, int errbuf_size) {
    for (int i = 0; i < n; i++) {
        if (ref[i] != actual[i]) {
            snprintf(errbuf, errbuf_size, "mismatch at [%d]: ref=%u, actual=%u",
                     i, ref[i], actual[i]);
            return 0;
        }
    }
    return 1;
}

int parse_float_output(const char *data, float *out, int max_n) {
    int n = 0;
    const char *p = data;
    while (*p && n < max_n) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (!*p) break;
        /* Try to parse a float */
        char *end;
        float val = strtof(p, &end);
        if (end == p) {
            /* Skip non-numeric line */
            while (*p && *p != '\n') p++;
            continue;
        }
        out[n++] = val;
        p = end;
    }
    return n;
}

int parse_int_output(const char *data, uint32_t *out, int max_n) {
    int n = 0;
    const char *p = data;
    while (*p && n < max_n) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (!*p) break;
        /* Try to parse an integer */
        char *end;
        long val = strtol(p, &end, 10);
        if (end == p) {
            /* Skip non-numeric line */
            while (*p && *p != '\n') p++;
            continue;
        }
        out[n++] = (uint32_t)val;
        p = end;
    }
    return n;
}

int parse_keyval_int(const char *data, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "%s=", key);
    const char *p = strstr(data, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    return atoi(p);
}

float parse_keyval_float(const char *data, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "%s=", key);
    const char *p = strstr(data, pattern);
    if (!p) return 0.0f;
    p += strlen(pattern);
    return strtof(p, NULL);
}
