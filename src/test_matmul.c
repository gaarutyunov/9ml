#include <u.h>
#include <libc.h>

void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void main(int, char**) {
    float w[] = {1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12};
    float x[] = {1, 2, 3, 4};
    float out[3];
    matmul(out, x, w, 4, 3);
    for (int i = 0; i < 3; i++) {
        print("%.6f\n", out[i]);
    }
    exits(0);
}
