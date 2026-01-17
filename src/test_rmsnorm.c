#include <u.h>
#include <libc.h>

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrt(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void main(int, char**) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {0.5f, 0.5f, 0.5f, 0.5f};
    float o[4];

    rmsnorm(o, x, w, 4);

    for (int i = 0; i < 4; i++) {
        print("%.6f\n", o[i]);
    }
    exits(0);
}
