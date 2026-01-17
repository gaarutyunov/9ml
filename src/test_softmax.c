#include <u.h>
#include <libc.h>

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void main(int, char**) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    softmax(x, 5);
    for (int i = 0; i < 5; i++) {
        print("%.6f\n", x[i]);
    }
    exits(0);
}
