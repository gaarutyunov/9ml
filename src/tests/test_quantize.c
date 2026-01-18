/* Test quantize/dequantize roundtrip */
#define DISABLE_OPTIMIZATIONS
#include "modelq.c"

void
main(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    /* Set global group size */
    GS = 32;

    float x[64];
    for (int i = 0; i < 64; i++) {
        x[i] = (float)i / 10.0f;
    }

    QuantizedTensor qx;
    qx.q = malloc(64 * sizeof(schar));
    qx.s = malloc(2 * sizeof(float));

    quantize(&qx, x, 64);

    float result[64];
    dequantize(&qx, result, 64);

    for (int i = 0; i < 64; i++) {
        print("%.6f\n", result[i]);
    }

    free(qx.q);
    free(qx.s);
    exits(0);
}
