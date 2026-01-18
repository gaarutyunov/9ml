/* Test matmul function */
#define DISABLE_THREADING
#include "model.c"

void
main(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    float w[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float x[] = {1, 2, 3, 4};
    float out[3];

    matmul(out, x, w, 4, 3);

    for (int i = 0; i < 3; i++) {
        print("%.6f\n", out[i]);
    }

    exits(0);
}
