/* Test rmsnorm function */
#define DISABLE_THREADING
#include "model.c"

void
main(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {0.5f, 0.5f, 0.5f, 0.5f};
    float o[4];

    rmsnorm(o, x, w, 4);

    for (int i = 0; i < 4; i++) {
        print("%.6f\n", o[i]);
    }

    exits(0);
}
