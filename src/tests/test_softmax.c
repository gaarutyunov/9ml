/* Test softmax function */
#define DISABLE_THREADING
#include "model.c"

void
main(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    softmax(x, 5);

    for (int i = 0; i < 5; i++) {
        print("%.6f\n", x[i]);
    }

    exits(0);
}
