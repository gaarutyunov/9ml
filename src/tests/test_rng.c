/* Test random number generator */
#include "model.c"

void
main(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    uvlong state = 42;

    for (int i = 0; i < 10; i++) {
        print("%ud\n", random_u32(&state));
    }

    exits(0);
}
