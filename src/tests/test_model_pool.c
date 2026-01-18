/* Test pool_create from model.c directly */
#include "model.c"

#define BENCH_D 1024
#define BENCH_N 1024
#define BENCH_ITERS 100

/* Same random generator as benchmark */
static uvlong bench_seed = 12345;
static float rand_float(void) {
    bench_seed = bench_seed * 1103515245 + 12345;
    return (float)(bench_seed & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

void
threadmain(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    print("=== Test pool_create from model.c ===\n");

    /* Allocate large arrays like benchmark does */
    print("Allocating large arrays (same as benchmark)\n");
    float *w = malloc(BENCH_D * BENCH_N * sizeof(float));
    float *x = malloc(BENCH_N * sizeof(float));
    float *out = malloc(BENCH_D * sizeof(float));

    if (!w || !x || !out) {
        fprint(2, "malloc failed\n");
        threadexits("malloc");
    }

    print("Arrays allocated at: w=%p x=%p out=%p\n", w, x, out);

    /* Initialize data like benchmark - with same random data */
    bench_seed = 12345;
    for (int i = 0; i < BENCH_D * BENCH_N; i++) {
        w[i] = rand_float() - 0.5f;
    }
    for (int i = 0; i < BENCH_N; i++) {
        x[i] = rand_float() - 0.5f;
    }

    print("Arrays initialized (with random data)\n");

    /* Run BASELINE matmul like benchmark */
    print("Running BASELINE matmul\n");
    opt_config.use_simd = 0;
    opt_config.nthreads = 1;
    opt_init();

    for (int i = 0; i < 100; i++) {
        matmul(out, x, w, BENCH_N, BENCH_D);
    }

    print("BASELINE complete\n");
    opt_cleanup();

    /* Now try to create a pool */
    print("Setting nthreads=0 (auto-detect), calling opt_init\n");
    opt_config.use_simd = 0;
    opt_config.nthreads = 0;  /* auto-detect, like benchmark */
    opt_init();

    print("opt_init completed\n");

    if (global_pool == nil) {
        print("ERROR: global_pool is nil\n");
    } else {
        print("SUCCESS: global_pool created with %d workers\n", global_pool->nworkers);
    }

    print("Calling opt_cleanup\n");
    opt_cleanup();
    print("opt_cleanup completed\n");

    print("test complete\n");
    threadexits(0);
}
