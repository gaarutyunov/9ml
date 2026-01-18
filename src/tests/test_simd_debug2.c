/* Minimal SIMD debug test 2 - isolate rmsnorm denormal issue */
#include "model.c"

void
threadmain(int argc, char *argv[])
{
    USED(argc);
    USED(argv);

    print("=== SIMD Debug Test 2 ===\n\n");

    /* Test 1: Check current FPU state */
    ulong fcr = getfcr();
    print("Initial FCR: %#ulx\n", fcr);
    print("  FPINVAL=%d FPZDIV=%d FPOVFL=%d FPUNFL=%d FPINEX=%d\n",
          (fcr & FPINVAL) ? 1 : 0,
          (fcr & FPZDIV) ? 1 : 0,
          (fcr & FPOVFL) ? 1 : 0,
          (fcr & FPUNFL) ? 1 : 0,
          (fcr & FPINEX) ? 1 : 0);

    /* Test 2: Try disabling FP exceptions */
    print("\nDisabling FP exceptions with setfcr...\n");
    setfcr(getfcr() & ~(FPINVAL|FPZDIV|FPOVFL|FPUNFL|FPINEX));
    fcr = getfcr();
    print("After setfcr: %#ulx\n", fcr);

    /* Test 3: matmul_simd works - verify */
    print("\nTesting matmul_simd (8x8)...\n");
    float *w = malloc(64 * sizeof(float));
    float *x = malloc(8 * sizeof(float));
    float *out = malloc(8 * sizeof(float));
    for (int i = 0; i < 64; i++) w[i] = 0.1f;
    for (int i = 0; i < 8; i++) x[i] = 0.5f;
    for (int i = 0; i < 8; i++) out[i] = -999.0f;

    opt_config.use_simd = 1;
    matmul(out, x, w, 8, 8);
    print("matmul_simd result[0]: %f (expected ~0.4)\n", out[0]);

    /* Test 4: rmsnorm with simple data - all 1.0s */
    print("\nTesting rmsnorm_simd (8 elements, all 1.0)...\n");
    float *rms_x = malloc(8 * sizeof(float));
    float *rms_w = malloc(8 * sizeof(float));
    float *rms_out = malloc(8 * sizeof(float));
    for (int i = 0; i < 8; i++) {
        rms_x[i] = 1.0f;  /* Simple: all 1s */
        rms_w[i] = 1.0f;
        rms_out[i] = -999.0f;
    }

    print("Before rmsnorm: x[0]=%f, w[0]=%f\n", rms_x[0], rms_w[0]);
    print("Calling rmsnorm with use_simd=1...\n");
    opt_config.use_simd = 1;
    rmsnorm(rms_out, rms_x, rms_w, 8);
    print("rmsnorm_simd result[0]: %f (expected ~1.0)\n", rms_out[0]);

    free(w);
    free(x);
    free(out);
    free(rms_x);
    free(rms_w);
    free(rms_out);

    print("\n=== Test Complete ===\n");
    threadexits(0);
}
