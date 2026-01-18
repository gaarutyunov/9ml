/* Quantized SIMD routines for Plan 9 amd64 */
/*
 * NOTE: Quantized matrix multiplication is complex due to:
 * 1. Mixed int8/float arithmetic
 * 2. Group-based scaling
 * 3. Need for packed byte operations (PMADDUBSW, etc.)
 *
 * The C implementation in modelq.c (matmul_q8_scalar) is used instead.
 * This file provides stubs that delegate to the C implementation.
 *
 * If SSE4.1 instructions (PMADDUBSW, PMADDWD) were supported by Plan 9's
 * assembler, this could be optimized. Currently x87 is too slow for
 * the byte-level operations required.
 */

/* Placeholder - actual implementation is in C */
