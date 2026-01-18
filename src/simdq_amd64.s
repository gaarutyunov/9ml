/* Quantized SIMD routines for Plan 9 amd64
 *
 * Uses BYTE-encoded SSE2 instructions since Plan 9's assembler
 * doesn't support integer SSE mnemonics directly.
 *
 * ModR/M encoding: mod(2)|reg(3)|r/m(3)
 * For XMM register-register: mod=11, reg=dst, r/m=src
 * X0=0, X1=1, X2=2, X3=3, X4=4, X5=5, X6=6, X7=7
 *
 * Common encodings:
 * PXOR X0,X0     = 66 0F EF C0
 * MOVDQU (R),Xn  = F3 0F 6F <modrm>  (load 16 bytes)
 * MOVQ (R),Xn    = F3 0F 7E <modrm>  (load 8 bytes to low half)
 * PCMPGTB Xn,Xm  = 66 0F 64 <modrm>  (compare for sign mask)
 * PUNPCKLBW Xn,Xm = 66 0F 60 <modrm> (unpack low bytes)
 * PMADDWD Xn,Xm  = 66 0F F5 <modrm>  (multiply words, add pairs to dword)
 * PADDD Xn,Xm    = 66 0F FE <modrm>  (add packed dwords)
 * CVTDQ2PS Xn,Xm = 0F 5B <modrm>     (convert dwords to floats)
 * MULPS Xn,Xm    = 0F 59 <modrm>     (multiply packed floats)
 * ADDPS Xn,Xm    = 0F 58 <modrm>     (add packed floats)
 * PSHUFD Xn,Xm,i = 66 0F 70 <modrm> <imm> (shuffle dwords)
 * MOVD Xn,R      = 66 0F 7E <modrm>  (move low dword to GPR)
 * MOVSS Xn,(R)   = F3 0F 11 <modrm>  (store scalar float)
 */

/* dot_product_q8_group - compute dot product of 32 int8 values
 *
 * Computes: sum(a[i] * b[i]) for i in 0..31
 * Returns: int32 result
 *
 * Args:
 *   BP    = a (pointer to 32 signed bytes)
 *   16(SP) = b (pointer to 32 signed bytes)
 *
 * Algorithm:
 *   Process 8 bytes at a time (4 iterations for 32 bytes)
 *   1. Load 8 bytes from a and b
 *   2. Sign-extend to int16 using PCMPGTB trick
 *   3. PMADDWD to get 4 int32 products
 *   4. Accumulate with PADDD
 *   5. Horizontal sum to get final int32
 */
TEXT dot_product_q8_group(SB), $0
    MOVQ    BP, DI              /* DI = a */
    MOVQ    16(SP), SI          /* SI = b */

    /* Zero accumulator X4 */
    /* PXOR X4, X4 = 66 0F EF E4 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xEF; BYTE $0xE4

    /* Process 4 groups of 8 bytes each (total 32 bytes) */
    MOVL    $4, CX              /* Loop counter */
    XORL    R8, R8              /* Offset = 0 */

dot_q8_loop:
    /* Load 8 bytes from a into X0 (low 64 bits) */
    /* MOVQ (DI+R8), X0 = F3 0F 7E 04 07 */
    /* Actually need: F3 REX.W 0F 7E /r with SIB */
    /* Let's use simpler: move DI+R8 to R9, then load */
    LEAQ    (DI)(R8*1), R9
    /* MOVQ (R9), X0 = F3 0F 7E 01 */
    BYTE $0xF3; BYTE $0x41; BYTE $0x0F; BYTE $0x7E; BYTE $0x01

    /* Load 8 bytes from b into X1 (low 64 bits) */
    LEAQ    (SI)(R8*1), R9
    /* MOVQ (R9), X1 = F3 0F 7E 09 */
    BYTE $0xF3; BYTE $0x41; BYTE $0x0F; BYTE $0x7E; BYTE $0x09

    /* Sign-extend a (X0) to int16 */
    /* Copy X0 to X2: MOVDQA X0, X2 = 66 0F 6F D0 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x6F; BYTE $0xD0

    /* Zero X3: PXOR X3, X3 = 66 0F EF DB */
    BYTE $0x66; BYTE $0x0F; BYTE $0xEF; BYTE $0xDB

    /* PCMPGTB X0, X3 (X3 = sign mask for a) = 66 0F 64 D8 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x64; BYTE $0xD8

    /* PUNPCKLBW X3, X0 (X0 = sign-extended a) = 66 0F 60 C3 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x60; BYTE $0xC3

    /* Sign-extend b (X1) to int16 */
    /* Copy X1 to X2: MOVDQA X1, X2 = 66 0F 6F D1 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x6F; BYTE $0xD1

    /* Zero X3: PXOR X3, X3 = 66 0F EF DB */
    BYTE $0x66; BYTE $0x0F; BYTE $0xEF; BYTE $0xDB

    /* PCMPGTB X1, X3 (X3 = sign mask for b) = 66 0F 64 D9 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x64; BYTE $0xD9

    /* PUNPCKLBW X3, X1 (X1 = sign-extended b) = 66 0F 60 CB */
    BYTE $0x66; BYTE $0x0F; BYTE $0x60; BYTE $0xCB

    /* PMADDWD X1, X0 (X0 = 4 x int32 products) = 66 0F F5 C1 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xF5; BYTE $0xC1

    /* PADDD X0, X4 (accumulate) = 66 0F FE E0 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xFE; BYTE $0xE0

    /* Next 8 bytes */
    ADDL    $8, R8
    DECL    CX
    JNZ     dot_q8_loop

    /* Horizontal sum of X4 (4 x int32) */
    /* PSHUFD X4, X5, 0xEE = shuffle [2,3,2,3] = 66 0F 70 EC EE */
    BYTE $0x66; BYTE $0x0F; BYTE $0x70; BYTE $0xEC; BYTE $0xEE

    /* PADDD X5, X4 = 66 0F FE E5 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xFE; BYTE $0xE5

    /* PSHUFD X4, X5, 0x55 = shuffle [1,1,1,1] = 66 0F 70 EC 55 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x70; BYTE $0xEC; BYTE $0x55

    /* PADDD X5, X4 = 66 0F FE E5 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xFE; BYTE $0xE5

    /* Move result to AX: MOVD X4, AX = 66 0F 7E E0 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x7E; BYTE $0xE0

    RET


/* matmul_q8_simd - quantized matrix-vector multiply
 *
 * W (d,n) @ x (n,) -> xout (d,)
 * Uses group-based quantization with group size GS.
 *
 * Args (Plan 9 amd64 calling convention):
 *   BP      = xout (pointer to d floats)
 *   16(SP)  = xq (pointer to n int8)
 *   24(SP)  = xs (pointer to n/GS floats - input scales)
 *   32(SP)  = wq (pointer to d*n int8)
 *   40(SP)  = ws (pointer to d*n/GS floats - weight scales)
 *   48(SP)  = n (int)
 *   56(SP)  = d (int)
 *   64(SP)  = gs (int) - group size
 *
 * Note: This is a simplified implementation that processes one row at a time.
 * For full performance, would need to unroll across multiple rows.
 */
TEXT matmul_q8_simd(SB), $64
    /* Save callee-saved registers */
    MOVQ    BP, 0(SP)           /* Save original BP */
    MOVQ    BX, 8(SP)
    MOVQ    R12, 16(SP)
    MOVQ    R13, 24(SP)
    MOVQ    R14, 32(SP)
    MOVQ    R15, 40(SP)

    /* Load parameters */
    MOVQ    BP, R15             /* R15 = xout */
    MOVQ    80(SP), DI          /* DI = xq (adjust for saved regs: 16+64=80) */
    MOVQ    88(SP), SI          /* SI = xs */
    MOVQ    96(SP), R8          /* R8 = wq */
    MOVQ    104(SP), R9         /* R9 = ws */
    MOVL    112(SP), R10        /* R10 = n */
    MOVL    120(SP), R11        /* R11 = d */
    MOVL    128(SP), R12        /* R12 = gs */

    /* Row loop: for i in 0..d */
    XORL    R13, R13            /* R13 = i = 0 */

matmul_q8_row_loop:
    CMPL    R13, R11
    JGE     matmul_q8_done

    /* Initialize float accumulator X7 to 0 */
    XORPS   X7, X7

    /* Group loop: for j in 0..n step gs */
    XORL    R14, R14            /* R14 = j = 0 */

matmul_q8_group_loop:
    /* Check if j + gs <= n */
    MOVL    R14, AX
    ADDL    R12, AX             /* AX = j + gs */
    CMPL    AX, R10
    JG      matmul_q8_store_row

    /* Compute integer dot product for this group using inline SIMD */
    /* We'll process 8 bytes at a time, gs/8 iterations */

    /* Zero integer accumulator X4 */
    /* PXOR X4, X4 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xEF; BYTE $0xE4

    /* Calculate pointers:
     * wq_ptr = wq + i*n + j = R8 + R13*R10 + R14
     * xq_ptr = xq + j = DI + R14
     */
    MOVQ    R13, AX
    IMULQ   R10, AX             /* AX = i * n */
    ADDQ    R14, AX             /* AX = i * n + j */
    LEAQ    (R8)(AX*1), BX      /* BX = wq + i*n + j */
    LEAQ    (DI)(R14*1), CX     /* CX = xq + j */

    /* Process gs bytes in chunks of 8 */
    MOVL    R12, DX             /* DX = gs */
    SHRL    $3, DX              /* DX = gs / 8 */

matmul_q8_inner_loop:
    TESTL   DX, DX
    JZ      matmul_q8_inner_done

    /* Load 8 bytes from wq */
    /* MOVQ (BX), X0 */
    BYTE $0xF3; BYTE $0x0F; BYTE $0x7E; BYTE $0x03

    /* Load 8 bytes from xq */
    /* MOVQ (CX), X1 */
    BYTE $0xF3; BYTE $0x0F; BYTE $0x7E; BYTE $0x09

    /* Sign-extend X0 (wq) */
    /* PXOR X3, X3 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xEF; BYTE $0xDB
    /* PCMPGTB X0, X3 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x64; BYTE $0xD8
    /* PUNPCKLBW X3, X0 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x60; BYTE $0xC3

    /* Sign-extend X1 (xq) */
    /* PXOR X3, X3 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xEF; BYTE $0xDB
    /* PCMPGTB X1, X3 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x64; BYTE $0xD9
    /* PUNPCKLBW X3, X1 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x60; BYTE $0xCB

    /* PMADDWD X1, X0 -> X0 = 4 x int32 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xF5; BYTE $0xC1

    /* PADDD X0, X4 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xFE; BYTE $0xE0

    ADDQ    $8, BX
    ADDQ    $8, CX
    DECL    DX
    JMP     matmul_q8_inner_loop

matmul_q8_inner_done:
    /* Horizontal sum of X4 to get total int32 */
    /* PSHUFD X4, X5, 0xEE */
    BYTE $0x66; BYTE $0x0F; BYTE $0x70; BYTE $0xEC; BYTE $0xEE
    /* PADDD X5, X4 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xFE; BYTE $0xE5
    /* PSHUFD X4, X5, 0x55 */
    BYTE $0x66; BYTE $0x0F; BYTE $0x70; BYTE $0xEC; BYTE $0x55
    /* PADDD X5, X4 */
    BYTE $0x66; BYTE $0x0F; BYTE $0xFE; BYTE $0xE5

    /* Convert int32 in X4 to float in X4 */
    /* CVTDQ2PS X4, X4 = 0F 5B E4 */
    BYTE $0x0F; BYTE $0x5B; BYTE $0xE4

    /* Multiply by weight scale: ws[(i*n + j) / gs] */
    /* Calculate scale index */
    MOVQ    R13, AX
    IMULQ   R10, AX             /* AX = i * n */
    ADDQ    R14, AX             /* AX = i * n + j */
    XORL    DX, DX
    DIVQ    R12                 /* AX = (i*n + j) / gs */
    /* Load scale: MOVSS (R9 + AX*4), X5 */
    MOVSS   (R9)(AX*4), X5
    MULSS   X5, X4              /* X4 *= ws */

    /* Multiply by input scale: xs[j / gs] */
    MOVQ    R14, AX
    XORL    DX, DX
    DIVQ    R12                 /* AX = j / gs */
    MOVSS   (SI)(AX*4), X5
    MULSS   X5, X4              /* X4 *= xs */

    /* Add to accumulator */
    ADDSS   X4, X7

    /* Next group */
    ADDL    R12, R14
    JMP     matmul_q8_group_loop

matmul_q8_store_row:
    /* Store result: xout[i] = X7 */
    MOVSS   X7, (R15)(R13*4)

    /* Next row */
    INCL    R13
    JMP     matmul_q8_row_loop

matmul_q8_done:
    /* Restore callee-saved registers */
    MOVQ    0(SP), BP
    MOVQ    8(SP), BX
    MOVQ    16(SP), R12
    MOVQ    24(SP), R13
    MOVQ    32(SP), R14
    MOVQ    40(SP), R15

    RET
