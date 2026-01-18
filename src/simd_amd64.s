// SIMD vectorized routines for Plan 9 amd64
// Uses SSE packed instructions for 4x float parallelism

// Plan 9 amd64 calling convention:
// - First integer/pointer arg is in BP (RARG)
// - Subsequent args are on stack at +8(FP), +16(FP), etc.
// - Return value in AX, float return in X0

// matmul_simd(xout, x, w, n, d)
// W (d,n) @ x (n,) -> xout (d,)
//
// Parameters (Plan 9 amd64):
//   xout: BP (RARG)   - output vector (d floats)
//   x:    +16(FP)     - input vector (n floats)
//   w:    +24(FP)     - weight matrix (d*n floats, row-major)
//   n:    +32(FP)     - inner dimension (columns)
//   d:    +40(FP)     - outer dimension (rows)
//
// Note: Stack layout verified empirically. First arg in BP, then
// there's 8 bytes of padding/frame info, then remaining args.
//
// Uses SSE packed operations for 4x speedup.
TEXT matmul_simd(SB), $0
	// Load parameters using explicit SP offsets (verified by stack dump)
	// Stack layout: 0(SP)=retaddr, 8(SP)=pad, 16(SP)=x, 24(SP)=w, 32(SP)=n, 40(SP)=d
	MOVQ	BP, DI			// DI = xout (first arg in RARG/BP)
	MOVQ	16(SP), SI		// SI = input vector x
	MOVQ	24(SP), DX		// DX = weight matrix w
	MOVL	32(SP), CX		// CX = n (inner dim)
	MOVL	40(SP), R8		// R8 = d (outer dim)

	XORL	R9, R9			// R9 = i = 0 (row counter)

matmul_row_loop:
	// Check if we've processed all rows
	MOVL	R8, R12			// R12 = d
	SUBL	R9, R12			// R12 = d - i
	TESTL	R12, R12		// set flags based on R12
	JLE	matmul_done		// exit if d - i <= 0 (i.e., i >= d)

	// Initialize accumulators to 0.0
	XORPS	X0, X0			// X0 = accumulator 0
	XORPS	X1, X1			// X1 = accumulator 1

	XORL	R10, R10		// R10 = j = 0 (column counter)
	MOVQ	DX, R11			// R11 = current row pointer

	// Main loop: process 8 elements at a time with 2 accumulators
matmul_col_loop8:
	MOVL	CX, R12
	SUBL	$7, R12			// R12 = n - 7
	MOVL	R12, R13		// R13 = n - 7
	SUBL	R10, R13		// R13 = (n - 7) - j
	TESTL	R13, R13
	JLE	matmul_col_loop4	// exit 8-loop if (n-7) - j <= 0

	// Load 8 weights and 8 inputs, multiply-add
	MOVUPS	(R11)(R10*4), X2	// w[j:j+4]
	MOVUPS	(SI)(R10*4), X3		// x[j:j+4]
	MULPS	X3, X2			// w * x
	ADDPS	X2, X0			// acc0 += w * x

	MOVUPS	16(R11)(R10*4), X2	// w[j+4:j+8]
	MOVUPS	16(SI)(R10*4), X3	// x[j+4:j+8]
	MULPS	X3, X2
	ADDPS	X2, X1			// acc1 += w * x

	ADDL	$8, R10
	JMP	matmul_col_loop8

	// Process 4 elements at a time
matmul_col_loop4:
	MOVL	CX, R12
	SUBL	$3, R12			// R12 = n - 3
	MOVL	R12, R13		// R13 = n - 3
	SUBL	R10, R13		// R13 = (n - 3) - j
	TESTL	R13, R13
	JLE	matmul_col_remainder	// exit 4-loop if (n-3) - j <= 0

	MOVUPS	(R11)(R10*4), X2	// w[j:j+4]
	MOVUPS	(SI)(R10*4), X3		// x[j:j+4]
	MULPS	X3, X2
	ADDPS	X2, X0

	ADDL	$4, R10
	JMP	matmul_col_loop4

matmul_col_remainder:
	// Combine accumulators (only once)
	ADDPS	X1, X0			// X0 = acc0 + acc1

	// Process remaining elements one by one
matmul_col_remainder_loop:
	MOVL	CX, R13			// R13 = n
	SUBL	R10, R13		// R13 = n - j
	TESTL	R13, R13
	JLE	matmul_horizontal_sum	// exit if n - j <= 0

	MOVSS	(R11)(R10*4), X2	// Load single weight
	MOVSS	(SI)(R10*4), X3		// Load single input
	MULSS	X3, X2			// Multiply
	ADDSS	X2, X0			// Add to accumulator (low element)

	INCL	R10
	JMP	matmul_col_remainder_loop

matmul_horizontal_sum:
	// Horizontal sum: X0 = [a, b, c, d] -> a + b + c + d
	MOVAPS	X0, X1
	SHUFPS	$0xB1, X1, X1		// X1 = [b, a, d, c]
	ADDPS	X1, X0			// X0 = [a+b, b+a, c+d, d+c]
	MOVAPS	X0, X1
	SHUFPS	$0x0A, X1, X1		// X1 = [c+d, c+d, a+b, a+b]
	ADDSS	X1, X0			// X0[0] = a+b+c+d

	// Store result to xout[i]
	MOVSS	X0, (DI)(R9*4)

	// Advance to next row: w += n*4 bytes
	MOVL	CX, R12
	SHLQ	$2, R12			// R12 = n * 4
	ADDQ	R12, DX			// DX += n*4

	INCL	R9			// i++
	JMP	matmul_row_loop

matmul_done:
	RET

// dot_product_simd(a, b, n) -> float
// Returns sum(a[i] * b[i]) for i in 0..n
//
// Parameters (Plan 9 amd64 stack layout):
//   a: BP (RARG)   - first vector
//   b: 16(SP)      - second vector
//   n: 24(SP)      - length
// Returns: float in X0 (and stored at ret+32(SP))
TEXT dot_product_simd(SB), $0
	// a is in BP (RARG)
	MOVQ	16(SP), SI		// SI = b
	MOVL	24(SP), CX		// CX = n

	XORPS	X0, X0			// X0 = accumulator
	XORPS	X1, X1			// X1 = accumulator 2
	XORL	AX, AX			// AX = i = 0

dot_loop8:
	MOVL	CX, DX
	SUBL	$7, DX			// DX = n - 7
	MOVL	DX, R8			// R8 = n - 7
	SUBL	AX, R8			// R8 = (n - 7) - i
	TESTL	R8, R8
	JLE	dot_loop4		// exit 8-loop if (n-7) - i <= 0

	MOVUPS	(BP)(AX*4), X2
	MOVUPS	(SI)(AX*4), X3
	MULPS	X3, X2
	ADDPS	X2, X0

	MOVUPS	16(BP)(AX*4), X2
	MOVUPS	16(SI)(AX*4), X3
	MULPS	X3, X2
	ADDPS	X2, X1

	ADDL	$8, AX
	JMP	dot_loop8

dot_loop4:
	MOVL	CX, DX
	SUBL	$3, DX			// DX = n - 3
	MOVL	DX, R8			// R8 = n - 3
	SUBL	AX, R8			// R8 = (n - 3) - i
	TESTL	R8, R8
	JLE	dot_remainder		// exit 4-loop if (n-3) - i <= 0

	MOVUPS	(BP)(AX*4), X2
	MOVUPS	(SI)(AX*4), X3
	MULPS	X3, X2
	ADDPS	X2, X0

	ADDL	$4, AX
	JMP	dot_loop4

dot_remainder:
	ADDPS	X1, X0			// Combine accumulators (only once)

dot_remainder_loop:
	MOVL	CX, R8			// R8 = n
	SUBL	AX, R8			// R8 = n - i
	TESTL	R8, R8
	JLE	dot_horizontal		// exit if n - i <= 0

	MOVSS	(BP)(AX*4), X2
	MOVSS	(SI)(AX*4), X3
	MULSS	X3, X2
	ADDSS	X2, X0

	INCL	AX
	JMP	dot_remainder_loop

dot_horizontal:
	// Horizontal sum
	MOVAPS	X0, X1
	SHUFPS	$0xB1, X1, X1
	ADDPS	X1, X0
	MOVAPS	X0, X1
	SHUFPS	$0x0A, X1, X1
	ADDSS	X1, X0

	// Return value in X0 and at ret+32(SP)
	MOVSS	X0, 32(SP)
	RET

// rmsnorm_simd(o, x, weight, size)
// RMS normalization: o[i] = weight[i] * (x[i] / rms(x))
//
// Parameters (Plan 9 amd64 stack layout):
//   o:      BP (RARG)  - output vector
//   x:      16(SP)     - input vector
//   weight: 24(SP)     - weight vector
//   size:   32(SP)     - vector size
//
// Note: Using $0 frame (no stack locals) to match matmul_simd.
// All temp values stored in registers.
TEXT rmsnorm_simd(SB), $0
	MOVQ	BP, DI			// DI = output
	MOVQ	16(SP), SI		// SI = input x
	MOVQ	24(SP), DX		// DX = weight
	MOVL	32(SP), CX		// CX = size

	// Validate pointers
	TESTQ	DI, DI
	JZ	rms_bad_ptr
	TESTQ	SI, SI
	JZ	rms_bad_ptr
	TESTQ	DX, DX
	JZ	rms_bad_ptr
	TESTL	CX, CX
	JLE	rms_bad_size

	// Save size in R14 for later use
	MOVL	CX, R14

	// First pass: compute sum of squares using SSE
	XORPS	X0, X0			// X0 = ss accumulator
	XORL	AX, AX			// AX = i = 0

rms_ss_loop4:
	MOVL	CX, R8
	SUBL	$3, R8			// R8 = size - 3
	MOVL	R8, R9			// R9 = size - 3
	SUBL	AX, R9			// R9 = (size - 3) - i
	TESTL	R9, R9
	JLE	rms_ss_remainder	// exit 4-loop if (size-3) - i <= 0

	MOVUPS	(SI)(AX*4), X2		// X2 = x[i:i+4]
	MULPS	X2, X2			// X2 = x^2
	ADDPS	X2, X0			// ss += x^2

	ADDL	$4, AX
	JMP	rms_ss_loop4

rms_ss_remainder:
	MOVL	CX, R9			// R9 = size
	SUBL	AX, R9			// R9 = size - i
	TESTL	R9, R9
	JLE	rms_compute_scale	// exit if size - i <= 0

	MOVSS	(SI)(AX*4), X2
	MULSS	X2, X2
	ADDSS	X2, X0

	INCL	AX
	JMP	rms_ss_remainder

rms_compute_scale:
	// Horizontal sum of X0
	MOVAPS	X0, X1
	SHUFPS	$0xB1, X1, X1
	ADDPS	X1, X0
	MOVAPS	X0, X1
	SHUFPS	$0x0A, X1, X1
	ADDSS	X1, X0
	// X0[0] = ss (sum of squares)

	// Divide by size: ss / size
	// Convert size (in R14) to float using CVTSI2SS reg-to-reg
	XORPS	X1, X1
	// CVTSI2SS: F3 0F 2A /r - Convert Dword Integer to Scalar Single
	// We want: CVTSI2SS R14, X1  (convert R14 to float in X1)
	// ModR/M = C6 = 11 000 110 = reg=X1, r/m=R14 (need REX.B for R14)
	// REX.W for 64-bit source: 48, REX.WB for R14: 49
	// F3 49 0F 2A CE = CVTSI2SS X1, R14
	BYTE $0xF3; BYTE $0x49; BYTE $0x0F; BYTE $0x2A; BYTE $0xCE
	DIVSS	X1, X0			// X0 = ss / size

	// Add epsilon (1e-5 = 0x3727C5AC in IEEE 754)
	// Load via MOVD from register
	MOVL	$0x3727C5AC, R8
	// MOVD R8d, X2: 66 41 0F 6E D0 (need REX.B for R8)
	BYTE $0x66; BYTE $0x41; BYTE $0x0F; BYTE $0x6E; BYTE $0xD0
	ADDSS	X2, X0			// X0 = ss/size + 1e-5

	// Compute 1/sqrt using exact SQRTSS + DIVSS (matches scalar sqrtf)
	// SQRTSS X0, X1: F3 0F 51 C8 (X1 = sqrt(X0))
	BYTE $0xF3; BYTE $0x0F; BYTE $0x51; BYTE $0xC8
	// Now X1 = sqrt(ss/size + 1e-5)
	// We need 1/sqrt, so: load 1.0 into X0 and divide
	MOVL	$0x3F800000, R8		// 1.0f in IEEE 754
	BYTE $0x66; BYTE $0x41; BYTE $0x0F; BYTE $0x6E; BYTE $0xC0  // MOVD R8d, X0
	DIVSS	X1, X0			// X0 = 1.0 / sqrt(ss/size + 1e-5)
	MOVAPS	X0, X1			// X1 = scale (for broadcast)

	// Broadcast scale to all lanes
	SHUFPS	$0x00, X1, X1		// X1 = [scale, scale, scale, scale]

	// Second pass: normalize and scale (4 at a time)
	XORL	AX, AX			// AX = i = 0

rms_norm_loop4:
	MOVL	CX, R8
	SUBL	$3, R8			// R8 = size - 3
	MOVL	R8, R9			// R9 = size - 3
	SUBL	AX, R9			// R9 = (size - 3) - i
	TESTL	R9, R9
	JLE	rms_norm_remainder	// exit 4-loop if (size-3) - i <= 0

	MOVUPS	(SI)(AX*4), X2		// X2 = x[i:i+4]
	MULPS	X1, X2			// X2 = scale * x
	MOVUPS	(DX)(AX*4), X3		// X3 = weight[i:i+4]
	MULPS	X3, X2			// X2 = weight * scale * x
	MOVUPS	X2, (DI)(AX*4)		// o[i:i+4] = result

	ADDL	$4, AX
	JMP	rms_norm_loop4

rms_norm_remainder:
	MOVL	CX, R9			// R9 = size
	SUBL	AX, R9			// R9 = size - i
	TESTL	R9, R9
	JLE	rms_done		// exit if size - i <= 0

	MOVSS	(SI)(AX*4), X2
	MULSS	X1, X2
	MOVSS	(DX)(AX*4), X3
	MULSS	X3, X2
	MOVSS	X2, (DI)(AX*4)

	INCL	AX
	JMP	rms_norm_remainder

rms_done:
	RET

rms_bad_ptr:
	// Bad pointer detected - just return without doing anything
	RET

rms_bad_size:
	// Bad size detected - just return without doing anything
	RET

// vec_add_simd(o, a, b, n)
// Vector addition: o[i] = a[i] + b[i]
//
// Parameters (Plan 9 amd64 stack layout):
//   o: BP (RARG)  - output vector
//   a: 16(SP)     - first input vector
//   b: 24(SP)     - second input vector
//   n: 32(SP)     - vector length
TEXT vec_add_simd(SB), $0
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVL	32(SP), CX
	XORL	AX, AX

vadd_loop4:
	MOVL	CX, R8
	SUBL	$3, R8			// R8 = n - 3
	MOVL	R8, R9			// R9 = n - 3
	SUBL	AX, R9			// R9 = (n - 3) - i
	TESTL	R9, R9
	JLE	vadd_remainder		// exit 4-loop if (n-3) - i <= 0

	MOVUPS	(SI)(AX*4), X0
	MOVUPS	(DX)(AX*4), X1
	ADDPS	X1, X0
	MOVUPS	X0, (BP)(AX*4)

	ADDL	$4, AX
	JMP	vadd_loop4

vadd_remainder:
	MOVL	CX, R9			// R9 = n
	SUBL	AX, R9			// R9 = n - i
	TESTL	R9, R9
	JLE	vadd_done		// exit if n - i <= 0

	MOVSS	(SI)(AX*4), X0
	MOVSS	(DX)(AX*4), X1
	ADDSS	X1, X0
	MOVSS	X0, (BP)(AX*4)

	INCL	AX
	JMP	vadd_remainder

vadd_done:
	RET

// vec_scale_simd(o, x, scalar, n)
// Scalar-vector multiplication: o[i] = scalar * x[i]
//
// Parameters (Plan 9 amd64 stack layout):
//   o:      BP (RARG) - output vector
//   x:      16(SP)    - input vector
//   scalar: 24(SP)    - scalar value (float)
//   n:      32(SP)    - vector length
TEXT vec_scale_simd(SB), $0
	MOVQ	16(SP), SI
	MOVSS	24(SP), X1		// X1 = scalar
	MOVL	32(SP), CX
	XORL	AX, AX

	// Broadcast scalar to all lanes
	SHUFPS	$0x00, X1, X1		// X1 = [s, s, s, s]

vscale_loop4:
	MOVL	CX, R8
	SUBL	$3, R8			// R8 = n - 3
	MOVL	R8, R9			// R9 = n - 3
	SUBL	AX, R9			// R9 = (n - 3) - i
	TESTL	R9, R9
	JLE	vscale_remainder	// exit 4-loop if (n-3) - i <= 0

	MOVUPS	(SI)(AX*4), X0
	MULPS	X1, X0
	MOVUPS	X0, (BP)(AX*4)

	ADDL	$4, AX
	JMP	vscale_loop4

vscale_remainder:
	MOVL	CX, R9			// R9 = n
	SUBL	AX, R9			// R9 = n - i
	TESTL	R9, R9
	JLE	vscale_done		// exit if n - i <= 0

	MOVSS	(SI)(AX*4), X0
	MULSS	X1, X0
	MOVSS	X0, (BP)(AX*4)

	INCL	AX
	JMP	vscale_remainder

vscale_done:
	RET

// NOTE: softmax_simd is not implemented - use C scalar version
