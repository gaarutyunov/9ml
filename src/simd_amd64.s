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

// ============================================================================
// Softmax SIMD Helper Functions
// ============================================================================

// softmax_max_simd(x, size) -> float
// Returns max value in array x[0:size]
//
// Parameters (Plan 9 amd64 stack layout):
//   x:    BP (RARG)  - input array
//   size: 16(SP)     - array length
// Returns: float in X0 (and at ret+24(SP))
TEXT softmax_max_simd(SB), $0
	MOVL	16(SP), CX		// CX = size
	TESTL	CX, CX
	JLE	smax_done_zero		// Return 0 if size <= 0

	// Initialize max with first element
	MOVSS	(BP), X0		// X0 = x[0] (max so far)
	MOVL	$1, AX			// AX = i = 1

	// If size <= 4, use scalar loop
	MOVL	CX, DX
	SUBL	$4, DX
	TESTL	DX, DX
	JLE	smax_remainder

	// Broadcast X0 to all 4 lanes
	SHUFPS	$0x00, X0, X0		// X0 = [max, max, max, max]

	// Main loop: process 4 elements at a time
smax_loop4:
	MOVL	CX, DX
	SUBL	$3, DX			// DX = size - 3
	MOVL	DX, R8
	SUBL	AX, R8			// R8 = (size - 3) - i
	TESTL	R8, R8
	JLE	smax_reduce		// exit if (size - 3) - i <= 0

	MOVUPS	(BP)(AX*4), X1		// X1 = x[i:i+4]
	MAXPS	X1, X0			// X0 = max(X0, X1) elementwise

	ADDL	$4, AX
	JMP	smax_loop4

smax_reduce:
	// Reduce 4 maxes to 1: horizontal max
	MOVAPS	X0, X1
	SHUFPS	$0xB1, X1, X1		// X1 = [X0[1], X0[0], X0[3], X0[2]]
	MAXPS	X1, X0			// X0 = [max01, max01, max23, max23]
	MOVAPS	X0, X1
	SHUFPS	$0x0A, X1, X1		// X1 = [X0[2], X0[2], X0[0], X0[0]]
	MAXSS	X1, X0			// X0[0] = final max

smax_remainder:
	// Handle remaining elements with scalar loop
	MOVL	CX, R8
	SUBL	AX, R8			// R8 = size - i
	TESTL	R8, R8
	JLE	smax_done

	MOVSS	(BP)(AX*4), X1
	MAXSS	X1, X0

	INCL	AX
	JMP	smax_remainder

smax_done:
	MOVSS	X0, 24(SP)		// Store return value
	RET

smax_done_zero:
	XORPS	X0, X0			// Return 0.0
	MOVSS	X0, 24(SP)
	RET

// softmax_sum_simd(x, size) -> float
// Returns sum of all values in array x[0:size]
//
// Parameters:
//   x:    BP (RARG)  - input array
//   size: 16(SP)     - array length
// Returns: float in X0 (and at ret+24(SP))
TEXT softmax_sum_simd(SB), $0
	MOVL	16(SP), CX		// CX = size
	TESTL	CX, CX
	JLE	ssum_done_zero		// Return 0 if size <= 0

	XORPS	X0, X0			// X0 = accumulator
	XORPS	X1, X1			// X1 = accumulator 2
	XORL	AX, AX			// AX = i = 0

	// Main loop: process 8 elements at a time
ssum_loop8:
	MOVL	CX, DX
	SUBL	$7, DX			// DX = size - 7
	MOVL	DX, R8
	SUBL	AX, R8			// R8 = (size - 7) - i
	TESTL	R8, R8
	JLE	ssum_loop4		// exit if (size - 7) - i <= 0

	MOVUPS	(BP)(AX*4), X2
	ADDPS	X2, X0
	MOVUPS	16(BP)(AX*4), X2
	ADDPS	X2, X1

	ADDL	$8, AX
	JMP	ssum_loop8

ssum_loop4:
	MOVL	CX, DX
	SUBL	$3, DX			// DX = size - 3
	MOVL	DX, R8
	SUBL	AX, R8			// R8 = (size - 3) - i
	TESTL	R8, R8
	JLE	ssum_reduce		// exit if (size - 3) - i <= 0

	MOVUPS	(BP)(AX*4), X2
	ADDPS	X2, X0

	ADDL	$4, AX
	JMP	ssum_loop4

ssum_reduce:
	// Combine accumulators
	ADDPS	X1, X0

	// Horizontal sum
	MOVAPS	X0, X1
	SHUFPS	$0xB1, X1, X1
	ADDPS	X1, X0
	MOVAPS	X0, X1
	SHUFPS	$0x0A, X1, X1
	ADDSS	X1, X0

	// Handle remaining elements
ssum_remainder:
	MOVL	CX, R8
	SUBL	AX, R8			// R8 = size - i
	TESTL	R8, R8
	JLE	ssum_done

	MOVSS	(BP)(AX*4), X1
	ADDSS	X1, X0

	INCL	AX
	JMP	ssum_remainder

ssum_done:
	MOVSS	X0, 24(SP)		// Store return value
	RET

ssum_done_zero:
	XORPS	X0, X0
	MOVSS	X0, 24(SP)
	RET

// softmax_scale_simd(x, scale, size)
// Multiplies all elements: x[i] *= scale
//
// Parameters:
//   x:     BP (RARG)  - array to scale (in-place)
//   scale: 16(SP)     - scalar multiplier
//   size:  24(SP)     - array length
TEXT softmax_scale_simd(SB), $0
	MOVSS	16(SP), X1		// X1 = scale
	MOVL	24(SP), CX		// CX = size
	TESTL	CX, CX
	JLE	sscale_done

	// Broadcast scale to all lanes
	SHUFPS	$0x00, X1, X1		// X1 = [s, s, s, s]

	XORL	AX, AX			// AX = i = 0

sscale_loop4:
	MOVL	CX, R8
	SUBL	$3, R8			// R8 = size - 3
	MOVL	R8, R9
	SUBL	AX, R9			// R9 = (size - 3) - i
	TESTL	R9, R9
	JLE	sscale_remainder

	MOVUPS	(BP)(AX*4), X0
	MULPS	X1, X0
	MOVUPS	X0, (BP)(AX*4)

	ADDL	$4, AX
	JMP	sscale_loop4

sscale_remainder:
	MOVL	CX, R9
	SUBL	AX, R9			// R9 = size - i
	TESTL	R9, R9
	JLE	sscale_done

	MOVSS	(BP)(AX*4), X0
	MULSS	X1, X0
	MOVSS	X0, (BP)(AX*4)

	INCL	AX
	JMP	sscale_remainder

sscale_done:
	RET

// softmax_subtract_simd(x, val, size)
// Subtracts scalar from all elements: x[i] -= val
//
// Parameters:
//   x:    BP (RARG)  - array to modify (in-place)
//   val:  16(SP)     - scalar to subtract
//   size: 24(SP)     - array length
TEXT softmax_subtract_simd(SB), $0
	MOVSS	16(SP), X1		// X1 = val
	MOVL	24(SP), CX		// CX = size
	TESTL	CX, CX
	JLE	ssub_done

	// Broadcast val to all lanes
	SHUFPS	$0x00, X1, X1

	XORL	AX, AX			// AX = i = 0

ssub_loop4:
	MOVL	CX, R8
	SUBL	$3, R8			// R8 = size - 3
	MOVL	R8, R9
	SUBL	AX, R9			// R9 = (size - 3) - i
	TESTL	R9, R9
	JLE	ssub_remainder

	MOVUPS	(BP)(AX*4), X0
	SUBPS	X1, X0
	MOVUPS	X0, (BP)(AX*4)

	ADDL	$4, AX
	JMP	ssub_loop4

ssub_remainder:
	MOVL	CX, R9
	SUBL	AX, R9			// R9 = size - i
	TESTL	R9, R9
	JLE	ssub_done

	MOVSS	(BP)(AX*4), X0
	SUBSS	X1, X0
	MOVSS	X0, (BP)(AX*4)

	INCL	AX
	JMP	ssub_remainder

ssub_done:
	RET

// exp_schraudolph_simd(x, size)
// Fast exp approximation using Schraudolph's method
// exp(x) ≈ reinterpret_float((int)(1512775 * x + 1072632447))
// Applied in-place to array x
//
// Parameters:
//   x:    BP (RARG)  - array to transform (in-place)
//   size: 16(SP)     - array length
//
// Schraudolph constants:
//   a = 2^23 / ln(2) = 12102203.161561 ≈ 12102203 (round down for stability)
//   b = 2^23 * (127 - c) where c ≈ 0.045 for bias correction
//   b = 2^23 * 126.96 ≈ 1064866805
//
// For single precision (float32):
//   a = 1512775.21 ≈ 1512775
//   b = 1072632447 (IEEE 754 representation of e^0 = 1.0 is close)
TEXT exp_schraudolph_simd(SB), $0
	MOVL	16(SP), CX		// CX = size
	TESTL	CX, CX
	JLE	schr_done

	// Load constants for Schraudolph approximation
	// a = 12102203.0f = 2^23/ln(2) = 0x4B38AA5B
	// b = 1064866805 = 127*2^23 - 486411 (bias with correction)
	MOVL	$0x4B38AA5B, R8		// 12102203.0f in IEEE754
	MOVL	R8, R9
	SHLQ	$32, R9
	ORQ	R8, R9			// R9 has 12102203.0f in both halves
	MOVQ	R9, X1			// Load into low 64 bits
	MOVLHPS	X1, X1			// Broadcast to all 4 lanes

	// b = 1064866805 = 0x3F7D6F8D (includes correction term)
	MOVL	$1064866805, R8
	MOVL	R8, R9
	SHLQ	$32, R9
	ORQ	R8, R9
	MOVQ	R9, X2
	MOVLHPS	X2, X2			// X2 = [b, b, b, b]

	XORL	AX, AX			// AX = i = 0

schr_loop4:
	MOVL	CX, R8
	SUBL	$3, R8
	MOVL	R8, R9
	SUBL	AX, R9
	TESTL	R9, R9
	JLE	schr_remainder

	MOVUPS	(BP)(AX*4), X0		// X0 = x[i:i+4]
	MULPS	X1, X0			// X0 = a * x
	// CVTTPS2DQ X0, X0: 66 0F 5B C0 (convert packed floats to ints with truncation)
	BYTE $0x66; BYTE $0x0F; BYTE $0x5B; BYTE $0xC0
	// PADDD X0, X2: 66 0F FE C2 (add packed dwords: X0 += X2)
	// ModR/M C2 = 11 000 010 (reg=X0=000, r/m=X2=010)
	BYTE $0x66; BYTE $0x0F; BYTE $0xFE; BYTE $0xC2
	// X0 now contains the IEEE 754 bit pattern to reinterpret as floats
	// PADDD result stays as the float bits when we store
	MOVUPS	X0, (BP)(AX*4)		// Store (reinterpret as float)

	ADDL	$4, AX
	JMP	schr_loop4

schr_remainder:
	MOVL	CX, R9
	SUBL	AX, R9
	TESTL	R9, R9
	JLE	schr_done

	// Scalar fallback for remaining elements
	MOVSS	(BP)(AX*4), X0
	MULSS	X1, X0
	// CVTTSS2SI R10, X0: F3 4C 0F 2C D0 (convert scalar float to 64-bit int)
	// REX.WR=0x4C (W=1 for 64-bit, R=1 for R10), ModR/M D0 = reg=R10[2:0]=010, r/m=X0=000
	BYTE $0xF3; BYTE $0x4C; BYTE $0x0F; BYTE $0x2C; BYTE $0xD0
	ADDL	$1064866805, R10	// Add bias
	MOVL	R10, (BP)(AX*4)		// Store (reinterpret as float)

	INCL	AX
	JMP	schr_remainder

schr_done:
	RET

// exp_poly_simd(x, size)
// Polynomial exp approximation using range reduction + 4th degree polynomial
// exp(x) = 2^i * exp(f) where i = floor(x/ln2), f = x - i*ln2
// exp(f) ≈ 1 + f + f²/2 + f³/6 + f⁴/24 for f in [0, ln2)
//
// Parameters:
//   x:    BP (RARG)  - array to transform (in-place)
//   size: 16(SP)     - array length
TEXT exp_poly_simd(SB), $0
	MOVL	16(SP), CX		// CX = size
	TESTL	CX, CX
	JLE	poly_done

	// Load constants for polynomial evaluation
	// ln2_inv = 1/ln(2) = 1.4426950408889634 = 0x3FB8AA3B
	MOVL	$0x3FB8AA3B, R8
	MOVL	R8, R9
	SHLQ	$32, R9
	ORQ	R8, R9
	MOVQ	R9, X1
	MOVLHPS	X1, X1			// X1 = ln2_inv broadcast

	// ln2 = 0.6931471805599453 = 0x3F317218
	MOVL	$0x3F317218, R8
	MOVL	R8, R9
	SHLQ	$32, R9
	ORQ	R8, R9
	MOVQ	R9, X2
	MOVLHPS	X2, X2			// X2 = ln2 broadcast

	// c1 = 1.0 = 0x3F800000 (coefficient for f^1)
	MOVL	$0x3F800000, R8
	MOVL	R8, R9
	SHLQ	$32, R9
	ORQ	R8, R9
	MOVQ	R9, X3
	MOVLHPS	X3, X3			// X3 = 1.0 broadcast

	// c2 = 0.5 = 0x3F000000 (coefficient for f^2)
	MOVL	$0x3F000000, R8
	MOVL	R8, R9
	SHLQ	$32, R9
	ORQ	R8, R9
	MOVQ	R9, X4
	MOVLHPS	X4, X4			// X4 = 0.5 broadcast

	// c3 = 1/6 ≈ 0.166666667 = 0x3E2AAAAB (coefficient for f^3)
	MOVL	$0x3E2AAAAB, R8
	MOVL	R8, R9
	SHLQ	$32, R9
	ORQ	R8, R9
	MOVQ	R9, X5
	MOVLHPS	X5, X5			// X5 = 1/6 broadcast

	// c4 = 1/24 ≈ 0.041666668 = 0x3D2AAAAB (coefficient for f^4)
	MOVL	$0x3D2AAAAB, R8
	MOVL	R8, R9
	SHLQ	$32, R9
	ORQ	R8, R9
	MOVQ	R9, X6
	MOVLHPS	X6, X6			// X6 = 1/24 broadcast

	XORL	AX, AX			// AX = i = 0

poly_loop1:
	// Process one element at a time for simplicity (vectorize later if needed)
	MOVL	CX, R8
	SUBL	AX, R8
	TESTL	R8, R8
	JLE	poly_done

	// Load x
	MOVSS	(BP)(AX*4), X0		// X0 = x

	// i = floor(x / ln2)
	MOVAPS	X0, X7
	MULSS	X1, X7			// X7 = x * ln2_inv
	// Floor: convert to int (truncate toward zero), then adjust for negative
	// CVTTSS2SI R10, X7: F3 4C 0F 2C D7 (convert X7 to int64 in R10)
	// REX.WR=0x4C (W=1, R=1), ModR/M D7 = reg=R10[2:0]=010, r/m=X7=111
	BYTE $0xF3; BYTE $0x4C; BYTE $0x0F; BYTE $0x2C; BYTE $0xD7
	// Check if we need to subtract 1 for negative numbers
	MOVL	R10, R11
	SARL	$31, R11		// R11 = -1 if R10 < 0, else 0
	// For proper floor, if original was negative and not exact, subtract 1
	// CVTSI2SS X7, R10: F3 49 0F 2A FA (convert R10 to float in X7)
	// REX.WB=0x49 (W=1, B=1), ModR/M FA = reg=X7=111, r/m=R10[2:0]=010
	BYTE $0xF3; BYTE $0x49; BYTE $0x0F; BYTE $0x2A; BYTE $0xFA
	MOVAPS	X0, X8
	MULSS	X1, X8			// X8 = x * ln2_inv
	// Skip floor adjustment for now - simpler approximation

	// f = x - i * ln2
	// CVTSI2SS X7, R10: F3 49 0F 2A FA (convert R10 to float in X7)
	BYTE $0xF3; BYTE $0x49; BYTE $0x0F; BYTE $0x2A; BYTE $0xFA
	MOVAPS	X7, X8
	MULSS	X2, X8			// X8 = i * ln2
	MOVAPS	X0, X7
	SUBSS	X8, X7			// X7 = f = x - i * ln2

	// Polynomial: 1 + f*(1 + f*(0.5 + f*(1/6 + f*1/24)))
	// Horner's method from inside out
	MOVAPS	X7, X8			// X8 = f
	MULSS	X6, X8			// X8 = f * (1/24)
	ADDSS	X5, X8			// X8 = 1/6 + f*(1/24)
	MULSS	X7, X8			// X8 = f * (1/6 + f*(1/24))
	ADDSS	X4, X8			// X8 = 0.5 + f*(...)
	MULSS	X7, X8			// X8 = f * (0.5 + ...)
	ADDSS	X3, X8			// X8 = 1 + f*(...)
	MULSS	X7, X8			// X8 = f * (1 + ...)
	ADDSS	X3, X8			// X8 = 1 + f*(...) = exp(f)

	// Result = exp(f) * 2^i
	// 2^i is done by adding i to the exponent field of the float
	// exponent = i + 127 (bias), then shift left 23 bits
	ADDL	$127, R10		// R10 = biased exponent
	// Clamp to valid range [0, 255]
	TESTL	R10, R10
	JL	poly_underflow
	CMPL	R10, $255
	JGE	poly_overflow
	SHLL	$23, R10		// R10 = 2^i in IEEE 754 format
	// MOVD X9, R10d: 66 45 0F 6E CA (move 32-bit from R10 to X9)
	// REX.RB=0x45 (R=1 for X9, B=1 for R10), ModR/M CA = reg=X9[2:0]=001, r/m=R10[2:0]=010
	BYTE $0x66; BYTE $0x45; BYTE $0x0F; BYTE $0x6E; BYTE $0xCA
	MULSS	X9, X8			// X8 = exp(f) * 2^i

	MOVSS	X8, (BP)(AX*4)		// Store result
	INCL	AX
	JMP	poly_loop1

poly_underflow:
	// Underflow: return 0
	XORPS	X8, X8
	MOVSS	X8, (BP)(AX*4)
	INCL	AX
	JMP	poly_loop1

poly_overflow:
	// Overflow: return large value
	MOVL	$0x7F7FFFFF, R10	// Max float
	MOVL	R10, (BP)(AX*4)
	INCL	AX
	JMP	poly_loop1

poly_done:
	RET
