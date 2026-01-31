// Snapdragon 480 optimized kernels using dot product instructions
// ARM64 assembly with UDOT/SDOT acceleration for quantized operations

#ifdef __aarch64__

// Enable dot product compilation target
.arch armv8.2-a+dotprod
.fpu crypto-neon-fp-armv8

// GEMM microkernel: 1x8 with Snapdragon 480 optimizations
// Optimized for Cortex-A76 with dot product instructions and L3 cache prefetching

.text
.align 4
.global xnn_qs8_gemm_minmax_ukernel_1x8__snapdragon480_dotprod_ld128
.type xnn_qs8_gemm_minmax_ukernel_1x8__snapdragon480_dotprod_ld128, %function

xnn_qs8_gemm_minmax_ukernel_1x8__snapdragon480_dotprod_ld128:
    // Arguments:
    // x0: mr (rows in matrix A)
    // x1: nr (columns in matrix B)
    // x2: k (inner dimension)
    // x3: a (matrix A)
    // x4: a_stride
    // x5: w (matrix B/weights)
    // x6: w_stride
    // x7: bias
    // stack: c, c_stride, params

    // Snapdragon 480 specific optimizations:
    // - Use UDOT instructions for quantized GEMM operations
    // - Prefetch data for L3 cache optimization
    // - Optimize for Cortex-A76 microarchitecture
    // - Use 128-bit loads for better memory bandwidth

    stp d8, d9, [sp, -64]!
    stp d10, d11, [sp, 16]
    stp d12, d13, [sp, 32]
    stp d14, d15, [sp, 48]

    // Load parameters from stack
    ldr x8, [sp, 64]      // c
    ldr x9, [sp, 72]      // c_stride
    ldr x10, [sp, 80]     // params

    // Initialize accumulators (8 columns)
    movi v0.4s, #0         // acc0-acc3
    movi v1.4s, #0         // acc4-acc7

    // Load quantization parameters
    ldr w11, [x10, 0]     // input_zero_point
    ldr w12, [x10, 4]     // kernel_zero_point
    ldr s2, [x10, 8]      // multiplier
    ldr w13, [x10, 12]    // output_zero_point
    ldr w14, [x10, 16]    // output_min
    ldr w15, [x10, 20]    // output_max

    // Duplicate quantization parameters to vectors
    dup v2.16b, w11       // input_zero_point
    dup v3.16b, w12       // kernel_zero_point
    dup v4.4s, w13        // output_zero_point
    dup v5.4s, w14        // output_min
    dup v6.4s, w15        // output_max

    // Main computation loop
    cmp x2, #0
    beq .Lgemm_end

.Lgemm_loop_k:
    // Prefetch next data for L3 cache (Snapdragon 480 optimization)
    prfm pldl3keep, [x3, 64]
    prfm pldl3keep, [x5, 128]

    // Load 1 row of A (8-bit quantized)
    ld1 {v7.8b}, [x3], #8

    // Subtract input zero point
    usubl v7.8h, v7.8b, v2.8b

    // Load 8 columns of B (8-bit quantized weights)
    ld1 {v16.8b, v17.8b, v18.8b, v19.8b}, [x5], #32
    ld1 {v20.8b, v21.8b, v22.8b, v23.8b}, [x5], #32

    // Subtract kernel zero point
    usubl v16.8h, v16.8b, v3.8b
    usubl v17.8h, v17.8b, v3.8b
    usubl v18.8h, v18.8b, v3.8b
    usubl v19.8h, v19.8b, v3.8b
    usubl v20.8h, v20.8b, v3.8b
    usubl v21.8h, v21.8b, v3.8b
    usubl v22.8h, v22.8b, v3.8b
    usubl v23.8h, v23.8b, v3.8b

    // Dot product operations (Snapdragon 480 key optimization)
    // UDOT: Unsigned dot product with accumulation
    udot v0.4s, v7.16b, v16.16b
    udot v1.4s, v7.16b, v20.16b

    // Continue with remaining columns...
    udot v0.4s, v7.16b, v17.16b
    udot v1.4s, v7.16b, v21.16b

    udot v0.4s, v7.16b, v18.16b
    udot v1.4s, v7.16b, v22.16b

    udot v0.4s, v7.16b, v19.16b
    udot v1.4s, v7.16b, v23.16b

    // Decrement k and loop
    subs x2, x2, #8
    bgt .Lgemm_loop_k

.Lgemm_end:
    // Convert to float and apply quantization
    scvtf v0.4s, v0.4s
    scvtf v1.4s, v1.4s

    // Apply multiplier
    fmul v0.4s, v0.4s, v2.4s
    fmul v1.4s, v1.4s, v2.4s

    // Add bias if provided
    cbz x7, .Lno_bias
    ld1 {v7.4s}, [x7]
    fadd v0.4s, v0.4s, v7.4s
    ld1 {v7.4s}, [x7, 16]
    fadd v1.4s, v1.4s, v7.4s

.Lno_bias:
    // Add output zero point
    fadd v0.4s, v0.4s, v4.4s
    fadd v1.4s, v1.4s, v4.4s

    // Clamp to output range
    fmax v0.4s, v0.4s, v5.4s
    fmax v1.4s, v1.4s, v5.4s
    fmin v0.4s, v0.4s, v6.4s
    fmin v1.4s, v1.4s, v6.4s

    // Convert to int8 and store
    fcvtzs v0.4s, v0.4s
    fcvtzs v1.4s, v1.4s

    // Store results
    st1 {v0.8b}, [x8], #8

    // Restore registers and return
    ldp d14, d15, [sp, 48]
    ldp d12, d13, [sp, 32]
    ldp d10, d11, [sp, 16]
    ldp d8, d9, [sp]
    add sp, sp, 64
    ret

.size xnn_qs8_gemm_minmax_ukernel_1x8__snapdragon480_dotprod_ld128, . - xnn_qs8_gemm_minmax_ukernel_1x8__snapdragon480_dotprod_ld128

#endif // __aarch64__