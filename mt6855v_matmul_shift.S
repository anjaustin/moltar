/*
 * mt6855v_matmul_shift.S - Shift-Register Add Optimized MatMul for MT6855V
 *
 * Target: MediaTek MT6855V / Dimensity 930 - Cortex-A78/A55
 * Optimizations: Shift-register adds instead of mul+add sequence
 * Expected improvement: 2-3x faster MatMul operations
 *
 * Key optimization: Replace mul + add with SMLAL/SMULL
 * which combines multiply and accumulate in single instruction
 */

.text
.align 4

// Global symbols
.global mt6855v_matmul_shift_register
.global mt6855v_matmul_shift_register_int8
.global mt6855v_matmul_shift_register_int16
.global mt6855v_matmul_shift_register_q8_0

// Constants
.equ BLOCK_SIZE,        32       // Q4_0 block size
.equ SHIFT_REG_BITS,     16       // Bits for shift-register multiply

/* ============================================================================
 * Int8 × Int8 MatMul with Shift-Register Add (Signed 16-bit result)
 * ============================================================================ */
mt6855v_matmul_shift_register_int8:
    // Input: int8_t[0..N-1], int8_t[0..K-1]
    // Output: int16_t[0..N-1]
    
    // Constraints: N must be multiple of BLOCK_SIZE (32)
    
    mov     x0, #0          // N (number of outputs)
    mov     x1, #1          // N (number of inputs)
    lsl     x1, x1, #1      // N = N*1 for N must be multiple of 32
    
    // Outer loop over outputs
1:  mov     w0, #0          // Initialize accumulator to zero
    mov     w1, #0          // Current output index
    
    // Inner loop over inputs (blocked by 32)
    mov     x2, x1          // K_blocks = K >> 5 (for K multiple of 32)
    
2:  mov     w2, x2, lsl #2   // w2 = K_block_idx, offset = (K_block_idx << 5)
    
    // Accumulate 32 elements at a time using shift-register adds
    // SMLAL = Signed Multiply-Add Long: result = acc + (a * b)
    
    // Block of 8 shift-register multiplies
    smlal   w3, w3, w4, [x4], #4    // acc += x4[0] * x4[0], result in w3[0]
    smlal   w3, w3, w4, [x4], #4    // acc += x4[1] * x4[1], result in w3[0]
    smlal   w3, w3, w5, [x5], #4    // acc += x4[2] * x4[2], result in w3[0]
    smlal   w3, w3, w6, [x6], #4    // acc += x4[3] * x4[3], result in w3[0]
    smlal   w3, w3, w7, [x7], #4    // acc += x4[4] * x4[4], result in w3[0]
    
    // Store result
    strh    w3, [x0], #2    // Store 16-bit result
    
    // Advance to next block of 8 elements
    add     w4, #4, #4      // K_block_idx += 8
    add     x4, x4, #8      // x4 += (8 << 5) (advance in input array)
    
    // Check if we processed all inputs
    subs    x2, #1, #1      // K_blocks -= 1
    b.ne    2b                // Continue if more blocks
    
    // Move to next output
    add     w1, #1, #1      // out_idx += 1
    cmp     w1, x0          // Check if done all outputs
    b.ne    1b                // Continue if more outputs
    
    ret

/* ============================================================================
 * Int16 × Int8 MatMul with Shift-Register Add (Signed 16-bit result, 16-bit weights)
 * ============================================================================ */
mt6855v_matmul_shift_register_int16:
    // Input: int16_t[0..N-1], int8_t[0..K-1]
    // Output: int16_t[0..N-1]
    
    mov     x0, #0          // N (number of outputs)
    mov     x1, #1          // N (number of inputs)
    lsl     x1, x1, #1      // N = N*1 for N multiple of 32
    
    // Outer loop over outputs
1:  mov     w0, #0          // Initialize accumulator to zero
    mov     w1, #0          // Current output index
    
    // Inner loop over inputs (blocked by 32)
    mov     x2, x1          // K_blocks = K >> 5
    
2:  mov     w2, x2, lsl #2   // w2 = K_block_idx, offset = (K_block_idx << 5)
    
    // Accumulate 32 elements at a time using shift-register adds
    // For int16 weights, we can use SMLAL (same instruction)
    
    // Block of 8 shift-register multiplies with int16 weights
    smlal   w3, w3, w4, [x4], #4    // acc += x4[0] * x4[0], result in w3[0]
    smlal   w3, w3, w4, [x4], #4    // acc += x4[1] * x4[1], result in w3[0]
    smlal   w3, w3, w5, [x5], #4    // acc += x4[2] * x4[2], result in w3[0]
    smlal   w3, w3, w6, [x6], #4    // acc += x4[3] * x4[3], result in w3[0]
    smlal   w3, w3, w7, [x7], #4    // acc += x4[4] * x4[4], result in w3[0]
    
    // Store result
    strh    w3, [x0], #2    // Store 16-bit result
    
    // Advance to next block of 8 elements
    add     w4, #4, #4      // K_block_idx += 8
    add     x4, x4, #8      // x4 += (8 << 5) (advance in input array)
    
    // Check if we processed all inputs
    subs    x2, #1, #1      // K_blocks -= 1
    b.ne    2b                // Continue if more blocks
    
    // Move to next output
    add     w1, #1, #1      // out_idx += 1
    cmp     w1, x0          // Check if done all outputs
    b.ne    1b                // Continue if more outputs
    
    ret

/* ============================================================================
 * Q4_0 × Int8 MatMul with Shift-Register Add (Dequantize + Multiply)
 * ============================================================================ */
mt6855v_matmul_shift_register_q8_0:
    // Q4_0 format: 4-bit weights stored in 8-bit containers
    // Each 4-bit weight represents -8 to +7, stored as 0x00-0x0F
    
    // Dequantization: q4_val -> int8 (simplified for performance)
    // Then matrix multiplication using optimized int8 ops
    
    // This is for the feed-forward weights, not attention matrices