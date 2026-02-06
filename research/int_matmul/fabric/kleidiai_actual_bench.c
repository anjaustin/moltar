/*
 * Benchmark that mimics the EXACT KleidiAI kernel inner loop
 * to understand if our optimization actually helps.
 */

#define _GNU_SOURCE
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

extern void* memalign(size_t alignment, size_t size);
#define aligned_alloc(align, size) memalign(align, size)

#define BL 32  // Block length
#define NR 4   // Process 4 output columns at a time

// Packed LHS block: [FP16 scale (2B)][32 x int8 (32B)]
typedef struct {
    uint16_t scale;
    int8_t qs[32];
} __attribute__((packed)) lhs_block_t;

// Packed RHS for 4 columns, 1 block: [4x FP16 scales (8B)][64B weights]
// Actually in KleidiAI it's [scales][weights interleaved]
typedef struct {
    uint16_t scales[4];
    uint8_t weights[64];  // 4 columns x 16 bytes
} __attribute__((packed)) rhs_block_4col_t;

// ============================================================================
// KleidiAI-style kernel (mirrors the assembly)
// ============================================================================
__attribute__((noinline))
void gemv_kleidiai_asm_style(
    size_t n,              // Number of output columns
    size_t num_blocks,     // k / 32
    const lhs_block_t* lhs,    // [num_blocks]
    const rhs_block_4col_t* rhs,  // [n/4 * num_blocks]
    float* dst
) {
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    
    for (size_t col4 = 0; col4 < n; col4 += NR) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        const lhs_block_t* lhs_ptr = lhs;
        const rhs_block_4col_t* rhs_ptr = &rhs[(col4 / NR) * num_blocks];
        
        for (size_t b = 0; b < num_blocks; b++) {
            // Load LHS scale and convert FP16->FP32
            float16x4_t lhs_scale_h = vld1_dup_f16((const __fp16*)&lhs_ptr->scale);
            float32x4_t lhs_scale = vcvt_f32_f16(lhs_scale_h);
            
            // Load RHS scales and convert FP16->FP32
            float16x4_t rhs_scales_h = vld1_f16((const __fp16*)rhs_ptr->scales);
            float32x4_t rhs_scales = vcvt_f32_f16(rhs_scales_h);
            
            // Combined scale
            float32x4_t combined = vmulq_f32(lhs_scale, rhs_scales);
            
            // Load LHS activations
            int8x16_t lhs_lo = vld1q_s8(lhs_ptr->qs);
            int8x16_t lhs_hi = vld1q_s8(lhs_ptr->qs + 16);
            
            // Process 4 columns
            int32_t dots_arr[4];
            
            for (int c = 0; c < 4; c++) {
                // Unpack Q4 weights for this column
                uint8x16_t w_packed = vld1q_u8(&rhs_ptr->weights[c * 16]);
                int8x16_t w_lo = vreinterpretq_s8_u8(vandq_u8(w_packed, mask_lo));
                int8x16_t w_hi = vreinterpretq_s8_u8(vshrq_n_u8(w_packed, 4));
                w_lo = vsubq_s8(w_lo, vdupq_n_s8(8));
                w_hi = vsubq_s8(w_hi, vdupq_n_s8(8));
                
                // Dot products
                int32x4_t d_lo = vdotq_s32(vdupq_n_s32(0), w_lo, lhs_lo);
                int32x4_t d_hi = vdotq_s32(vdupq_n_s32(0), w_hi, lhs_hi);
                dots_arr[c] = vaddvq_s32(vaddq_s32(d_lo, d_hi));
            }
            int32x4_t dots = vld1q_s32(dots_arr);
            
            // Convert and accumulate (like scvtf + fmla)
            float32x4_t fdots = vcvtq_f32_s32(dots);
            acc = vmlaq_f32(acc, fdots, combined);
            
            lhs_ptr++;
            rhs_ptr++;
        }
        
        vst1q_f32(&dst[col4], acc);
    }
}

// ============================================================================
// Our batched kernel with precomputed scales
// ============================================================================
__attribute__((noinline))
void gemv_batched_precomputed(
    size_t n,
    size_t num_blocks,
    const lhs_block_t* lhs,
    const rhs_block_4col_t* rhs,
    const float* lhs_scales_fp32,    // [num_blocks] precomputed
    const float* rhs_scales_fp32,    // [n * num_blocks] precomputed
    float* dst
) {
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    
    for (size_t col4 = 0; col4 < n; col4 += NR) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        
        const lhs_block_t* lhs_ptr = lhs;
        const rhs_block_4col_t* rhs_ptr = &rhs[(col4 / NR) * num_blocks];
        const float* rhs_scales_ptr = &rhs_scales_fp32[col4 * num_blocks];
        
        size_t b = 0;
        
        // Process 2 blocks at a time for better ILP
        for (; b + 1 < num_blocks; b += 2) {
            // Precomputed LHS scales
            float ls0 = lhs_scales_fp32[b];
            float ls1 = lhs_scales_fp32[b + 1];
            
            // Load LHS activations for both blocks
            int8x16_t lhs_lo0 = vld1q_s8(lhs_ptr[0].qs);
            int8x16_t lhs_hi0 = vld1q_s8(lhs_ptr[0].qs + 16);
            int8x16_t lhs_lo1 = vld1q_s8(lhs_ptr[1].qs);
            int8x16_t lhs_hi1 = vld1q_s8(lhs_ptr[1].qs + 16);
            
            // Compute dots for all 4 columns, both blocks
            int32_t dots0_arr[4], dots1_arr[4];
            
            for (int c = 0; c < 4; c++) {
                // Block 0
                uint8x16_t w0 = vld1q_u8(&rhs_ptr[0].weights[c * 16]);
                int8x16_t w0_lo = vreinterpretq_s8_u8(vandq_u8(w0, mask_lo));
                int8x16_t w0_hi = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                w0_lo = vsubq_s8(w0_lo, vdupq_n_s8(8));
                w0_hi = vsubq_s8(w0_hi, vdupq_n_s8(8));
                int32x4_t d0 = vaddq_s32(
                    vdotq_s32(vdupq_n_s32(0), w0_lo, lhs_lo0),
                    vdotq_s32(vdupq_n_s32(0), w0_hi, lhs_hi0)
                );
                dots0_arr[c] = vaddvq_s32(d0);
                
                // Block 1
                uint8x16_t w1 = vld1q_u8(&rhs_ptr[1].weights[c * 16]);
                int8x16_t w1_lo = vreinterpretq_s8_u8(vandq_u8(w1, mask_lo));
                int8x16_t w1_hi = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
                w1_lo = vsubq_s8(w1_lo, vdupq_n_s8(8));
                w1_hi = vsubq_s8(w1_hi, vdupq_n_s8(8));
                int32x4_t d1 = vaddq_s32(
                    vdotq_s32(vdupq_n_s32(0), w1_lo, lhs_lo1),
                    vdotq_s32(vdupq_n_s32(0), w1_hi, lhs_hi1)
                );
                dots1_arr[c] = vaddvq_s32(d1);
            }
            
            // Batched convert (2x vcvtq_f32_s32)
            float32x4_t fdots0 = vcvtq_f32_s32(vld1q_s32(dots0_arr));
            float32x4_t fdots1 = vcvtq_f32_s32(vld1q_s32(dots1_arr));
            
            // Load precomputed RHS scales and compute combined
            float32x4_t rs0 = {
                rhs_scales_ptr[0 * num_blocks + b],
                rhs_scales_ptr[1 * num_blocks + b],
                rhs_scales_ptr[2 * num_blocks + b],
                rhs_scales_ptr[3 * num_blocks + b]
            };
            float32x4_t rs1 = {
                rhs_scales_ptr[0 * num_blocks + b + 1],
                rhs_scales_ptr[1 * num_blocks + b + 1],
                rhs_scales_ptr[2 * num_blocks + b + 1],
                rhs_scales_ptr[3 * num_blocks + b + 1]
            };
            
            float32x4_t combined0 = vmulq_n_f32(rs0, ls0);
            float32x4_t combined1 = vmulq_n_f32(rs1, ls1);
            
            acc0 = vmlaq_f32(acc0, fdots0, combined0);
            acc1 = vmlaq_f32(acc1, fdots1, combined1);
            
            lhs_ptr += 2;
            rhs_ptr += 2;
        }
        
        // Handle remaining block
        for (; b < num_blocks; b++) {
            float ls = lhs_scales_fp32[b];
            
            int8x16_t lhs_lo = vld1q_s8(lhs_ptr->qs);
            int8x16_t lhs_hi = vld1q_s8(lhs_ptr->qs + 16);
            
            int32_t dots_arr[4];
            for (int c = 0; c < 4; c++) {
                uint8x16_t w = vld1q_u8(&rhs_ptr->weights[c * 16]);
                int8x16_t w_lo = vreinterpretq_s8_u8(vandq_u8(w, mask_lo));
                int8x16_t w_hi = vreinterpretq_s8_u8(vshrq_n_u8(w, 4));
                w_lo = vsubq_s8(w_lo, vdupq_n_s8(8));
                w_hi = vsubq_s8(w_hi, vdupq_n_s8(8));
                int32x4_t d = vaddq_s32(
                    vdotq_s32(vdupq_n_s32(0), w_lo, lhs_lo),
                    vdotq_s32(vdupq_n_s32(0), w_hi, lhs_hi)
                );
                dots_arr[c] = vaddvq_s32(d);
            }
            
            float32x4_t fdots = vcvtq_f32_s32(vld1q_s32(dots_arr));
            float32x4_t rs = {
                rhs_scales_ptr[0 * num_blocks + b],
                rhs_scales_ptr[1 * num_blocks + b],
                rhs_scales_ptr[2 * num_blocks + b],
                rhs_scales_ptr[3 * num_blocks + b]
            };
            float32x4_t combined = vmulq_n_f32(rs, ls);
            acc0 = vmlaq_f32(acc0, fdots, combined);
            
            lhs_ptr++;
            rhs_ptr++;
        }
        
        vst1q_f32(&dst[col4], vaddq_f32(acc0, acc1));
    }
}

// Helper to convert FP16 bits to FP32
static inline float fp16_to_fp32(uint16_t h) {
    __fp16 hf;
    memcpy(&hf, &h, 2);
    return (float)hf;
}

int main(int argc, char** argv) {
    int n_out = 1024;
    int n_in = 4608;
    int iters = 1000;
    
    if (argc > 1) n_in = atoi(argv[1]);
    if (argc > 2) n_out = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);
    
    // Round to NR
    n_out = (n_out + NR - 1) / NR * NR;
    size_t num_blocks = n_in / BL;
    
    printf("KleidiAI Actual Kernel Comparison\n");
    printf("==================================\n");
    printf("Output: %d, Input: %d (%zu blocks), %d iterations\n\n", n_out, n_in, num_blocks, iters);
    
    // Allocate
    lhs_block_t* lhs = aligned_alloc(64, num_blocks * sizeof(lhs_block_t));
    rhs_block_4col_t* rhs = aligned_alloc(64, (n_out / NR) * num_blocks * sizeof(rhs_block_4col_t));
    float* lhs_scales = aligned_alloc(64, num_blocks * sizeof(float));
    float* rhs_scales = aligned_alloc(64, n_out * num_blocks * sizeof(float));
    float* dst1 = aligned_alloc(64, n_out * sizeof(float));
    float* dst2 = aligned_alloc(64, n_out * sizeof(float));
    
    // Initialize
    srand(42);
    for (size_t b = 0; b < num_blocks; b++) {
        lhs[b].scale = 0x3000 + (rand() % 0x400);
        lhs_scales[b] = fp16_to_fp32(lhs[b].scale);
        for (int i = 0; i < 32; i++) {
            lhs[b].qs[i] = (rand() % 256) - 128;
        }
    }
    
    for (size_t g = 0; g < n_out / NR; g++) {
        for (size_t b = 0; b < num_blocks; b++) {
            rhs_block_4col_t* blk = &rhs[g * num_blocks + b];
            for (int c = 0; c < 4; c++) {
                blk->scales[c] = 0x3000 + (rand() % 0x400);
                rhs_scales[(g * 4 + c) * num_blocks + b] = fp16_to_fp32(blk->scales[c]);
                for (int i = 0; i < 16; i++) {
                    blk->weights[c * 16 + i] = rand() & 0xFF;
                }
            }
        }
    }
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        gemv_kleidiai_asm_style(n_out, num_blocks, lhs, rhs, dst1);
        gemv_batched_precomputed(n_out, num_blocks, lhs, rhs, lhs_scales, rhs_scales, dst2);
    }
    
    // Validate
    gemv_kleidiai_asm_style(n_out, num_blocks, lhs, rhs, dst1);
    gemv_batched_precomputed(n_out, num_blocks, lhs, rhs, lhs_scales, rhs_scales, dst2);
    
    float max_err = 0;
    for (int i = 0; i < n_out; i++) {
        float err = fabsf(dst1[i] - dst2[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max abs error: %.2e\n\n", max_err);
    
    // Benchmark
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        gemv_kleidiai_asm_style(n_out, num_blocks, lhs, rhs, dst1);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double kleidiai_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        gemv_batched_precomputed(n_out, num_blocks, lhs, rhs, lhs_scales, rhs_scales, dst2);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double batched_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    
    printf("Results:\n");
    printf("  KleidiAI-style:     %8.1f us (1.00x)\n", kleidiai_us);
    printf("  Batched precomp:    %8.1f us (%.2fx)\n", batched_us, kleidiai_us / batched_us);
    
    free(lhs);
    free(rhs);
    free(lhs_scales);
    free(rhs_scales);
    free(dst1);
    free(dst2);
    
    return 0;
}
