/*
 * Compare our batched approach vs KleidiAI-like inner loop
 * 
 * Goal: Understand what optimization actually helps in KleidiAI's context.
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

#define BLOCK_SIZE 32
#define KAI_NR 4  // KleidiAI processes 4 output columns at once

// Packed LHS block (matches KleidiAI qsi8d32p format)
// [FP16 scale (2B)] [32 x int8 values]
typedef struct {
    uint16_t scale;
    int8_t qs[32];
} lhs_block_t;

// Packed RHS block for 4 columns (matches KleidiAI qsi4c32p4x8 format)
// [4x FP16 scales (8B)] [4 rows x 16B weights = 64B]
typedef struct {
    uint16_t scales[4];  // One scale per column
    uint8_t qs[4][16];   // 4 columns, each with 16 bytes (32 x 4-bit)
} rhs_block_t;

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp += 127 - 15;
    }
    
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// KleidiAI-style inner loop (one block at a time, with FP16->FP32 conversion)
__attribute__((noinline))
void gemv_kleidiai_style(
    int n_out,      // Must be multiple of 4
    int n_in,
    const lhs_block_t* lhs,    // [nb] blocks
    const rhs_block_t* rhs,    // [n_out/4 * nb] blocks
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int col4 = 0; col4 < n_out; col4 += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        for (int b = 0; b < nb; b++) {
            const lhs_block_t* lhs_blk = &lhs[b];
            const rhs_block_t* rhs_blk = &rhs[(col4/4) * nb + b];
            
            // Convert LHS scale FP16 -> FP32
            float lhs_scale = fp16_to_fp32(lhs_blk->scale);
            
            // Convert RHS scales FP16 -> FP32
            float32x4_t rhs_scales = {
                fp16_to_fp32(rhs_blk->scales[0]),
                fp16_to_fp32(rhs_blk->scales[1]),
                fp16_to_fp32(rhs_blk->scales[2]),
                fp16_to_fp32(rhs_blk->scales[3])
            };
            
            // Combined scale = lhs_scale * rhs_scales
            float32x4_t combined = vmulq_n_f32(rhs_scales, lhs_scale);
            
            // Compute 4 dot products (one per output column)
            int32_t dots_arr[4];
            
            for (int c = 0; c < 4; c++) {
                int32_t dot = 0;
                for (int i = 0; i < 16; i++) {
                    int8_t lo = (rhs_blk->qs[c][i] & 0x0F) - 8;
                    int8_t hi = (rhs_blk->qs[c][i] >> 4) - 8;
                    dot += lo * lhs_blk->qs[i];
                    dot += hi * lhs_blk->qs[16 + i];
                }
                dots_arr[c] = dot;
            }
            int32x4_t dots = vld1q_s32(dots_arr);
            
            // Convert to float and accumulate
            float32x4_t fdots = vcvtq_f32_s32(dots);
            acc = vmlaq_f32(acc, fdots, combined);
        }
        
        vst1q_f32(&output[col4], acc);
    }
}

// KleidiAI-style with NEON dot product (more realistic)
__attribute__((noinline))
void gemv_kleidiai_neon(
    int n_out,
    int n_in,
    const lhs_block_t* lhs,
    const rhs_block_t* rhs,
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int col4 = 0; col4 < n_out; col4 += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        for (int b = 0; b < nb; b++) {
            const lhs_block_t* lhs_blk = &lhs[b];
            const rhs_block_t* rhs_blk = &rhs[(col4/4) * nb + b];
            
            // Load and convert scales
            float lhs_scale = fp16_to_fp32(lhs_blk->scale);
            float32x4_t rhs_scales = {
                fp16_to_fp32(rhs_blk->scales[0]),
                fp16_to_fp32(rhs_blk->scales[1]),
                fp16_to_fp32(rhs_blk->scales[2]),
                fp16_to_fp32(rhs_blk->scales[3])
            };
            float32x4_t combined = vmulq_n_f32(rhs_scales, lhs_scale);
            
            // Load LHS data
            int8x16_t lhs_lo = vld1q_s8(lhs_blk->qs);
            int8x16_t lhs_hi = vld1q_s8(lhs_blk->qs + 16);
            
            // Process 4 columns
            int32_t dots_arr2[4];
            
            for (int c = 0; c < 4; c++) {
                // Unpack Q4 weights
                uint8x16_t packed = vld1q_u8(rhs_blk->qs[c]);
                int8x16_t rhs_lo = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F)));
                int8x16_t rhs_hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
                rhs_lo = vsubq_s8(rhs_lo, vdupq_n_s8(8));
                rhs_hi = vsubq_s8(rhs_hi, vdupq_n_s8(8));
                
                // Dot products
                int32x4_t d_lo = vdotq_s32(vdupq_n_s32(0), rhs_lo, lhs_lo);
                int32x4_t d_hi = vdotq_s32(vdupq_n_s32(0), rhs_hi, lhs_hi);
                dots_arr2[c] = vaddvq_s32(vaddq_s32(d_lo, d_hi));
            }
            int32x4_t dots = vld1q_s32(dots_arr2);
            
            float32x4_t fdots = vcvtq_f32_s32(dots);
            acc = vmlaq_f32(acc, fdots, combined);
        }
        
        vst1q_f32(&output[col4], acc);
    }
}

// Batched-8 variant with precomputed combined scales
__attribute__((noinline))
void gemv_batched8_precomputed(
    int n_out,
    int n_in,
    const lhs_block_t* lhs,
    const rhs_block_t* rhs,
    const float* combined_scales,  // [n_out * nb] precomputed
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int col4 = 0; col4 < n_out; col4 += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        int b = 0;
        // Process 2 blocks at a time (8 scale loads -> 2 vcvtq)
        // Actually, scales are already FP32 so we don't need vcvtq for scales
        for (; b + 1 < nb; b += 2) {
            // Block 0
            const lhs_block_t* lhs_blk0 = &lhs[b];
            const rhs_block_t* rhs_blk0 = &rhs[(col4/4) * nb + b];
            const float* scales0 = &combined_scales[col4 * nb + b * 4];
            
            // Block 1  
            const lhs_block_t* lhs_blk1 = &lhs[b+1];
            const rhs_block_t* rhs_blk1 = &rhs[(col4/4) * nb + b + 1];
            const float* scales1 = &combined_scales[col4 * nb + (b+1) * 4];
            
            int8x16_t lhs_lo0 = vld1q_s8(lhs_blk0->qs);
            int8x16_t lhs_hi0 = vld1q_s8(lhs_blk0->qs + 16);
            int8x16_t lhs_lo1 = vld1q_s8(lhs_blk1->qs);
            int8x16_t lhs_hi1 = vld1q_s8(lhs_blk1->qs + 16);
            
            // Compute 8 dot products total (4 per block)
            int32_t dots0[4], dots1[4];
            
            for (int c = 0; c < 4; c++) {
                uint8x16_t packed0 = vld1q_u8(rhs_blk0->qs[c]);
                int8x16_t rhs_lo = vreinterpretq_s8_u8(vandq_u8(packed0, vdupq_n_u8(0x0F)));
                int8x16_t rhs_hi = vreinterpretq_s8_u8(vshrq_n_u8(packed0, 4));
                rhs_lo = vsubq_s8(rhs_lo, vdupq_n_s8(8));
                rhs_hi = vsubq_s8(rhs_hi, vdupq_n_s8(8));
                int32x4_t d_lo = vdotq_s32(vdupq_n_s32(0), rhs_lo, lhs_lo0);
                int32x4_t d_hi = vdotq_s32(vdupq_n_s32(0), rhs_hi, lhs_hi0);
                dots0[c] = vaddvq_s32(vaddq_s32(d_lo, d_hi));
                
                uint8x16_t packed1 = vld1q_u8(rhs_blk1->qs[c]);
                rhs_lo = vreinterpretq_s8_u8(vandq_u8(packed1, vdupq_n_u8(0x0F)));
                rhs_hi = vreinterpretq_s8_u8(vshrq_n_u8(packed1, 4));
                rhs_lo = vsubq_s8(rhs_lo, vdupq_n_s8(8));
                rhs_hi = vsubq_s8(rhs_hi, vdupq_n_s8(8));
                d_lo = vdotq_s32(vdupq_n_s32(0), rhs_lo, lhs_lo1);
                d_hi = vdotq_s32(vdupq_n_s32(0), rhs_hi, lhs_hi1);
                dots1[c] = vaddvq_s32(vaddq_s32(d_lo, d_hi));
            }
            
            // Batched conversion and accumulate
            float32x4_t fdots0 = vcvtq_f32_s32(vld1q_s32(dots0));
            float32x4_t fdots1 = vcvtq_f32_s32(vld1q_s32(dots1));
            float32x4_t s0 = vld1q_f32(scales0);
            float32x4_t s1 = vld1q_f32(scales1);
            acc = vmlaq_f32(acc, fdots0, s0);
            acc = vmlaq_f32(acc, fdots1, s1);
        }
        
        // Handle remaining block
        for (; b < nb; b++) {
            const lhs_block_t* lhs_blk = &lhs[b];
            const rhs_block_t* rhs_blk = &rhs[(col4/4) * nb + b];
            const float* scales = &combined_scales[col4 * nb + b * 4];
            
            int8x16_t lhs_lo = vld1q_s8(lhs_blk->qs);
            int8x16_t lhs_hi = vld1q_s8(lhs_blk->qs + 16);
            
            int32_t dots[4];
            for (int c = 0; c < 4; c++) {
                uint8x16_t packed = vld1q_u8(rhs_blk->qs[c]);
                int8x16_t rhs_lo = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F)));
                int8x16_t rhs_hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
                rhs_lo = vsubq_s8(rhs_lo, vdupq_n_s8(8));
                rhs_hi = vsubq_s8(rhs_hi, vdupq_n_s8(8));
                int32x4_t d_lo = vdotq_s32(vdupq_n_s32(0), rhs_lo, lhs_lo);
                int32x4_t d_hi = vdotq_s32(vdupq_n_s32(0), rhs_hi, lhs_hi);
                dots[c] = vaddvq_s32(vaddq_s32(d_lo, d_hi));
            }
            
            float32x4_t fdots = vcvtq_f32_s32(vld1q_s32(dots));
            float32x4_t s = vld1q_f32(scales);
            acc = vmlaq_f32(acc, fdots, s);
        }
        
        vst1q_f32(&output[col4], acc);
    }
}

int main(int argc, char** argv) {
    int n_in = 4608;
    int n_out = 1024;
    int iters = 1000;
    
    if (argc > 1) n_in = atoi(argv[1]);
    if (argc > 2) n_out = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);
    
    // Ensure n_out is multiple of 4
    n_out = (n_out + 3) & ~3;
    int nb = n_in / BLOCK_SIZE;
    
    printf("KleidiAI-style Comparison\n");
    printf("=========================\n");
    printf("Output: %d, Input: %d (%d blocks), %d iterations\n\n", n_out, n_in, nb, iters);
    
    // Allocate
    lhs_block_t* lhs = aligned_alloc(64, nb * sizeof(lhs_block_t));
    rhs_block_t* rhs = aligned_alloc(64, (n_out/4) * nb * sizeof(rhs_block_t));
    float* combined_scales = aligned_alloc(64, n_out * nb * sizeof(float));
    float* output = aligned_alloc(64, n_out * sizeof(float));
    
    // Initialize
    srand(42);
    for (int b = 0; b < nb; b++) {
        lhs[b].scale = 0x3000 + (rand() % 0x400);
        for (int i = 0; i < 32; i++) {
            lhs[b].qs[i] = (rand() % 256) - 128;
        }
    }
    for (int g = 0; g < n_out/4; g++) {
        for (int b = 0; b < nb; b++) {
            rhs_block_t* blk = &rhs[g * nb + b];
            for (int c = 0; c < 4; c++) {
                blk->scales[c] = 0x3000 + (rand() % 0x400);
                for (int i = 0; i < 16; i++) {
                    blk->qs[c][i] = rand() & 0xFF;
                }
            }
        }
    }
    
    // Precompute combined scales
    for (int col = 0; col < n_out; col++) {
        int g = col / 4;
        int c = col % 4;
        for (int b = 0; b < nb; b++) {
            float lhs_scale = fp16_to_fp32(lhs[b].scale);
            float rhs_scale = fp16_to_fp32(rhs[g * nb + b].scales[c]);
            combined_scales[col * nb + b] = lhs_scale * rhs_scale;
        }
    }
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        gemv_kleidiai_neon(n_out, n_in, lhs, rhs, output);
    }
    
    struct timespec start, end;
    
    // KleidiAI-style NEON
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        gemv_kleidiai_neon(n_out, n_in, lhs, rhs, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double kleidiai_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    
    // Batched with precomputed scales
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        gemv_batched8_precomputed(n_out, n_in, lhs, rhs, combined_scales, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double batched_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    
    printf("Results:\n");
    printf("  KleidiAI-style:     %8.1f us (1.00x)\n", kleidiai_us);
    printf("  Batched (precomp):  %8.1f us (%.2fx)\n", batched_us, kleidiai_us / batched_us);
    
    printf("\nMemory overhead:\n");
    size_t rhs_bytes = (n_out/4) * nb * sizeof(rhs_block_t);
    size_t scale_bytes = n_out * nb * sizeof(float);
    printf("  RHS packed: %.2f MB\n", rhs_bytes / (1024.0 * 1024.0));
    printf("  Combined scales: %.2f MB (+%.1f%% of RHS)\n",
           scale_bytes / (1024.0 * 1024.0),
           100.0 * scale_bytes / rhs_bytes);
    
    free(lhs);
    free(rhs);
    free(combined_scales);
    free(output);
    
    return 0;
}
