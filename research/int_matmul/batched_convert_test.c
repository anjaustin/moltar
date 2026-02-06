/*
 * Batched SCVTF Test - Can we hide conversion latency?
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

typedef struct {
    uint16_t d;
    uint8_t qs[16];
} block_q4_0;

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

// Helper: compute single block dot product
static inline int32_t compute_block_dot(
    const block_q4_0* blk,
    const int8_t* act
) {
    uint8x16_t q4_packed = vld1q_u8(blk->qs);
    int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
    int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
    q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
    q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
    
    int8x16_t act_lo = vld1q_s8(act);
    int8x16_t act_hi = vld1q_s8(act + 16);
    
    int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
    int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
    return vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
}

// ============================================================================
// VERSION 1: Scalar conversion per block
// ============================================================================
__attribute__((noinline))
void matvec_scalar(int n_out, int n_in, const block_q4_0* weights,
                   const int8_t* activations, const float* act_scales, float* output) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        float sum = 0.0f;
        
        for (int b = 0; b < nb; b++) {
            int32_t dot = compute_block_dot(&row_weights[b], activations + b * 32);
            float w_scale = fp16_to_fp32(row_weights[b].d);
            sum += (float)dot * w_scale * act_scales[b];
        }
        output[row] = sum;
    }
}

// ============================================================================
// VERSION 2: Batched 4 blocks with precomputed scales
// ============================================================================
__attribute__((noinline))
void matvec_batched4(int n_out, int n_in, const block_q4_0* weights,
                     const int8_t* activations, const float* combined_scales, float* output) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        const float* row_scales = combined_scales + row * nb;
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        int b = 0;
        for (; b + 3 < nb; b += 4) {
            // Compute 4 dot products
            int32_t d0 = compute_block_dot(&row_weights[b], activations + b * 32);
            int32_t d1 = compute_block_dot(&row_weights[b+1], activations + (b+1) * 32);
            int32_t d2 = compute_block_dot(&row_weights[b+2], activations + (b+2) * 32);
            int32_t d3 = compute_block_dot(&row_weights[b+3], activations + (b+3) * 32);
            
            int32_t dot_arr[4] = {d0, d1, d2, d3};
            int32x4_t dots = vld1q_s32(dot_arr);
            
            // Batched convert + scale
            float32x4_t fdots = vcvtq_f32_s32(dots);
            float32x4_t scales = vld1q_f32(row_scales + b);
            acc = vmlaq_f32(acc, fdots, scales);
        }
        
        float sum = vaddvq_f32(acc);
        
        for (; b < nb; b++) {
            int32_t dot = compute_block_dot(&row_weights[b], activations + b * 32);
            sum += (float)dot * row_scales[b];
        }
        output[row] = sum;
    }
}

// ============================================================================
// VERSION 3: Batched 8 blocks (more parallelism)
// ============================================================================
__attribute__((noinline))
void matvec_batched8(int n_out, int n_in, const block_q4_0* weights,
                     const int8_t* activations, const float* combined_scales, float* output) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        const float* row_scales = combined_scales + row * nb;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        
        int b = 0;
        for (; b + 7 < nb; b += 8) {
            // First 4
            int32_t d0 = compute_block_dot(&row_weights[b], activations + b * 32);
            int32_t d1 = compute_block_dot(&row_weights[b+1], activations + (b+1) * 32);
            int32_t d2 = compute_block_dot(&row_weights[b+2], activations + (b+2) * 32);
            int32_t d3 = compute_block_dot(&row_weights[b+3], activations + (b+3) * 32);
            
            // Second 4
            int32_t d4 = compute_block_dot(&row_weights[b+4], activations + (b+4) * 32);
            int32_t d5 = compute_block_dot(&row_weights[b+5], activations + (b+5) * 32);
            int32_t d6 = compute_block_dot(&row_weights[b+6], activations + (b+6) * 32);
            int32_t d7 = compute_block_dot(&row_weights[b+7], activations + (b+7) * 32);
            
            int32_t arr0[4] = {d0, d1, d2, d3};
            int32_t arr1[4] = {d4, d5, d6, d7};
            
            float32x4_t fdots0 = vcvtq_f32_s32(vld1q_s32(arr0));
            float32x4_t fdots1 = vcvtq_f32_s32(vld1q_s32(arr1));
            float32x4_t scales0 = vld1q_f32(row_scales + b);
            float32x4_t scales1 = vld1q_f32(row_scales + b + 4);
            
            acc0 = vmlaq_f32(acc0, fdots0, scales0);
            acc1 = vmlaq_f32(acc1, fdots1, scales1);
        }
        
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        
        // Handle remaining 4-block batch
        for (; b + 3 < nb; b += 4) {
            int32_t d0 = compute_block_dot(&row_weights[b], activations + b * 32);
            int32_t d1 = compute_block_dot(&row_weights[b+1], activations + (b+1) * 32);
            int32_t d2 = compute_block_dot(&row_weights[b+2], activations + (b+2) * 32);
            int32_t d3 = compute_block_dot(&row_weights[b+3], activations + (b+3) * 32);
            
            int32_t arr[4] = {d0, d1, d2, d3};
            float32x4_t fdots = vcvtq_f32_s32(vld1q_s32(arr));
            float32x4_t scales = vld1q_f32(row_scales + b);
            sum += vaddvq_f32(vmulq_f32(fdots, scales));
        }
        
        for (; b < nb; b++) {
            int32_t dot = compute_block_dot(&row_weights[b], activations + b * 32);
            sum += (float)dot * row_scales[b];
        }
        output[row] = sum;
    }
}

// ============================================================================
// VERSION 4: No precomputed scales (compute w_scale*a_scale inline)
// ============================================================================
__attribute__((noinline))
void matvec_batched4_inline_scales(int n_out, int n_in, const block_q4_0* weights,
                                    const int8_t* activations, const float* act_scales, float* output) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        int b = 0;
        for (; b + 3 < nb; b += 4) {
            int32_t d0 = compute_block_dot(&row_weights[b], activations + b * 32);
            int32_t d1 = compute_block_dot(&row_weights[b+1], activations + (b+1) * 32);
            int32_t d2 = compute_block_dot(&row_weights[b+2], activations + (b+2) * 32);
            int32_t d3 = compute_block_dot(&row_weights[b+3], activations + (b+3) * 32);
            
            int32_t dot_arr[4] = {d0, d1, d2, d3};
            int32x4_t dots = vld1q_s32(dot_arr);
            
            // Compute scales inline
            float s0 = fp16_to_fp32(row_weights[b].d) * act_scales[b];
            float s1 = fp16_to_fp32(row_weights[b+1].d) * act_scales[b+1];
            float s2 = fp16_to_fp32(row_weights[b+2].d) * act_scales[b+2];
            float s3 = fp16_to_fp32(row_weights[b+3].d) * act_scales[b+3];
            float scale_arr[4] = {s0, s1, s2, s3};
            float32x4_t scales = vld1q_f32(scale_arr);
            
            float32x4_t fdots = vcvtq_f32_s32(dots);
            acc = vmlaq_f32(acc, fdots, scales);
        }
        
        float sum = vaddvq_f32(acc);
        
        for (; b < nb; b++) {
            int32_t dot = compute_block_dot(&row_weights[b], activations + b * 32);
            float w_scale = fp16_to_fp32(row_weights[b].d);
            sum += (float)dot * w_scale * act_scales[b];
        }
        output[row] = sum;
    }
}

int main(int argc, char** argv) {
    int n_in = 4608;
    int n_out = 1024;
    int iters = 1000;
    
    if (argc > 1) n_in = atoi(argv[1]);
    if (argc > 2) n_out = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);
    
    int nb = n_in / BLOCK_SIZE;
    
    printf("Batched Convert Test\n");
    printf("====================\n");
    printf("Matrix: %d x %d (%d blocks/row), %d iterations\n\n", n_out, n_in, nb, iters);
    
    block_q4_0* weights = aligned_alloc(64, n_out * nb * sizeof(block_q4_0));
    int8_t* activations = aligned_alloc(64, n_in);
    float* act_scales = aligned_alloc(64, nb * sizeof(float));
    float* combined_scales = aligned_alloc(64, n_out * nb * sizeof(float));
    float* output = aligned_alloc(64, n_out * sizeof(float));
    
    srand(42);
    for (int i = 0; i < n_out * nb; i++) {
        weights[i].d = 0x3000 + (rand() % 0x400);
        for (int j = 0; j < 16; j++) weights[i].qs[j] = rand() & 0xFF;
    }
    for (int b = 0; b < nb; b++) {
        act_scales[b] = 0.01f + (rand() % 50) * 0.001f;
    }
    for (int i = 0; i < n_in; i++) {
        activations[i] = (rand() % 256) - 128;
    }
    for (int row = 0; row < n_out; row++) {
        for (int b = 0; b < nb; b++) {
            combined_scales[row * nb + b] = fp16_to_fp32(weights[row * nb + b].d) * act_scales[b];
        }
    }
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        matvec_scalar(n_out, n_in, weights, activations, act_scales, output);
    }
    
    struct timespec start, end;
    
    // Scalar
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_scalar(n_out, n_in, weights, activations, act_scales, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double scalar_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    printf("Scalar (baseline):      %8.1f us\n", scalar_us);
    
    // Batched 4 with precomputed
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_batched4(n_out, n_in, weights, activations, combined_scales, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double batched4_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    printf("Batched4 (precomp):     %8.1f us (%.2fx)\n", batched4_us, scalar_us / batched4_us);
    
    // Batched 8 with precomputed
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_batched8(n_out, n_in, weights, activations, combined_scales, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double batched8_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    printf("Batched8 (precomp):     %8.1f us (%.2fx)\n", batched8_us, scalar_us / batched8_us);
    
    // Batched 4 inline scales (no precompute)
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_batched4_inline_scales(n_out, n_in, weights, activations, act_scales, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double inline_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    printf("Batched4 (inline):      %8.1f us (%.2fx)\n", inline_us, scalar_us / inline_us);
    
    printf("\nMemory overhead:\n");
    printf("  Precomputed scales: %.2f MB (+%.1f%% of weights)\n",
           n_out * nb * sizeof(float) / (1024.0 * 1024.0),
           100.0 * sizeof(float) / sizeof(block_q4_0));
    
    free(weights);
    free(activations);
    free(act_scales);
    free(combined_scales);
    free(output);
    
    return 0;
}
