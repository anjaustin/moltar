/*
 * Batched GEMV Implementation for Q4_0 weights
 * 
 * Key optimization: Batch 8 blocks before int32->float32 conversion
 * to hide SCVTF latency through instruction parallelism.
 */

#define _GNU_SOURCE
#include "batched_gemv.h"

#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

// Android doesn't have aligned_alloc, use memalign
extern void* memalign(size_t alignment, size_t size);
#define aligned_alloc(align, size) memalign(align, size)

// FP16 -> FP32 conversion
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

// Single block dot product using SDOT
static inline int32_t compute_block_dot(
    const batched_q4_0_block* blk,
    const int8_t* act
) {
    // Load and unpack Q4_0 weights
    uint8x16_t q4_packed = vld1q_u8(blk->qs);
    int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
    int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
    
    // Subtract zero point (8 for unsigned Q4)
    q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
    q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
    
    // Load activations
    int8x16_t act_lo = vld1q_s8(act);
    int8x16_t act_hi = vld1q_s8(act + 16);
    
    // Compute dot products
    int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
    int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
    
    return vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
}

int batched_gemv_init_scales(
    batched_gemv_scales_t* scales,
    const batched_q4_0_block* weights,
    size_t n_out,
    size_t n_in
) {
    size_t nb = n_in / BATCHED_GEMV_BLOCK_SIZE;
    
    scales->scales = (float*)aligned_alloc(64, n_out * nb * sizeof(float));
    if (!scales->scales) return -1;
    
    scales->n_out = n_out;
    scales->n_in = n_in;
    scales->nb = nb;
    
    // Extract and convert weight scales from FP16 to FP32
    for (size_t row = 0; row < n_out; row++) {
        for (size_t b = 0; b < nb; b++) {
            scales->scales[row * nb + b] = fp16_to_fp32(weights[row * nb + b].d);
        }
    }
    
    return 0;
}

void batched_gemv_free_scales(batched_gemv_scales_t* scales) {
    if (scales->scales) {
        free(scales->scales);
        scales->scales = NULL;
    }
}

void batched_gemv_combine_scales(
    const batched_gemv_scales_t* weight_scales,
    const float* act_scales,
    float* combined_out
) {
    size_t n_out = weight_scales->n_out;
    size_t nb = weight_scales->nb;
    
    for (size_t row = 0; row < n_out; row++) {
        const float* ws = weight_scales->scales + row * nb;
        float* out = combined_out + row * nb;
        
        // Vectorized multiply
        size_t b = 0;
        for (; b + 3 < nb; b += 4) {
            float32x4_t w = vld1q_f32(ws + b);
            float32x4_t a = vld1q_f32(act_scales + b);
            vst1q_f32(out + b, vmulq_f32(w, a));
        }
        for (; b < nb; b++) {
            out[b] = ws[b] * act_scales[b];
        }
    }
}

// Main batched-8 GEMV kernel
void batched_gemv_q4_0(
    size_t n_out,
    size_t n_in,
    const batched_q4_0_block* weights,
    const int8_t* activations,
    const float* combined_scales,
    float* output
) {
    size_t nb = n_in / BATCHED_GEMV_BLOCK_SIZE;
    
    for (size_t row = 0; row < n_out; row++) {
        const batched_q4_0_block* row_weights = weights + row * nb;
        const float* row_scales = combined_scales + row * nb;
        
        // Dual accumulators for better ILP
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        
        size_t b = 0;
        
        // Process 8 blocks at a time for maximum SCVTF efficiency
        for (; b + 7 < nb; b += 8) {
            // First 4 blocks
            int32_t d0 = compute_block_dot(&row_weights[b], activations + b * 32);
            int32_t d1 = compute_block_dot(&row_weights[b+1], activations + (b+1) * 32);
            int32_t d2 = compute_block_dot(&row_weights[b+2], activations + (b+2) * 32);
            int32_t d3 = compute_block_dot(&row_weights[b+3], activations + (b+3) * 32);
            
            // Second 4 blocks
            int32_t d4 = compute_block_dot(&row_weights[b+4], activations + (b+4) * 32);
            int32_t d5 = compute_block_dot(&row_weights[b+5], activations + (b+5) * 32);
            int32_t d6 = compute_block_dot(&row_weights[b+6], activations + (b+6) * 32);
            int32_t d7 = compute_block_dot(&row_weights[b+7], activations + (b+7) * 32);
            
            // Pack into vectors
            int32_t arr0[4] = {d0, d1, d2, d3};
            int32_t arr1[4] = {d4, d5, d6, d7};
            
            // Batched convert: 2x vcvtq_f32_s32 (hides SCVTF latency)
            float32x4_t fdots0 = vcvtq_f32_s32(vld1q_s32(arr0));
            float32x4_t fdots1 = vcvtq_f32_s32(vld1q_s32(arr1));
            
            // Load precomputed combined scales
            float32x4_t scales0 = vld1q_f32(row_scales + b);
            float32x4_t scales1 = vld1q_f32(row_scales + b + 4);
            
            // Vectorized multiply-accumulate
            acc0 = vmlaq_f32(acc0, fdots0, scales0);
            acc1 = vmlaq_f32(acc1, fdots1, scales1);
        }
        
        // Reduce accumulators
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
        
        // Handle remaining blocks
        for (; b < nb; b++) {
            int32_t dot = compute_block_dot(&row_weights[b], activations + b * 32);
            sum += (float)dot * row_scales[b];
        }
        
        output[row] = sum;
    }
}

// Inline scales version (no precomputation)
void batched_gemv_q4_0_inline(
    size_t n_out,
    size_t n_in,
    const batched_q4_0_block* weights,
    const int8_t* activations,
    const float* act_scales,
    float* output
) {
    size_t nb = n_in / BATCHED_GEMV_BLOCK_SIZE;
    
    for (size_t row = 0; row < n_out; row++) {
        const batched_q4_0_block* row_weights = weights + row * nb;
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        size_t b = 0;
        for (; b + 3 < nb; b += 4) {
            int32_t d0 = compute_block_dot(&row_weights[b], activations + b * 32);
            int32_t d1 = compute_block_dot(&row_weights[b+1], activations + (b+1) * 32);
            int32_t d2 = compute_block_dot(&row_weights[b+2], activations + (b+2) * 32);
            int32_t d3 = compute_block_dot(&row_weights[b+3], activations + (b+3) * 32);
            
            int32_t dot_arr[4] = {d0, d1, d2, d3};
            int32x4_t dots = vld1q_s32(dot_arr);
            
            // Compute scales inline (slower due to FP16->FP32 conversion)
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

// Multi-threaded version
void batched_gemv_q4_0_threaded(
    size_t n_out,
    size_t n_in,
    const batched_q4_0_block* weights,
    const int8_t* activations,
    const float* combined_scales,
    float* output,
    int thread_id,
    int n_threads
) {
    size_t nb = n_in / BATCHED_GEMV_BLOCK_SIZE;
    
    // Divide rows among threads
    size_t rows_per_thread = (n_out + n_threads - 1) / n_threads;
    size_t row_start = thread_id * rows_per_thread;
    size_t row_end = row_start + rows_per_thread;
    if (row_end > n_out) row_end = n_out;
    
    for (size_t row = row_start; row < row_end; row++) {
        const batched_q4_0_block* row_weights = weights + row * nb;
        const float* row_scales = combined_scales + row * nb;
        
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        
        size_t b = 0;
        
        for (; b + 7 < nb; b += 8) {
            int32_t d0 = compute_block_dot(&row_weights[b], activations + b * 32);
            int32_t d1 = compute_block_dot(&row_weights[b+1], activations + (b+1) * 32);
            int32_t d2 = compute_block_dot(&row_weights[b+2], activations + (b+2) * 32);
            int32_t d3 = compute_block_dot(&row_weights[b+3], activations + (b+3) * 32);
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
