/*
 * Integer-only Q4_0 × Q8_0 matmul with DEFERRED float conversion
 * 
 * The insight: KleidiAI does per-block float ops:
 *   fcvtl, fcvtl, fmul, scvtf, fmla = 5 float ops per 32 weights
 * 
 * We can defer ALL float ops to the end:
 *   1. Accumulate SDOT results as int32 per block
 *   2. Store block-wise int32 sums + scale indices
 *   3. Single vectorized float pass at the end
 * 
 * Or even better: fixed-point scales
 *   1. Pre-convert FP16 scales to INT16 (global scale factor)
 *   2. Accumulate int32_dot * int16_scale as int64
 *   3. Single float conversion per output channel
 * 
 * Build for Android:
 *   $NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android35-clang \
 *     -O3 -march=armv8.2-a+dotprod -o int_matmul_deferred int_matmul_deferred.c -lm
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
#define SCALE_BITS 14  // Fixed-point scale precision

// Q4_0 block: 32 weights in 18 bytes
typedef struct {
    uint16_t d;        // FP16 scale (stored as raw bits)
    uint8_t qs[16];    // 32 x 4-bit weights packed
} block_q4_0;

// Convert FP16 bits to FP32
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

// ============================================================================
// VERSION 1: KleidiAI-style (float per block) - BASELINE
// ============================================================================
__attribute__((noinline))
void matvec_kleidiai_style(
    int n_out, int n_in,
    const block_q4_0* weights,  // [n_out][n_in/32] blocks
    const int8_t* activations,  // [n_in] quantized activations  
    const float* act_scales,    // [n_in/32] activation scales per block
    float* output               // [n_out]
) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        const block_q4_0* row_weights = weights + row * nb;
        
        for (int b = 0; b < nb; b++) {
            // Load and unpack Q4 weights
            uint8x16_t q4_packed = vld1q_u8(row_weights[b].qs);
            int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
            int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
            q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
            q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
            
            // Load activations
            int8x16_t act_lo = vld1q_s8(activations + b * 32);
            int8x16_t act_hi = vld1q_s8(activations + b * 32 + 16);
            
            // SDOT
            int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
            int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
            int32_t dot = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
            
            // Per-block float operations (THIS IS THE COST)
            float w_scale = fp16_to_fp32(row_weights[b].d);
            float a_scale = act_scales[b];
            float combined_scale = w_scale * a_scale;
            
            // scvtf + fmla
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(vdupq_n_s32(dot)), combined_scale);
        }
        
        output[row] = vaddvq_f32(acc);
    }
}

// ============================================================================
// VERSION 2: Integer accumulate, deferred float (per output channel)
// ============================================================================
__attribute__((noinline))
void matvec_deferred_float(
    int n_out, int n_in,
    const block_q4_0* weights,
    const int8_t* activations,
    const float* act_scales,
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    
    // Temporary storage for block-wise integer results
    int32_t* block_dots = (int32_t*)aligned_alloc(64, nb * sizeof(int32_t));
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        
        // Pass 1: Integer-only dot products
        for (int b = 0; b < nb; b++) {
            uint8x16_t q4_packed = vld1q_u8(row_weights[b].qs);
            int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
            int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
            q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
            q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
            
            int8x16_t act_lo = vld1q_s8(activations + b * 32);
            int8x16_t act_hi = vld1q_s8(activations + b * 32 + 16);
            
            int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
            int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
            block_dots[b] = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
        }
        
        // Pass 2: Vectorized float scaling (4 blocks at a time)
        float sum = 0.0f;
        int b = 0;
        for (; b + 3 < nb; b += 4) {
            // Load 4 integer dots
            int32x4_t dots = vld1q_s32(block_dots + b);
            
            // Load and combine scales
            float scales[4];
            for (int i = 0; i < 4; i++) {
                scales[i] = fp16_to_fp32(row_weights[b + i].d) * act_scales[b + i];
            }
            float32x4_t scale_v = vld1q_f32(scales);
            
            // Vectorized: convert + multiply + accumulate
            float32x4_t result = vmulq_f32(vcvtq_f32_s32(dots), scale_v);
            sum += vaddvq_f32(result);
        }
        
        // Handle remainder
        for (; b < nb; b++) {
            float w_scale = fp16_to_fp32(row_weights[b].d);
            float a_scale = act_scales[b];
            sum += (float)block_dots[b] * w_scale * a_scale;
        }
        
        output[row] = sum;
    }
    
    free(block_dots);
}

// ============================================================================
// VERSION 3: Fixed-point scales (INT16), INT64 accumulator
// ============================================================================
__attribute__((noinline))
void matvec_fixedpoint(
    int n_out, int n_in,
    const block_q4_0* weights,
    const int8_t* activations,
    const int16_t* w_scales_i16,   // Pre-converted weight scales [n_out][nb]
    const int16_t* act_scales_i16, // Pre-converted activation scales [nb]
    float w_scale_factor,          // Global weight scale factor
    float act_scale_factor,        // Global activation scale factor
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    float global_scale = w_scale_factor * act_scale_factor;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        const int16_t* row_w_scales = w_scales_i16 + row * nb;
        
        int64_t acc = 0;  // 64-bit accumulator for precision
        
        for (int b = 0; b < nb; b++) {
            // Integer dot product
            uint8x16_t q4_packed = vld1q_u8(row_weights[b].qs);
            int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
            int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
            q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
            q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
            
            int8x16_t act_lo = vld1q_s8(activations + b * 32);
            int8x16_t act_hi = vld1q_s8(activations + b * 32 + 16);
            
            int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
            int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
            int32_t dot = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
            
            // Integer scale multiply (no float!)
            int32_t scaled = dot * (int32_t)row_w_scales[b] * (int32_t)act_scales_i16[b];
            acc += scaled;
        }
        
        // Single float conversion per output channel
        output[row] = (float)acc * global_scale / (float)(1 << (2 * SCALE_BITS));
    }
}

// ============================================================================
// VERSION 4: Simple deferred - store block results, vectorized scale at end
// ============================================================================
__attribute__((noinline))
void matvec_simple_deferred(
    int n_out, int n_in,
    const block_q4_0* weights,
    const int8_t* activations,
    const float* act_scales,
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        
        // Integer-only inner loop - no float ops
        int32_t block_sums[256];  // Max 256 blocks (8K input dim)
        
        for (int b = 0; b < nb; b++) {
            uint8x16_t q4_packed = vld1q_u8(row_weights[b].qs);
            int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
            int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
            q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
            q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
            
            int8x16_t act_lo = vld1q_s8(activations + b * 32);
            int8x16_t act_hi = vld1q_s8(activations + b * 32 + 16);
            
            int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
            int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
            block_sums[b] = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
        }
        
        // Float operations - vectorized, at the end
        float sum = 0.0f;
        int b = 0;
        
        // Process 4 blocks at a time with NEON
        for (; b + 3 < nb; b += 4) {
            int32x4_t sums = vld1q_s32(block_sums + b);
            
            // Compute combined scales
            float32x4_t scales;
            scales = vsetq_lane_f32(fp16_to_fp32(row_weights[b].d) * act_scales[b], scales, 0);
            scales = vsetq_lane_f32(fp16_to_fp32(row_weights[b+1].d) * act_scales[b+1], scales, 1);
            scales = vsetq_lane_f32(fp16_to_fp32(row_weights[b+2].d) * act_scales[b+2], scales, 2);
            scales = vsetq_lane_f32(fp16_to_fp32(row_weights[b+3].d) * act_scales[b+3], scales, 3);
            
            float32x4_t result = vmulq_f32(vcvtq_f32_s32(sums), scales);
            sum += vaddvq_f32(result);
        }
        
        // Remainder
        for (; b < nb; b++) {
            float w_scale = fp16_to_fp32(row_weights[b].d);
            sum += (float)block_sums[b] * w_scale * act_scales[b];
        }
        
        output[row] = sum;
    }
}

// ============================================================================
// VERSION 5: Fused integer (blocked, no intermediate storage)
// ============================================================================
__attribute__((noinline))
void matvec_fused_int(
    int n_out, int n_in,
    const block_q4_0* weights,
    const int8_t* activations,
    const int16_t* w_scales_i16,
    const int16_t* act_scales_i16,
    float w_scale_factor,
    float act_scale_factor,
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    float global_scale = w_scale_factor * act_scale_factor / (float)(1 << (2 * SCALE_BITS));
    
    // Process 4 output channels at a time
    for (int row = 0; row < n_out; row += 4) {
        int rows_this_iter = (row + 4 <= n_out) ? 4 : (n_out - row);
        
        int64_t acc[4] = {0, 0, 0, 0};
        
        for (int b = 0; b < nb; b++) {
            int16_t a_scale = act_scales_i16[b];
            int8x16_t act_lo = vld1q_s8(activations + b * 32);
            int8x16_t act_hi = vld1q_s8(activations + b * 32 + 16);
            
            for (int r = 0; r < rows_this_iter; r++) {
                const block_q4_0* blk = weights + (row + r) * nb + b;
                
                uint8x16_t q4_packed = vld1q_u8(blk->qs);
                int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
                int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
                q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
                q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
                
                int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
                int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
                int32_t dot = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
                
                int16_t w_scale = w_scales_i16[(row + r) * nb + b];
                acc[r] += (int64_t)dot * w_scale * a_scale;
            }
        }
        
        for (int r = 0; r < rows_this_iter; r++) {
            output[row + r] = (float)acc[r] * global_scale;
        }
    }
}

// ============================================================================
// Benchmark setup
// ============================================================================

void prepare_test_data(
    int n_out, int n_in,
    block_q4_0** weights,
    int8_t** activations,
    float** act_scales,
    int16_t** w_scales_i16,
    int16_t** act_scales_i16,
    float* w_scale_factor,
    float* act_scale_factor
) {
    int nb = n_in / BLOCK_SIZE;
    
    *weights = (block_q4_0*)aligned_alloc(64, n_out * nb * sizeof(block_q4_0));
    *activations = (int8_t*)aligned_alloc(64, n_in * sizeof(int8_t));
    *act_scales = (float*)aligned_alloc(64, nb * sizeof(float));
    *w_scales_i16 = (int16_t*)aligned_alloc(64, n_out * nb * sizeof(int16_t));
    *act_scales_i16 = (int16_t*)aligned_alloc(64, nb * sizeof(int16_t));
    
    srand(42);
    
    // Initialize with realistic values
    float max_w_scale = 0, max_a_scale = 0;
    
    for (int i = 0; i < n_out * nb; i++) {
        // Random weight scale (typical range 0.001-0.1)
        float ws = 0.001f + (rand() % 100) * 0.001f;
        (*weights)[i].d = 0x3C00;  // Placeholder FP16 (~1.0)
        
        // Random quantized weights
        for (int j = 0; j < 16; j++) {
            (*weights)[i].qs[j] = rand() & 0xFF;
        }
        
        if (ws > max_w_scale) max_w_scale = ws;
    }
    
    for (int b = 0; b < nb; b++) {
        (*act_scales)[b] = 0.01f + (rand() % 50) * 0.001f;
        if ((*act_scales)[b] > max_a_scale) max_a_scale = (*act_scales)[b];
    }
    
    for (int i = 0; i < n_in; i++) {
        (*activations)[i] = (rand() % 256) - 128;
    }
    
    // Convert to fixed-point scales - use the SAME scales as float version
    *w_scale_factor = max_w_scale / (1 << SCALE_BITS);
    *act_scale_factor = max_a_scale / (1 << SCALE_BITS);
    
    // Store actual float scales for later conversion
    float* w_scales_float = (float*)malloc(n_out * nb * sizeof(float));
    srand(42);  // Reset seed to match
    for (int i = 0; i < n_out * nb; i++) {
        float ws = 0.001f + (rand() % 100) * 0.001f;
        w_scales_float[i] = ws;
        // Skip the qs regeneration
        for (int j = 0; j < 16; j++) rand();
    }
    
    // Now convert stored scales to fixed-point
    for (int i = 0; i < n_out * nb; i++) {
        (*w_scales_i16)[i] = (int16_t)(w_scales_float[i] / *w_scale_factor);
    }
    free(w_scales_float);
    
    for (int b = 0; b < nb; b++) {
        (*act_scales_i16)[b] = (int16_t)((*act_scales)[b] / *act_scale_factor);
    }
}

int main(int argc, char** argv) {
    // LFM2 FFN dimensions: 4608 x 1024
    int n_in = 4608;
    int n_out = 1024;
    int iters = 1000;
    
    if (argc > 1) n_in = atoi(argv[1]);
    if (argc > 2) n_out = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);
    
    printf("Deferred Float Matmul Benchmark\n");
    printf("================================\n");
    printf("Matrix: %d x %d, Iterations: %d\n\n", n_out, n_in, iters);
    
    block_q4_0* weights;
    int8_t* activations;
    float* act_scales;
    int16_t* w_scales_i16;
    int16_t* act_scales_i16;
    float w_scale_factor, act_scale_factor;
    
    prepare_test_data(n_out, n_in, &weights, &activations, &act_scales,
                      &w_scales_i16, &act_scales_i16, &w_scale_factor, &act_scale_factor);
    
    float* output = (float*)aligned_alloc(64, n_out * sizeof(float));
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        matvec_kleidiai_style(n_out, n_in, weights, activations, act_scales, output);
    }
    
    struct timespec start, end;
    
    // Benchmark 1: KleidiAI-style
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_kleidiai_style(n_out, n_in, weights, activations, act_scales, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double kleidiai_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    printf("KleidiAI-style (baseline):  %8.1f us\n", kleidiai_us);
    float ref_sum = 0;
    for (int i = 0; i < n_out; i++) ref_sum += output[i];
    
    // Benchmark 2: Deferred float
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_deferred_float(n_out, n_in, weights, activations, act_scales, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double deferred_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    float def_sum = 0;
    for (int i = 0; i < n_out; i++) def_sum += output[i];
    printf("Deferred float:             %8.1f us (%.2fx) [sum diff: %.2e]\n", 
           deferred_us, kleidiai_us / deferred_us, fabsf(def_sum - ref_sum) / fabsf(ref_sum));
    
    // Benchmark 3: Fixed-point
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_fixedpoint(n_out, n_in, weights, activations, w_scales_i16, act_scales_i16,
                         w_scale_factor, act_scale_factor, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double fixedpoint_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    float fp_sum = 0;
    for (int i = 0; i < n_out; i++) fp_sum += output[i];
    printf("Fixed-point scales:         %8.1f us (%.2fx) [sum diff: %.2e]\n",
           fixedpoint_us, kleidiai_us / fixedpoint_us, fabsf(fp_sum - ref_sum) / fabsf(ref_sum));
    
    // Benchmark 4: Fused integer
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_fused_int(n_out, n_in, weights, activations, w_scales_i16, act_scales_i16,
                        w_scale_factor, act_scale_factor, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double fused_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    float fused_sum = 0;
    for (int i = 0; i < n_out; i++) fused_sum += output[i];
    printf("Fused integer:              %8.1f us (%.2fx) [sum diff: %.2e]\n",
           fused_us, kleidiai_us / fused_us, fabsf(fused_sum - ref_sum) / fabsf(ref_sum));
    
    // Benchmark 5: Simple deferred
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_simple_deferred(n_out, n_in, weights, activations, act_scales, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double simple_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    float simple_sum = 0;
    for (int i = 0; i < n_out; i++) simple_sum += output[i];
    printf("Simple deferred:            %8.1f us (%.2fx) [sum diff: %.2e]\n",
           simple_us, kleidiai_us / simple_us, fabsf(simple_sum - ref_sum) / fabsf(ref_sum));
    
    printf("\n--- Memory Analysis ---\n");
    size_t weight_bytes = n_out * (n_in / 32) * sizeof(block_q4_0);
    size_t act_bytes = n_in * sizeof(int8_t) + (n_in / 32) * sizeof(float);
    printf("Weights: %.2f MB, Activations: %.2f KB\n", 
           weight_bytes / (1024.0 * 1024.0), act_bytes / 1024.0);
    printf("Bandwidth (KleidiAI): %.2f GB/s\n", 
           (weight_bytes + act_bytes) * 1e6 / kleidiai_us / 1e9);
    printf("Bandwidth (Fixed-pt): %.2f GB/s\n",
           (weight_bytes + act_bytes) * 1e6 / fixedpoint_us / 1e9);
    
    free(weights);
    free(activations);
    free(act_scales);
    free(w_scales_i16);
    free(act_scales_i16);
    free(output);
    
    return 0;
}
