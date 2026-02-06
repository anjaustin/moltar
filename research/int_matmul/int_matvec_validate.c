/*
 * Integer Matvec Validation - Numerical accuracy + speed test
 * 
 * Tests fixed-point scale approach against float baseline.
 * Uses real Q4_0 scale distributions and validates per-element error.
 *
 * Build for Android:
 *   $NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang \
 *     -O3 -march=armv8.2-a+dotprod -o int_matvec_validate int_matvec_validate.c -static -lm
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

// Q4_0 block structure (matches llama.cpp)
typedef struct {
    uint16_t d;        // FP16 scale
    uint8_t qs[16];    // 32 x 4-bit weights packed
} block_q4_0;

// FP16 <-> FP32 conversion
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

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant = (mant | 0x400) >> (1 - exp);
        return sign | mant;
    } else if (exp >= 31) {
        return sign | 0x7C00;
    }
    return sign | (exp << 10) | mant;
}

// ============================================================================
// BASELINE: Float scales per block (KleidiAI style)
// ============================================================================
__attribute__((noinline))
void matvec_float_baseline(
    int n_out, int n_in,
    const block_q4_0* weights,
    const int8_t* activations,
    const float* act_scales,
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        float sum = 0.0f;
        
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
            int32_t dot = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
            
            float w_scale = fp16_to_fp32(row_weights[b].d);
            float combined = w_scale * act_scales[b];
            sum += (float)dot * combined;
        }
        
        output[row] = sum;
    }
}

// ============================================================================
// VERSION 1: Deferred float - integer inner loop, float at end
// ============================================================================
__attribute__((noinline))
void matvec_deferred_v1(
    int n_out, int n_in,
    const block_q4_0* weights,
    const int8_t* activations,
    const float* act_scales,
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    int32_t* block_dots = (int32_t*)aligned_alloc(64, nb * sizeof(int32_t));
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        
        // Pass 1: Pure integer dot products
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
        
        // Pass 2: Float scaling (vectorized)
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            float w_scale = fp16_to_fp32(row_weights[b].d);
            sum += (float)block_dots[b] * w_scale * act_scales[b];
        }
        
        output[row] = sum;
    }
    
    free(block_dots);
}

// ============================================================================
// VERSION 2: Fixed-point scales with INT32 accumulator
// Uses 12-bit fixed-point for scales (fits in int32 accumulator)
// ============================================================================
#define FP_BITS 12
#define FP_SCALE (1 << FP_BITS)

typedef struct {
    int16_t w_scale_fp;   // Fixed-point weight scale
    int16_t a_scale_fp;   // Fixed-point activation scale
} fp_scales_t;

__attribute__((noinline))
void matvec_fixedpoint_v2(
    int n_out, int n_in,
    const block_q4_0* weights,
    const int8_t* activations,
    const int16_t* w_scales_fp,    // [n_out * nb] pre-converted
    const int16_t* a_scales_fp,    // [nb] pre-converted
    float global_scale,             // To convert back to float
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        const int16_t* row_w_scales = w_scales_fp + row * nb;
        
        int64_t acc = 0;
        
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
            int32_t dot = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
            
            // Integer scale multiply
            int32_t scaled = dot * (int32_t)row_w_scales[b];
            acc += (int64_t)scaled * a_scales_fp[b];
        }
        
        // Single float conversion per output
        output[row] = (float)acc * global_scale;
    }
}

// ============================================================================
// VERSION 3: Hybrid - integer dots, precomputed combined scales as float
// ============================================================================
__attribute__((noinline))
void matvec_hybrid_v3(
    int n_out, int n_in,
    const block_q4_0* weights,
    const int8_t* activations,
    const float* combined_scales,   // [n_out * nb] = w_scale * a_scale precomputed
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        const float* row_scales = combined_scales + row * nb;
        
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        // Process 4 blocks at a time
        int b = 0;
        for (; b + 3 < nb; b += 4) {
            int32_t dots[4];
            
            for (int i = 0; i < 4; i++) {
                uint8x16_t q4_packed = vld1q_u8(row_weights[b + i].qs);
                int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
                int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
                q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
                q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
                
                int8x16_t act_lo = vld1q_s8(activations + (b + i) * 32);
                int8x16_t act_hi = vld1q_s8(activations + (b + i) * 32 + 16);
                
                int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
                int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
                dots[i] = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
            }
            
            // Vectorized scale + accumulate
            int32x4_t dot_vec = vld1q_s32(dots);
            float32x4_t scale_vec = vld1q_f32(row_scales + b);
            acc = vmlaq_f32(acc, vcvtq_f32_s32(dot_vec), scale_vec);
        }
        
        float sum = vaddvq_f32(acc);
        
        // Remainder
        for (; b < nb; b++) {
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
            
            sum += (float)dot * row_scales[b];
        }
        
        output[row] = sum;
    }
}

// ============================================================================
// Validation helpers
// ============================================================================

void compute_error_stats(
    const float* ref, const float* test, int n,
    float* max_abs_err, float* max_rel_err, float* rmse
) {
    float sum_sq_err = 0;
    *max_abs_err = 0;
    *max_rel_err = 0;
    
    for (int i = 0; i < n; i++) {
        float abs_err = fabsf(ref[i] - test[i]);
        float rel_err = (fabsf(ref[i]) > 1e-6f) ? abs_err / fabsf(ref[i]) : 0;
        
        if (abs_err > *max_abs_err) *max_abs_err = abs_err;
        if (rel_err > *max_rel_err) *max_rel_err = rel_err;
        sum_sq_err += abs_err * abs_err;
    }
    
    *rmse = sqrtf(sum_sq_err / n);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    int n_in = 4608;   // LFM2 FFN input dim
    int n_out = 1024;  // Output channels
    int iters = 1000;
    
    if (argc > 1) n_in = atoi(argv[1]);
    if (argc > 2) n_out = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);
    
    int nb = n_in / BLOCK_SIZE;
    
    printf("Integer Matvec Validation\n");
    printf("=========================\n");
    printf("Matrix: %d x %d (%d blocks/row)\n", n_out, n_in, nb);
    printf("Iterations: %d\n\n", iters);
    
    // Allocate
    block_q4_0* weights = aligned_alloc(64, n_out * nb * sizeof(block_q4_0));
    int8_t* activations = aligned_alloc(64, n_in);
    float* act_scales = aligned_alloc(64, nb * sizeof(float));
    int16_t* w_scales_fp = aligned_alloc(64, n_out * nb * sizeof(int16_t));
    int16_t* a_scales_fp = aligned_alloc(64, nb * sizeof(int16_t));
    float* combined_scales = aligned_alloc(64, n_out * nb * sizeof(float));
    
    float* output_ref = aligned_alloc(64, n_out * sizeof(float));
    float* output_test = aligned_alloc(64, n_out * sizeof(float));
    
    // Initialize with realistic data
    srand(42);
    
    // Q4_0 scales are typically in range [0.0001, 0.1]
    // Activation scales similar
    float max_w_scale = 0, max_a_scale = 0;
    
    for (int i = 0; i < n_out * nb; i++) {
        // Realistic Q4_0 scale distribution (log-normal-ish)
        float ws = 0.0001f + (rand() % 1000) * 0.0001f;  // 0.0001 to 0.1
        weights[i].d = fp32_to_fp16(ws);
        
        for (int j = 0; j < 16; j++) {
            weights[i].qs[j] = rand() & 0xFF;
        }
        
        if (ws > max_w_scale) max_w_scale = ws;
    }
    
    for (int b = 0; b < nb; b++) {
        act_scales[b] = 0.001f + (rand() % 100) * 0.001f;  // 0.001 to 0.1
        if (act_scales[b] > max_a_scale) max_a_scale = act_scales[b];
    }
    
    for (int i = 0; i < n_in; i++) {
        activations[i] = (rand() % 256) - 128;
    }
    
    // Convert to fixed-point scales
    float w_scale_unit = max_w_scale / FP_SCALE;
    float a_scale_unit = max_a_scale / FP_SCALE;
    float global_scale = w_scale_unit * a_scale_unit;
    
    for (int i = 0; i < n_out * nb; i++) {
        float ws = fp16_to_fp32(weights[i].d);
        w_scales_fp[i] = (int16_t)(ws / w_scale_unit + 0.5f);
    }
    
    for (int b = 0; b < nb; b++) {
        a_scales_fp[b] = (int16_t)(act_scales[b] / a_scale_unit + 0.5f);
    }
    
    // Precompute combined scales for hybrid version
    for (int row = 0; row < n_out; row++) {
        for (int b = 0; b < nb; b++) {
            float ws = fp16_to_fp32(weights[row * nb + b].d);
            combined_scales[row * nb + b] = ws * act_scales[b];
        }
    }
    
    printf("Scale ranges: w=[%.4f, %.4f], a=[%.4f, %.4f]\n",
           0.0001f, max_w_scale, 0.001f, max_a_scale);
    printf("Fixed-point: %d bits, global_scale=%.2e\n\n", FP_BITS, global_scale);
    
    // Compute reference
    matvec_float_baseline(n_out, n_in, weights, activations, act_scales, output_ref);
    
    // Validate each version
    float max_abs, max_rel, rmse;
    struct timespec start, end;
    
    printf("%-25s %10s %12s %12s %10s\n",
           "Version", "Time(us)", "MaxAbsErr", "MaxRelErr", "RMSE");
    printf("%-25s %10s %12s %12s %10s\n",
           "-------------------------", "----------", "------------", "------------", "----------");
    
    // Baseline
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_float_baseline(n_out, n_in, weights, activations, act_scales, output_test);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double baseline_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    compute_error_stats(output_ref, output_test, n_out, &max_abs, &max_rel, &rmse);
    printf("%-25s %10.1f %12.2e %12.2e %10.2e\n",
           "Float baseline", baseline_us, max_abs, max_rel, rmse);
    
    // Deferred v1
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_deferred_v1(n_out, n_in, weights, activations, act_scales, output_test);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double deferred_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    compute_error_stats(output_ref, output_test, n_out, &max_abs, &max_rel, &rmse);
    printf("%-25s %10.1f %12.2e %12.2e %10.2e  (%.2fx)\n",
           "Deferred v1", deferred_us, max_abs, max_rel, rmse, baseline_us / deferred_us);
    
    // Fixed-point v2
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_fixedpoint_v2(n_out, n_in, weights, activations, w_scales_fp, a_scales_fp, global_scale, output_test);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double fixedpoint_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    compute_error_stats(output_ref, output_test, n_out, &max_abs, &max_rel, &rmse);
    printf("%-25s %10.1f %12.2e %12.2e %10.2e  (%.2fx)\n",
           "Fixed-point v2", fixedpoint_us, max_abs, max_rel, rmse, baseline_us / fixedpoint_us);
    
    // Hybrid v3
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_hybrid_v3(n_out, n_in, weights, activations, combined_scales, output_test);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double hybrid_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    compute_error_stats(output_ref, output_test, n_out, &max_abs, &max_rel, &rmse);
    printf("%-25s %10.1f %12.2e %12.2e %10.2e  (%.2fx)\n",
           "Hybrid v3", hybrid_us, max_abs, max_rel, rmse, baseline_us / hybrid_us);
    
    // Print sample outputs
    printf("\nSample outputs (first 5):\n");
    printf("%-10s", "Index");
    printf("%15s", "Reference");
    
    matvec_deferred_v1(n_out, n_in, weights, activations, act_scales, output_test);
    printf("%15s", "Deferred");
    
    float* output_fp = aligned_alloc(64, n_out * sizeof(float));
    matvec_fixedpoint_v2(n_out, n_in, weights, activations, w_scales_fp, a_scales_fp, global_scale, output_fp);
    printf("%15s", "FixedPt");
    
    float* output_hyb = aligned_alloc(64, n_out * sizeof(float));
    matvec_hybrid_v3(n_out, n_in, weights, activations, combined_scales, output_hyb);
    printf("%15s\n", "Hybrid");
    
    for (int i = 0; i < 5; i++) {
        printf("%-10d%15.4f%15.4f%15.4f%15.4f\n",
               i, output_ref[i], output_test[i], output_fp[i], output_hyb[i]);
    }
    
    // Memory analysis
    printf("\n--- Memory Analysis ---\n");
    size_t weight_bytes = n_out * nb * sizeof(block_q4_0);
    size_t extra_fp_bytes = n_out * nb * sizeof(int16_t) + nb * sizeof(int16_t);
    size_t extra_hybrid_bytes = n_out * nb * sizeof(float);
    
    printf("Weight data:     %.2f MB\n", weight_bytes / (1024.0 * 1024.0));
    printf("Fixed-pt extra:  %.2f MB (+%.1f%%)\n", 
           extra_fp_bytes / (1024.0 * 1024.0),
           100.0 * extra_fp_bytes / weight_bytes);
    printf("Hybrid extra:    %.2f MB (+%.1f%%)\n",
           extra_hybrid_bytes / (1024.0 * 1024.0),
           100.0 * extra_hybrid_bytes / weight_bytes);
    
    free(weights);
    free(activations);
    free(act_scales);
    free(w_scales_fp);
    free(a_scales_fp);
    free(combined_scales);
    free(output_ref);
    free(output_test);
    free(output_fp);
    free(output_hyb);
    
    return 0;
}
