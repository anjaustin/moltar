/*
 * Integer-only Q4_0 × Q8_0 matmul for ARM NEON with DOTPROD
 * 
 * Goal: Measure the overhead of F32 scale operations vs pure integer
 * 
 * Q4_0 format: 32 4-bit weights + 1 FP16 scale per block
 * Q8_0 format: 32 8-bit values + 1 FP16 scale per block
 */

#define _GNU_SOURCE
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Android doesn't have aligned_alloc in older API levels
extern void* memalign(size_t alignment, size_t size);
#define aligned_alloc(align, size) memalign(align, size)

// Q4_0 block: 32 weights in 18 bytes
typedef struct {
    uint16_t d;        // FP16 scale (stored as raw bits)
    uint8_t qs[16];    // 32 x 4-bit weights packed
} block_q4_0;

// Q8_0 block: 32 values in 34 bytes  
typedef struct {
    uint16_t d;        // FP16 scale
    int8_t qs[32];     // 32 x 8-bit values
} block_q8_0;

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
// VERSION 1: Standard (what KleidiAI does)
// Uses NEON DOTPROD, accumulates in FP32 with scale multiply per block
// ============================================================================
__attribute__((noinline))
float vec_dot_standard(int n, const block_q4_0* x, const block_q8_0* y) {
    int nb = n / 32;
    float32x4_t sumv = vdupq_n_f32(0.0f);
    
    for (int i = 0; i < nb; i++) {
        // Load Q4_0 weights and unpack to INT8
        uint8x16_t q4_packed = vld1q_u8(x[i].qs);
        int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
        int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
        
        // Subtract 8 to center around zero (Q4_0 stores 0-15, we want -8 to +7)
        q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
        q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
        
        // Load Q8_0 activations
        int8x16_t q8_lo = vld1q_s8(y[i].qs);
        int8x16_t q8_hi = vld1q_s8(y[i].qs + 16);
        
        // Dot product using SDOT (4 x INT8 -> INT32)
        int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, q8_lo);
        int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, q8_hi);
        int32x4_t dot = vaddq_s32(dot_lo, dot_hi);
        
        // Sum the 4 lanes
        int32_t sum = vaddvq_s32(dot);
        
        // Scale and accumulate - THIS IS THE F32 PART
        float scale = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(vdupq_n_s32(sum)), scale);
    }
    
    return vaddvq_f32(sumv);
}

// ============================================================================
// VERSION 2: Pure integer dot (no scales at all)
// Returns raw INT32 sum - caller must apply global scale
// ============================================================================
__attribute__((noinline))
int32_t vec_dot_pure_int(int n, const uint8_t* x_qs, const int8_t* y_qs) {
    int nb = n / 32;
    int32x4_t sumv = vdupq_n_s32(0);
    
    for (int i = 0; i < nb; i++) {
        // Load Q4_0 weights and unpack to INT8
        uint8x16_t q4_packed = vld1q_u8(x_qs + i * 16);
        int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
        int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
        q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
        q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
        
        // Load Q8_0 activations (no scale)
        int8x16_t q8_lo = vld1q_s8(y_qs + i * 32);
        int8x16_t q8_hi = vld1q_s8(y_qs + i * 32 + 16);
        
        // Accumulate dot products
        sumv = vdotq_s32(sumv, q4_lo, q8_lo);
        sumv = vdotq_s32(sumv, q4_hi, q8_hi);
    }
    
    return vaddvq_s32(sumv);
}

// ============================================================================
// VERSION 3: Batched 4 blocks at once with vectorized scale
// ============================================================================
__attribute__((noinline))
float vec_dot_batched4(int n, const block_q4_0* x, const block_q8_0* y) {
    int nb = n / 32;
    float32x4_t sumv = vdupq_n_f32(0.0f);
    
    // Process 4 blocks at a time
    int i = 0;
    for (; i + 3 < nb; i += 4) {
        int32_t dot_arr[4];
        float scales[4];
        
        // Compute 4 dot products
        for (int j = 0; j < 4; j++) {
            uint8x16_t q4_packed = vld1q_u8(x[i+j].qs);
            int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
            int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
            q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
            q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
            
            int8x16_t q8_lo = vld1q_s8(y[i+j].qs);
            int8x16_t q8_hi = vld1q_s8(y[i+j].qs + 16);
            
            int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, q8_lo);
            int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, q8_hi);
            
            dot_arr[j] = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
            scales[j] = fp16_to_fp32(x[i+j].d) * fp16_to_fp32(y[i+j].d);
        }
        
        // Vectorized scale multiply
        int32x4_t dots = vld1q_s32(dot_arr);
        float32x4_t scale_v = vld1q_f32(scales);
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(dots), scale_v);
    }
    
    // Handle remainder
    for (; i < nb; i++) {
        uint8x16_t q4_packed = vld1q_u8(x[i].qs);
        int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
        int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
        q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
        q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
        
        int8x16_t q8_lo = vld1q_s8(y[i].qs);
        int8x16_t q8_hi = vld1q_s8(y[i].qs + 16);
        
        int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, q8_lo);
        int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, q8_hi);
        int32_t dot = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
        
        float scale = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(vdupq_n_s32(dot)), scale);
    }
    
    return vaddvq_f32(sumv);
}

// ============================================================================
// VERSION 4: Fixed-point scales (16-bit integer scales)
// Approximate FP16 scale as INT16, accumulate in INT64
// ============================================================================
__attribute__((noinline))
float vec_dot_fixedpoint(int n, const block_q4_0* x, const block_q8_0* y,
                         const int16_t* x_scales_i16, const int16_t* y_scales_i16,
                         float x_scale_factor, float y_scale_factor) {
    int nb = n / 32;
    int64_t acc = 0;
    
    for (int i = 0; i < nb; i++) {
        // Load Q4_0 weights and unpack to INT8
        uint8x16_t q4_packed = vld1q_u8(x[i].qs);
        int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
        int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
        q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
        q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
        
        int8x16_t q8_lo = vld1q_s8(y[i].qs);
        int8x16_t q8_hi = vld1q_s8(y[i].qs + 16);
        
        int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, q8_lo);
        int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, q8_hi);
        int32_t dot = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
        
        // Integer scale multiply (16-bit scales)
        int32_t scaled = dot * (int32_t)x_scales_i16[i] * (int32_t)y_scales_i16[i];
        acc += scaled;
    }
    
    // Final conversion to float with scale correction
    return (float)acc * x_scale_factor * y_scale_factor;
}

// ============================================================================
// VERSION 5: Deferred scale - accumulate int products, apply scales at end
// ============================================================================
__attribute__((noinline))
float vec_dot_deferred(int n, const block_q4_0* x, const block_q8_0* y) {
    int nb = n / 32;
    
    // Allocate temp storage for integer dot products
    int32_t* dots = (int32_t*)aligned_alloc(64, nb * sizeof(int32_t));
    
    // First pass: compute all integer dot products
    int32x4_t sumv = vdupq_n_s32(0);
    for (int i = 0; i < nb; i++) {
        uint8x16_t q4_packed = vld1q_u8(x[i].qs);
        int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
        int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
        q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
        q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
        
        int8x16_t q8_lo = vld1q_s8(y[i].qs);
        int8x16_t q8_hi = vld1q_s8(y[i].qs + 16);
        
        int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, q8_lo);
        int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, q8_hi);
        dots[i] = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
    }
    
    // Second pass: apply scales
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        float scale = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        sum += (float)dots[i] * scale;
    }
    
    free(dots);
    return sum;
}

int main() {
    // Benchmark setup
    // Test multiple sizes: 1K (cache-friendly) and 1M (memory-bound)
    const int SIZES[] = {1024, 4096, 32768, 262144};  // 1K, 4K, 32K, 256K elements
    const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
    
    for (int sz = 0; sz < NUM_SIZES; sz++) {
    const int K = SIZES[sz];
    const int ITERS = 100000000 / K;  // Scale iterations to keep runtime reasonable
    
    int nb = K / 32;
    
    // Allocate aligned memory
    block_q4_0* weights = aligned_alloc(64, nb * sizeof(block_q4_0));
    block_q8_0* activations = aligned_alloc(64, nb * sizeof(block_q8_0));
    
    // Packed arrays for pure-int version
    uint8_t* w_qs = aligned_alloc(64, nb * 16);
    int8_t* a_qs = aligned_alloc(64, nb * 32);
    
    // Fixed-point scales
    int16_t* w_scales = aligned_alloc(64, nb * sizeof(int16_t));
    int16_t* a_scales = aligned_alloc(64, nb * sizeof(int16_t));
    
    // Initialize with random data
    srand(42);
    float max_w_scale = 0, max_a_scale = 0;
    for (int i = 0; i < nb; i++) {
        // Random scales in reasonable range (0.001 to 0.1)
        float ws = 0.001f + (rand() % 100) * 0.001f;
        float as = 0.001f + (rand() % 100) * 0.001f;
        
        // Store as FP16 (simplified)
        weights[i].d = 0x3C00;  // ~1.0
        activations[i].d = 0x3C00;
        
        if (ws > max_w_scale) max_w_scale = ws;
        if (as > max_a_scale) max_a_scale = as;
        
        for (int j = 0; j < 16; j++) {
            weights[i].qs[j] = rand() & 0xFF;
            w_qs[i * 16 + j] = weights[i].qs[j];
        }
        for (int j = 0; j < 32; j++) {
            activations[i].qs[j] = (rand() % 256) - 128;
            a_qs[i * 32 + j] = activations[i].qs[j];
        }
        
        // Convert to 16-bit fixed point (normalized to max scale)
        w_scales[i] = (int16_t)(ws / max_w_scale * 32767);
        a_scales[i] = (int16_t)(as / max_a_scale * 32767);
    }
    
    float w_scale_factor = max_w_scale / 32767.0f;
    float a_scale_factor = max_a_scale / 32767.0f;
    
    // Warmup
    volatile float result = 0;
    volatile int32_t iresult = 0;
    for (int i = 0; i < 1000; i++) {
        result += vec_dot_standard(K, weights, activations);
        iresult += vec_dot_pure_int(K, w_qs, a_qs);
    }
    
    printf("Q4_0 x Q8_0 dot product benchmark (K=%d, %d iterations)\n\n", K, ITERS);
    
    // Benchmark standard
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERS; i++) {
        result += vec_dot_standard(K, weights, activations);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double standard_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    // Benchmark pure integer (no scales)
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERS; i++) {
        iresult += vec_dot_pure_int(K, w_qs, a_qs);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double pure_int_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    // Benchmark batched
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERS; i++) {
        result += vec_dot_batched4(K, weights, activations);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double batched_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    // Benchmark fixed-point
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERS; i++) {
        result += vec_dot_fixedpoint(K, weights, activations, w_scales, a_scales,
                                     w_scale_factor, a_scale_factor);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double fixedpoint_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    printf("%-20s %8.2f ns/dot, %6.2f GOPS\n", 
           "Standard (baseline):", standard_ns / ITERS, 
           (double)K * 2 * ITERS / standard_ns);
    printf("%-20s %8.2f ns/dot, %6.2f GOPS (%.2fx)\n", 
           "Pure INT (no scale):", pure_int_ns / ITERS,
           (double)K * 2 * ITERS / pure_int_ns,
           standard_ns / pure_int_ns);
    printf("%-20s %8.2f ns/dot, %6.2f GOPS (%.2fx)\n", 
           "Batched x4:", batched_ns / ITERS,
           (double)K * 2 * ITERS / batched_ns,
           standard_ns / batched_ns);
    printf("%-20s %8.2f ns/dot, %6.2f GOPS (%.2fx)\n", 
           "Fixed-point scale:", fixedpoint_ns / ITERS,
           (double)K * 2 * ITERS / fixedpoint_ns,
           standard_ns / fixedpoint_ns);
    
    // Correctness check
    printf("\n=== Correctness check ===\n");
    float std_result = vec_dot_standard(K, weights, activations);
    int32_t int_result = vec_dot_pure_int(K, w_qs, a_qs);
    float bat_result = vec_dot_batched4(K, weights, activations);
    
    printf("Standard:   %.6f\n", std_result);
    printf("Pure INT:   %d (needs scale to compare)\n", int_result);
    printf("Batched:    %.6f (diff: %.6f)\n", bat_result, bat_result - std_result);
    
    // Memory bandwidth analysis
    printf("\n=== Memory analysis ===\n");
    size_t q4_bytes = nb * sizeof(block_q4_0);  // 18 bytes per block
    size_t q8_bytes = nb * sizeof(block_q8_0);  // 34 bytes per block
    size_t pure_int_bytes = nb * 16 + nb * 32;  // Just the quantized values
    
    printf("Standard Q4+Q8: %zu bytes read\n", q4_bytes + q8_bytes);
    printf("Pure INT:       %zu bytes read (%.1f%% of standard)\n", 
           pure_int_bytes, 100.0 * pure_int_bytes / (q4_bytes + q8_bytes));
    
    double bw_standard = (q4_bytes + q8_bytes) * ITERS / (standard_ns / 1e9) / 1e9;
    double bw_pure_int = pure_int_bytes * ITERS / (pure_int_ns / 1e9) / 1e9;
    printf("Standard BW: %.2f GB/s\n", bw_standard);
    printf("Pure INT BW: %.2f GB/s\n", bw_pure_int);
    
    free(weights);
    free(activations);
    free(w_qs);
    free(a_qs);
    free(w_scales);
    free(a_scales);
    
    printf("\n");
    }  // End size loop
    
    return 0;
}
