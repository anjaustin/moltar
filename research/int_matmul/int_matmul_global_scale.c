/*
 * Integer matmul with GLOBAL SCALE (one scale per tensor, not per block)
 * 
 * Current Q4_0: 18 bytes per 32 values (16 bytes quants + 2 bytes FP16 scale)
 * Global scale: 16 bytes per 32 values (just quants) + 1 global FP32 scale
 * 
 * Memory savings: ~11% less data to read
 * Compute savings: No per-block FP32 scale multiply, just final global scale
 * 
 * Accuracy cost: Significant - the whole point of per-block scales is to handle
 *                varying magnitude across the tensor
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

// Standard Q4_0 with per-block scale
typedef struct {
    uint16_t d;
    uint8_t qs[16];
} block_q4_0;

// Global-scale Q4: just packed int4 values, one scale for whole tensor
// 16 bytes per 32 values (vs 18 bytes for block_q4_0)
typedef struct {
    uint8_t qs[16];  // 32 x 4-bit values
} block_q4_global;

// Convert FP16 to FP32
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) { exp = 255; }
    else { exp += 127 - 15; }
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4);
    return f;
}

// Standard Q4_0 dot product (with per-block scales)
__attribute__((noinline))
float dot_q4_standard(int n, const block_q4_0* x, const int8_t* y, float y_scale) {
    int nb = n / 32;
    float32x4_t sumv = vdupq_n_f32(0.0f);
    
    for (int i = 0; i < nb; i++) {
        uint8x16_t q4 = vld1q_u8(x[i].qs);
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(q4, vdupq_n_u8(0x0F)));
        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(q4, 4));
        lo = vsubq_s8(lo, vdupq_n_s8(8));
        hi = vsubq_s8(hi, vdupq_n_s8(8));
        
        int8x16_t y_lo = vld1q_s8(y + i * 32);
        int8x16_t y_hi = vld1q_s8(y + i * 32 + 16);
        
        int32x4_t dot = vdotq_s32(vdupq_n_s32(0), lo, y_lo);
        dot = vdotq_s32(dot, hi, y_hi);
        
        float scale = fp16_to_fp32(x[i].d) * y_scale;
        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(vdupq_n_s32(vaddvq_s32(dot))), scale);
    }
    return vaddvq_f32(sumv);
}

// Global scale Q4 dot product (one scale for whole tensor)
__attribute__((noinline))
float dot_q4_global(int n, const block_q4_global* x, float x_scale, 
                    const int8_t* y, float y_scale) {
    int nb = n / 32;
    int32x4_t sumv = vdupq_n_s32(0);
    
    for (int i = 0; i < nb; i++) {
        uint8x16_t q4 = vld1q_u8(x[i].qs);
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(q4, vdupq_n_u8(0x0F)));
        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(q4, 4));
        lo = vsubq_s8(lo, vdupq_n_s8(8));
        hi = vsubq_s8(hi, vdupq_n_s8(8));
        
        int8x16_t y_lo = vld1q_s8(y + i * 32);
        int8x16_t y_hi = vld1q_s8(y + i * 32 + 16);
        
        sumv = vdotq_s32(sumv, lo, y_lo);
        sumv = vdotq_s32(sumv, hi, y_hi);
    }
    
    // Single scale multiply at the end
    return (float)vaddvq_s32(sumv) * x_scale * y_scale;
}

// Simulate what would happen in a full matmul (N output elements)
void matmul_benchmark(int K, int N, int iters) {
    int nb = K / 32;
    
    // Allocate
    block_q4_0* w_std = aligned_alloc(64, N * nb * sizeof(block_q4_0));
    block_q4_global* w_glb = aligned_alloc(64, N * nb * sizeof(block_q4_global));
    float* w_scales = aligned_alloc(64, N * sizeof(float));
    int8_t* activations = aligned_alloc(64, K * sizeof(int8_t));
    float a_scale = 0.01f;
    float* output = aligned_alloc(64, N * sizeof(float));
    
    // Initialize
    srand(42);
    for (int j = 0; j < N; j++) {
        w_scales[j] = 0.001f + (rand() % 100) * 0.001f;
        for (int i = 0; i < nb; i++) {
            w_std[j * nb + i].d = 0x3C00;  // ~1.0
            for (int k = 0; k < 16; k++) {
                uint8_t val = rand() & 0xFF;
                w_std[j * nb + i].qs[k] = val;
                w_glb[j * nb + i].qs[k] = val;
            }
        }
    }
    for (int i = 0; i < K; i++) {
        activations[i] = (rand() % 256) - 128;
    }
    
    // Warmup
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < N; j++) {
            output[j] = dot_q4_global(K, &w_glb[j * nb], w_scales[j], activations, a_scale);
        }
    }
    
    // Benchmark standard
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < iters; iter++) {
        for (int j = 0; j < N; j++) {
            output[j] = dot_q4_standard(K, &w_std[j * nb], activations, a_scale);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double std_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    // Benchmark global scale
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < iters; iter++) {
        for (int j = 0; j < N; j++) {
            output[j] = dot_q4_global(K, &w_glb[j * nb], w_scales[j], activations, a_scale);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double glb_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    // Memory sizes
    size_t std_bytes = N * nb * sizeof(block_q4_0) + K;  // weights + activations
    size_t glb_bytes = N * nb * sizeof(block_q4_global) + N * sizeof(float) + K;
    
    printf("MatMul K=%d, N=%d (%d iters):\n", K, N, iters);
    printf("  Standard: %.2f ms, %.2f GOPS\n", 
           std_ns / 1e6 / iters, (double)K * N * 2 * iters / std_ns);
    printf("  Global:   %.2f ms, %.2f GOPS (%.2fx)\n",
           glb_ns / 1e6 / iters, (double)K * N * 2 * iters / glb_ns, std_ns / glb_ns);
    printf("  Memory: std=%zu KB, global=%zu KB (%.1f%% savings)\n",
           std_bytes / 1024, glb_bytes / 1024, 
           100.0 * (1.0 - (double)glb_bytes / std_bytes));
    printf("  Effective BW: std=%.2f GB/s, global=%.2f GB/s\n",
           std_bytes * iters / (std_ns / 1e9) / 1e9,
           glb_bytes * iters / (glb_ns / 1e9) / 1e9);
    
    free(w_std);
    free(w_glb);
    free(w_scales);
    free(activations);
    free(output);
}

int main() {
    printf("=== Global Scale Q4 Matmul Benchmark ===\n\n");
    
    // LFM2-350M FFN dimensions: K=1024 (hidden), N=4608 (FFN intermediate)
    matmul_benchmark(1024, 4608, 100);
    printf("\n");
    
    // Also test larger K
    matmul_benchmark(4096, 4608, 50);
    printf("\n");
    
    // Embedding dimension
    matmul_benchmark(1024, 65536, 10);
    
    return 0;
}
