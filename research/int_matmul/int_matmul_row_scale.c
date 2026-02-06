/*
 * Integer matmul with PER-ROW SCALE (one scale per output row, not per block)
 * 
 * This is a middle ground between:
 * - Per-block (Q4_0): accurate but slow (scale multiply per 32 elements)
 * - Global: fast but inaccurate (one scale for entire tensor)
 * 
 * Per-row: One scale per row of the weight matrix
 *          For W[K,N], we have N scales (one per output)
 *          Compute: sum(w[k] * x[k]) for k in [0,K), then multiply by row_scale
 */

#define _GNU_SOURCE
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void* memalign(size_t alignment, size_t size);
#define aligned_alloc(align, size) memalign(align, size)

// Per-row Q4: packed int4 values with one scale per row
// For a row of K elements, we store K/2 bytes of packed int4 + 1 float scale
typedef struct {
    float scale;      // Scale for this entire row
    uint8_t qs[];     // K/2 bytes of packed 4-bit values
} row_q4;

// Standard per-block Q4_0
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
    } else if (exp == 31) { exp = 255; }
    else { exp += 127 - 15; }
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4);
    return f;
}

// Standard Q4_0 dot product (per-block scales)
__attribute__((noinline))
float dot_q4_perblock(int K, const block_q4_0* w, const int8_t* x, float x_scale) {
    int nb = K / 32;
    float32x4_t sumv = vdupq_n_f32(0.0f);
    
    for (int i = 0; i < nb; i++) {
        uint8x16_t q4 = vld1q_u8(w[i].qs);
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(q4, vdupq_n_u8(0x0F)));
        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(q4, 4));
        lo = vsubq_s8(lo, vdupq_n_s8(8));
        hi = vsubq_s8(hi, vdupq_n_s8(8));
        
        int8x16_t x_lo = vld1q_s8(x + i * 32);
        int8x16_t x_hi = vld1q_s8(x + i * 32 + 16);
        
        int32x4_t dot = vdotq_s32(vdupq_n_s32(0), lo, x_lo);
        dot = vdotq_s32(dot, hi, x_hi);
        
        float scale = fp16_to_fp32(w[i].d) * x_scale;
        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(vdupq_n_s32(vaddvq_s32(dot))), scale);
    }
    return vaddvq_f32(sumv);
}

// Per-row scale dot product (single scale for entire row)
__attribute__((noinline))
float dot_q4_perrow(int K, const uint8_t* w_qs, float w_scale, 
                    const int8_t* x, float x_scale) {
    int nb = K / 32;
    int32x4_t sumv = vdupq_n_s32(0);
    
    for (int i = 0; i < nb; i++) {
        uint8x16_t q4 = vld1q_u8(w_qs + i * 16);
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(q4, vdupq_n_u8(0x0F)));
        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(q4, 4));
        lo = vsubq_s8(lo, vdupq_n_s8(8));
        hi = vsubq_s8(hi, vdupq_n_s8(8));
        
        int8x16_t x_lo = vld1q_s8(x + i * 32);
        int8x16_t x_hi = vld1q_s8(x + i * 32 + 16);
        
        sumv = vdotq_s32(sumv, lo, x_lo);
        sumv = vdotq_s32(sumv, hi, x_hi);
    }
    
    // Single scale multiply at the end
    return (float)vaddvq_s32(sumv) * w_scale * x_scale;
}

// Benchmark full matmul: Y = W * X where W is [N, K], X is [K, 1]
void benchmark_matmul(int K, int N, int iters) {
    int nb = K / 32;
    
    // Allocate per-block weights
    block_q4_0* w_perblock = aligned_alloc(64, N * nb * sizeof(block_q4_0));
    
    // Allocate per-row weights (packed + scales)
    uint8_t* w_perrow_qs = aligned_alloc(64, N * (K / 2));  // K/2 bytes per row
    float* w_perrow_scales = aligned_alloc(64, N * sizeof(float));
    
    // Activations (INT8)
    int8_t* x = aligned_alloc(64, K);
    float x_scale = 0.01f;
    
    // Output
    float* y = aligned_alloc(64, N * sizeof(float));
    
    // Initialize
    srand(42);
    for (int j = 0; j < N; j++) {
        w_perrow_scales[j] = 0.001f + (rand() % 100) * 0.001f;
        for (int i = 0; i < nb; i++) {
            w_perblock[j * nb + i].d = 0x3C00;  // ~1.0
            for (int k = 0; k < 16; k++) {
                uint8_t val = rand() & 0xFF;
                w_perblock[j * nb + i].qs[k] = val;
                w_perrow_qs[j * (K/2) + i * 16 + k] = val;
            }
        }
    }
    for (int i = 0; i < K; i++) {
        x[i] = (rand() % 256) - 128;
    }
    
    // Warmup
    for (int iter = 0; iter < 100; iter++) {
        for (int j = 0; j < N; j++) {
            y[j] = dot_q4_perrow(K, &w_perrow_qs[j * (K/2)], w_perrow_scales[j], x, x_scale);
        }
    }
    
    // Benchmark per-block
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < iters; iter++) {
        for (int j = 0; j < N; j++) {
            y[j] = dot_q4_perblock(K, &w_perblock[j * nb], x, x_scale);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double perblock_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    // Benchmark per-row
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < iters; iter++) {
        for (int j = 0; j < N; j++) {
            y[j] = dot_q4_perrow(K, &w_perrow_qs[j * (K/2)], w_perrow_scales[j], x, x_scale);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double perrow_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    // Memory analysis
    size_t perblock_bytes = N * nb * sizeof(block_q4_0);  // 18 bytes per block
    size_t perrow_bytes = N * (K/2) + N * sizeof(float);  // K/2 bytes + 4 bytes scale per row
    
    printf("MatMul K=%d, N=%d (%d iters):\n", K, N, iters);
    printf("  Per-block: %.2f ms, %.2f GOPS, %zu KB\n",
           perblock_ns / 1e6 / iters, (double)K * N * 2 * iters / perblock_ns,
           perblock_bytes / 1024);
    printf("  Per-row:   %.2f ms, %.2f GOPS, %zu KB (%.2fx faster, %.1f%% smaller)\n",
           perrow_ns / 1e6 / iters, (double)K * N * 2 * iters / perrow_ns,
           perrow_bytes / 1024,
           perblock_ns / perrow_ns,
           100.0 * (1.0 - (double)perrow_bytes / perblock_bytes));
    
    // Effective bandwidth
    printf("  BW: perblock=%.2f GB/s, perrow=%.2f GB/s\n",
           (perblock_bytes + K) * iters / (perblock_ns / 1e9) / 1e9,
           (perrow_bytes + K) * iters / (perrow_ns / 1e9) / 1e9);
    
    free(w_perblock);
    free(w_perrow_qs);
    free(w_perrow_scales);
    free(x);
    free(y);
}

int main() {
    printf("=== Per-Row Scale Q4 Matmul Benchmark ===\n\n");
    
    // LFM2-350M FFN: K=1024 (hidden), N=4608 (FFN intermediate)
    benchmark_matmul(1024, 4608, 100);
    printf("\n");
    
    // Embedding: K=1024, N=65536 (vocab)
    benchmark_matmul(1024, 65536, 10);
    printf("\n");
    
    // Larger K (attention)
    benchmark_matmul(4096, 4096, 50);
    
    return 0;
}
