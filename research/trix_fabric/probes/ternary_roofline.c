/*
 * ternary_roofline.c — Measure roofline for ternary matmul
 *
 * Tests:
 * 1. Pure memory read (memcpy)
 * 2. Pure DOT compute (fake weights in registers)
 * 3. TBL + DOT (real decode)
 * 4. Full kernel
 *
 * This tells us where the bottleneck actually is.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sched.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

#ifdef __linux__
static void pin_cpu(int cpu) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}
#else
static void pin_cpu(int cpu) { (void)cpu; }
#endif

static const int8_t TRIT_LUT[16] __attribute__((aligned(16))) = {
    0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0
};

int main(int argc, char *argv[]) {
    int N = 4608, K = 1024;
    int iters = 500;
    
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    
    K = (K / 64) * 64;
    N = (N / 8) * 8;
    
    size_t weights_sz = (size_t)N * K / 4;  /* 2-bit packed */
    size_t act_sz = K;
    size_t out_sz = N * sizeof(float);
    
    uint8_t *weights = malloc(weights_sz);
    int8_t *activations = malloc(act_sz);
    float *output = malloc(out_sz);
    
    srand(42);
    for (size_t i = 0; i < weights_sz; i++) weights[i] = rand() & 0xFF;
    for (int i = 0; i < K; i++) activations[i] = (rand() % 256) - 128;
    
    pin_cpu(6);
    
    printf("\n");
    printf("========================================\n");
    printf("  Ternary MatMul Roofline Analysis\n");
    printf("  N=%d, K=%d\n", N, K);
    printf("========================================\n\n");
    
    volatile int64_t sink = 0;
    
#ifdef __ARM_NEON
    int8x16_t lut = vld1q_s8(TRIT_LUT);
    uint8x16_t mask = vdupq_n_u8(0x03);
    
    /* Test 1: Pure memory read */
    printf("  Test 1: Pure memory read (weights only)\n");
    double t0 = now_us();
    for (int iter = 0; iter < iters; iter++) {
        int64_t sum = 0;
        for (size_t i = 0; i < weights_sz; i += 64) {
            uint8x16x4_t w = vld4q_u8(weights + i);
            sum += vaddlvq_u8(w.val[0]) + vaddlvq_u8(w.val[1]);
        }
        sink += sum;
    }
    double mem_time = (now_us() - t0) / iters;
    double mem_bw = weights_sz / (mem_time * 1e3);
    printf("    Time: %.1f us, Bandwidth: %.2f GB/s\n\n", mem_time, mem_bw);
    
    /* Test 2: Pure DOT compute (weights in register, no decode) */
    printf("  Test 2: Pure DOT compute (no memory, no decode)\n");
    int8x16_t fake_w = vdupq_n_s8(1);  /* All +1 */
    t0 = now_us();
    for (int iter = 0; iter < iters; iter++) {
        int32x4_t acc = vdupq_n_s32(0);
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k += 64) {
                int8x16x4_t a = vld4q_s8(activations + (k % K));
                acc = vdotq_s32(acc, fake_w, a.val[0]);
                acc = vdotq_s32(acc, fake_w, a.val[1]);
                acc = vdotq_s32(acc, fake_w, a.val[2]);
                acc = vdotq_s32(acc, fake_w, a.val[3]);
            }
        }
        sink += vaddvq_s32(acc);
    }
    double dot_time = (now_us() - t0) / iters;
    double dot_ops = 2.0 * N * K;
    double dot_gops = dot_ops / (dot_time * 1e3);
    printf("    Time: %.1f us, Throughput: %.2f GOP/s\n\n", dot_time, dot_gops);
    
    /* Test 3: TBL + DOT (decode from memory, but don't accumulate across N) */
    printf("  Test 3: TBL decode + DOT (per-row, no cross-N accumulate)\n");
    t0 = now_us();
    for (int iter = 0; iter < iters; iter++) {
        for (int n = 0; n < N; n++) {
            int32x4_t acc = vdupq_n_s32(0);
            const uint8_t *w_ptr = weights + n * (K / 4);
            const int8_t *a_ptr = activations;
            
            for (int k = 0; k < K; k += 64) {
                uint8x16_t w = vld1q_u8(w_ptr);
                w_ptr += 16;
                int8x16x4_t a = vld4q_s8(a_ptr);
                a_ptr += 64;
                
                acc = vdotq_s32(acc, vqtbl1q_s8(lut, vandq_u8(w, mask)), a.val[0]);
                acc = vdotq_s32(acc, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 2), mask)), a.val[1]);
                acc = vdotq_s32(acc, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 4), mask)), a.val[2]);
                acc = vdotq_s32(acc, vqtbl1q_s8(lut, vshrq_n_u8(w, 6)), a.val[3]);
            }
            sink += vaddvq_s32(acc);
        }
    }
    double tbl_dot_time = (now_us() - t0) / iters;
    double tbl_dot_gops = dot_ops / (tbl_dot_time * 1e3);
    double tbl_dot_bw = weights_sz / (tbl_dot_time * 1e3);
    printf("    Time: %.1f us, Throughput: %.2f GOP/s, BW: %.2f GB/s\n\n", 
           tbl_dot_time, tbl_dot_gops, tbl_dot_bw);
    
    /* Test 4: Full 8OC kernel */
    printf("  Test 4: Full 8OC kernel\n");
    t0 = now_us();
    for (int iter = 0; iter < iters; iter++) {
        for (int n = 0; n < N; n += 8) {
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0);
            int32x4_t acc5 = vdupq_n_s32(0);
            int32x4_t acc6 = vdupq_n_s32(0);
            int32x4_t acc7 = vdupq_n_s32(0);
            
            const int8_t *a_ptr = activations;
            const uint8_t *w0 = weights + (n + 0) * (K / 4);
            const uint8_t *w1 = weights + (n + 1) * (K / 4);
            const uint8_t *w2 = weights + (n + 2) * (K / 4);
            const uint8_t *w3 = weights + (n + 3) * (K / 4);
            const uint8_t *w4 = weights + (n + 4) * (K / 4);
            const uint8_t *w5 = weights + (n + 5) * (K / 4);
            const uint8_t *w6 = weights + (n + 6) * (K / 4);
            const uint8_t *w7 = weights + (n + 7) * (K / 4);
            
            for (int k = 0; k < K; k += 64) {
                int8x16x4_t a = vld4q_s8(a_ptr);
                a_ptr += 64;
                
                #define DO_ROW(ACC, WP) { \
                    uint8x16_t w = vld1q_u8(WP); WP += 16; \
                    ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(w, mask)), a.val[0]); \
                    ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 2), mask)), a.val[1]); \
                    ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 4), mask)), a.val[2]); \
                    ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vshrq_n_u8(w, 6)), a.val[3]); \
                }
                
                DO_ROW(acc0, w0); DO_ROW(acc1, w1); DO_ROW(acc2, w2); DO_ROW(acc3, w3);
                DO_ROW(acc4, w4); DO_ROW(acc5, w5); DO_ROW(acc6, w6); DO_ROW(acc7, w7);
                #undef DO_ROW
            }
            
            output[n+0] = (float)vaddvq_s32(acc0);
            output[n+1] = (float)vaddvq_s32(acc1);
            output[n+2] = (float)vaddvq_s32(acc2);
            output[n+3] = (float)vaddvq_s32(acc3);
            output[n+4] = (float)vaddvq_s32(acc4);
            output[n+5] = (float)vaddvq_s32(acc5);
            output[n+6] = (float)vaddvq_s32(acc6);
            output[n+7] = (float)vaddvq_s32(acc7);
        }
        sink += (int64_t)output[0];
    }
    double full_time = (now_us() - t0) / iters;
    double full_gops = dot_ops / (full_time * 1e3);
    double full_bw = weights_sz / (full_time * 1e3);
    printf("    Time: %.1f us, Throughput: %.2f GOP/s, BW: %.2f GB/s\n\n", 
           full_time, full_gops, full_bw);
    
    /* Summary */
    printf("========================================\n");
    printf("  ROOFLINE SUMMARY\n");
    printf("========================================\n");
    printf("  Memory ceiling:  %.2f GB/s\n", mem_bw);
    printf("  Compute ceiling: %.2f GOP/s\n", dot_gops);
    printf("  TBL+DOT kernel:  %.2f GOP/s (%.1f%% of compute)\n", 
           tbl_dot_gops, 100.0 * tbl_dot_gops / dot_gops);
    printf("  Full 8OC:        %.2f GOP/s (%.1f%% of compute)\n", 
           full_gops, 100.0 * full_gops / dot_gops);
    printf("\n");
    printf("  Arithmetic intensity: %.2f ops/byte\n", dot_ops / weights_sz);
    printf("  Balance point: %.2f GOP/s per GB/s\n", dot_gops / mem_bw);
    printf("\n");
    
    if (full_bw > 0.9 * mem_bw) {
        printf("  STATUS: MEMORY BOUND\n");
        printf("  → Optimization: reduce memory traffic\n");
    } else if (full_gops > 0.9 * dot_gops) {
        printf("  STATUS: COMPUTE BOUND\n");
        printf("  → Optimization: use faster instructions\n");
    } else {
        printf("  STATUS: BALANCED (or inefficient)\n");
        printf("  → Check instruction scheduling\n");
    }
    
    printf("\n  (sink=%ld)\n", (long)sink);
#else
    printf("  NEON not available\n");
#endif
    
    free(weights);
    free(activations);
    free(output);
    return 0;
}
