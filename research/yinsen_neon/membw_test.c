/*
 * membw_test.c - Pure memory bandwidth test for Android
 *
 * Tests sequential read bandwidth to understand device limits.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* Pure sequential read - should hit max bandwidth */
void seq_read_neon(const int8_t* data, int64_t size) {
    int8x16_t acc = vdupq_n_s8(0);
    const int8_t* ptr = data;
    const int8_t* end = data + size;
    
    while (ptr < end) {
        /* Read 64 bytes per iteration */
        int8x16_t v0 = vld1q_s8(ptr);
        int8x16_t v1 = vld1q_s8(ptr + 16);
        int8x16_t v2 = vld1q_s8(ptr + 32);
        int8x16_t v3 = vld1q_s8(ptr + 48);
        ptr += 64;
        
        /* Minimal compute to prevent optimization */
        acc = vaddq_s8(acc, v0);
        acc = vaddq_s8(acc, v1);
        acc = vaddq_s8(acc, v2);
        acc = vaddq_s8(acc, v3);
    }
    
    /* Prevent optimization */
    volatile int8_t sink = vgetq_lane_s8(acc, 0);
    (void)sink;
}

/* Sequential read with prefetch */
void seq_read_neon_prefetch(const int8_t* data, int64_t size) {
    int8x16_t acc = vdupq_n_s8(0);
    const int8_t* ptr = data;
    const int8_t* end = data + size;
    
    while (ptr < end) {
        /* Prefetch ahead */
        __builtin_prefetch(ptr + 512, 0, 3);
        __builtin_prefetch(ptr + 576, 0, 3);
        
        /* Read 64 bytes */
        int8x16_t v0 = vld1q_s8(ptr);
        int8x16_t v1 = vld1q_s8(ptr + 16);
        int8x16_t v2 = vld1q_s8(ptr + 32);
        int8x16_t v3 = vld1q_s8(ptr + 48);
        ptr += 64;
        
        acc = vaddq_s8(acc, v0);
        acc = vaddq_s8(acc, v1);
        acc = vaddq_s8(acc, v2);
        acc = vaddq_s8(acc, v3);
    }
    
    volatile int8_t sink = vgetq_lane_s8(acc, 0);
    (void)sink;
}

/* Read + light SDOT work (simulating actual kernel work/byte ratio) */
void seq_read_sdot(const int8_t* weights, const int8_t* act, int64_t w_size) {
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    const int8_t* w_ptr = weights;
    const int8_t* w_end = weights + w_size;
    
    /* Reuse same activations (they're in cache) */
    int8x16_t a0 = vld1q_s8(act);
    int8x16_t a1 = vld1q_s8(act + 16);
    int8x16_t a2 = vld1q_s8(act + 32);
    int8x16_t a3 = vld1q_s8(act + 48);
    
    while (w_ptr < w_end) {
        __builtin_prefetch(w_ptr + 512, 0, 3);
        
        int8x16_t w0 = vld1q_s8(w_ptr);
        int8x16_t w1 = vld1q_s8(w_ptr + 16);
        int8x16_t w2 = vld1q_s8(w_ptr + 32);
        int8x16_t w3 = vld1q_s8(w_ptr + 48);
        w_ptr += 64;
        
        acc = vdotq_s32(acc, w0, a0);
        acc = vdotq_s32(acc, w1, a1);
        acc = vdotq_s32(acc, w2, a2);
        acc = vdotq_s32(acc, w3, a3);
    }
    
    volatile int32_t sink = vaddvq_s32(acc);
    (void)sink;
#endif
}

int main(int argc, char** argv) {
    int64_t size_mb = 16;  /* Default 16 MB */
    int iters = 10;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            size_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        }
    }
    
    int64_t size = size_mb * 1024 * 1024;
    
    printf("=== Memory Bandwidth Test ===\n");
    printf("Buffer size: %lld MB\n", (long long)size_mb);
    printf("Iterations: %d\n", iters);
    printf("\n");
    
    /* Allocate */
    int8_t* data = (int8_t*)malloc(size);
    int8_t* act = (int8_t*)malloc(64);
    
    if (!data || !act) {
        printf("ERROR: Failed to allocate %lld MB\n", (long long)size_mb);
        return 1;
    }
    
    /* Initialize */
    memset(data, 1, size);
    memset(act, 1, 64);
    
    uint64_t t0, t1;
    double ns, gbps;
    
    /* Warmup */
    seq_read_neon(data, size);
    
    /* Test 1: Pure sequential read */
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        seq_read_neon(data, size);
    }
    t1 = get_time_ns();
    ns = (double)(t1 - t0) / iters;
    gbps = (double)size / ns;
    printf("Sequential read (NEON):        %.1f ms, %.2f GB/s\n", ns / 1e6, gbps);
    
    /* Test 2: Sequential read with prefetch */
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        seq_read_neon_prefetch(data, size);
    }
    t1 = get_time_ns();
    ns = (double)(t1 - t0) / iters;
    gbps = (double)size / ns;
    printf("Sequential read (prefetch):    %.1f ms, %.2f GB/s\n", ns / 1e6, gbps);
    
#if defined(__ARM_FEATURE_DOTPROD)
    /* Test 3: Read with light SDOT compute */
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        seq_read_sdot(data, act, size);
    }
    t1 = get_time_ns();
    ns = (double)(t1 - t0) / iters;
    gbps = (double)size / ns;
    printf("Sequential read + SDOT:        %.1f ms, %.2f GB/s\n", ns / 1e6, gbps);
#endif
    
    printf("\n");
    printf("Theoretical: ~13 GB/s (LPDDR4X)\n");
    
    free(data);
    free(act);
    
    return 0;
}
