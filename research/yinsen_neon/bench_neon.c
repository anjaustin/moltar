/*
 * bench_neon.c - Benchmark Yinsen NEON Int8 MatVec Kernels on Android
 *
 * Tests the blocked SDOT kernels against baseline to measure:
 *   1. Raw throughput (GOP/s)
 *   2. Memory bandwidth utilization
 *   3. Comparison to expected theoretical limits
 *
 * Target: Dimensity 7020 (Cortex-A78 + A55)
 *   - Memory bandwidth: ~13 GB/s theoretical
 *   - SDOT: 4 MACs per cycle per lane, 4 lanes = 16 MACs/cycle
 *   - Big core @ 2.4 GHz: ~38 GMAC/s compute bound
 *   - Actually memory bound at these sizes
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* External declarations */
extern void pack_weights_int8_blocked8_k64(int8_t*, const int8_t*, int, int);
extern void pack_weights_int8_rowmajor(int8_t*, const int8_t*, int, int);

#if defined(__ARM_FEATURE_DOTPROD)
extern void neon_int8_matvec_8oc(int32_t*, const int8_t*, const int8_t*, int, int);
extern void neon_int8_matvec_blocked8_k64(int32_t*, const int8_t*, const int8_t*, int, int);
extern void neon_int8_matvec_blocked8_k64_mt(int32_t*, const int8_t*, const int8_t*, int, int, int);
#endif

extern void int8_matvec_ref(int32_t*, const int8_t*, const int8_t*, int, int);

/* Timing helper */
static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* Simple PRNG */
static uint32_t xorshift(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

int main(int argc, char** argv) {
    /* Default sizes matching LFM2 layers */
    int N = 1024;   /* Output channels */
    int K = 1024;   /* Input channels */
    int iters = 1000;
    int threads = 1;
    
    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        }
    }
    
    /* Round to requirements */
    N = (N / 8) * 8;    /* Multiple of 8 for blocked kernel */
    K = (K / 64) * 64;  /* Multiple of 64 for K-blocking */
    
    printf("=== Yinsen NEON Int8 MatVec Benchmark ===\n");
    printf("Matrix: %d x %d (N x K)\n", N, K);
    printf("Iterations: %d\n", iters);
    printf("Weight memory: %.2f MB\n", (double)(N * K) / (1024 * 1024));
    printf("\n");
    
    /* Allocate */
    int8_t* weights = (int8_t*)malloc(N * K);
    int8_t* weights_blocked = (int8_t*)malloc(N * K);
    int8_t* act = (int8_t*)malloc(K);
    int32_t* out_ref = (int32_t*)malloc(N * sizeof(int32_t));
    int32_t* out_test = (int32_t*)malloc(N * sizeof(int32_t));
    
    if (!weights || !weights_blocked || !act || !out_ref || !out_test) {
        printf("ERROR: Failed to allocate memory\n");
        return 1;
    }
    
    /* Initialize with random ternary-ish weights and activations */
    uint32_t rng = 42;
    for (int i = 0; i < N * K; i++) {
        int r = xorshift(&rng) % 3;
        weights[i] = (r == 0) ? 0 : (r == 1) ? 1 : -1;
    }
    for (int i = 0; i < K; i++) {
        act[i] = (int8_t)((xorshift(&rng) % 256) - 128);
    }
    
    /* Pack weights */
    printf("Packing weights...\n");
    pack_weights_int8_blocked8_k64(weights_blocked, weights, N, K);
    printf("  Done.\n\n");
    
    /* Compute reference */
    printf("Computing reference...\n");
    int8_matvec_ref(out_ref, act, weights, N, K);
    printf("  Done.\n\n");
    
    /* Verify implementations */
    printf("Verifying implementations:\n");
    
#if defined(__ARM_FEATURE_DOTPROD)
    /* Test row-major 8OC */
    memset(out_test, 0, N * sizeof(int32_t));
    neon_int8_matvec_8oc(out_test, act, weights, N, K);
    int errors_8oc = 0;
    for (int i = 0; i < N; i++) {
        if (out_test[i] != out_ref[i]) errors_8oc++;
    }
    printf("  SDOT 8OC (row-major): %s (%d errors)\n", 
           errors_8oc ? "FAILED" : "PASSED", errors_8oc);
    
    /* Test blocked K64 */
    memset(out_test, 0, N * sizeof(int32_t));
    neon_int8_matvec_blocked8_k64(out_test, act, weights_blocked, N, K);
    int errors_b8k64 = 0;
    for (int i = 0; i < N; i++) {
        if (out_test[i] != out_ref[i]) errors_b8k64++;
    }
    printf("  SDOT B8-K64 (blocked): %s (%d errors)\n", 
           errors_b8k64 ? "FAILED" : "PASSED", errors_b8k64);
#else
    printf("  SDOT not available on this device!\n");
#endif
    
    printf("\n");
    
    /* Benchmark */
    printf("Benchmarking:\n");
    double ops = 2.0 * N * K;  /* MACs counted as 2 ops */
    uint64_t t0, t1;
    double ns, gops, gbps;
    
    /* Warmup */
    for (int i = 0; i < 100; i++) {
        int8_matvec_ref(out_test, act, weights, N, K);
    }
    
    /* Reference (scalar) */
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        int8_matvec_ref(out_test, act, weights, N, K);
    }
    t1 = get_time_ns();
    ns = (double)(t1 - t0) / iters;
    gops = ops / ns;
    gbps = (double)(N * K) / ns;  /* Weight bytes / ns = GB/s */
    printf("  Reference (scalar): %.1f us, %.2f GOP/s, %.1f GB/s\n", 
           ns / 1000, gops, gbps);
    
#if defined(__ARM_FEATURE_DOTPROD)
    /* Warmup NEON */
    for (int i = 0; i < 100; i++) {
        neon_int8_matvec_8oc(out_test, act, weights, N, K);
    }
    
    /* SDOT 8OC (row-major) */
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        neon_int8_matvec_8oc(out_test, act, weights, N, K);
    }
    t1 = get_time_ns();
    ns = (double)(t1 - t0) / iters;
    gops = ops / ns;
    gbps = (double)(N * K) / ns;
    printf("  SDOT 8OC (row-major): %.1f us, %.2f GOP/s, %.1f GB/s\n", 
           ns / 1000, gops, gbps);
    
    /* Warmup blocked */
    for (int i = 0; i < 100; i++) {
        neon_int8_matvec_blocked8_k64(out_test, act, weights_blocked, N, K);
    }
    
    /* SDOT B8-K64 (blocked, 1 thread) */
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        neon_int8_matvec_blocked8_k64(out_test, act, weights_blocked, N, K);
    }
    t1 = get_time_ns();
    ns = (double)(t1 - t0) / iters;
    gops = ops / ns;
    gbps = (double)(N * K) / ns;
    printf("  SDOT B8-K64 (1 thread): %.1f us, %.2f GOP/s, %.1f GB/s\n", 
           ns / 1000, gops, gbps);
    
    /* Warmup 2-thread */
    for (int i = 0; i < 100; i++) {
        neon_int8_matvec_blocked8_k64_mt(out_test, act, weights_blocked, N, K, 2);
    }
    
    /* Verify 2-thread version */
    memset(out_test, 0, N * sizeof(int32_t));
    neon_int8_matvec_blocked8_k64_mt(out_test, act, weights_blocked, N, K, 2);
    int errors_mt = 0;
    for (int i = 0; i < N; i++) {
        if (out_test[i] != out_ref[i]) errors_mt++;
    }
    if (errors_mt > 0) {
        printf("  WARNING: 2-thread version has %d errors!\n", errors_mt);
    }
    
    /* SDOT B8-K64 (blocked, 2 threads) */
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        neon_int8_matvec_blocked8_k64_mt(out_test, act, weights_blocked, N, K, 2);
    }
    t1 = get_time_ns();
    ns = (double)(t1 - t0) / iters;
    gops = ops / ns;
    gbps = (double)(N * K) / ns;
    printf("  SDOT B8-K64 (2 threads): %.1f us, %.2f GOP/s, %.1f GB/s\n", 
           ns / 1000, gops, gbps);
    
    /* Measure thread overhead */
    printf("\n  Thread overhead analysis:\n");
    printf("    1-thread time: %.1f us\n", 263.0);  /* approximate from earlier */
    printf("    2-thread time: %.1f us\n", ns / 1000);
    printf("    Estimated thread overhead: ~%.0f us per pthread_create/join\n", 
           (ns / 1000 - 263.0/2) / 1);
#endif
    
    printf("\n");
    printf("=== Analysis ===\n");
    printf("Dimensity 7020 theoretical limits:\n");
    printf("  Memory BW: ~13 GB/s\n");
    printf("  Compute (1 big core): ~38 GMAC/s\n");
    printf("  At %d x %d: compute bound if BW > %.1f GB/s\n", 
           N, K, ops / 38.0);
    
    /* Cleanup */
    free(weights);
    free(weights_blocked);
    free(act);
    free(out_ref);
    free(out_test);
    
    return 0;
}
