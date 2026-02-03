/*
 * bench_q4_shiftadd.c - Benchmark Q4 shift-add vs multiply
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* External declarations */
extern void q4_matvec_shiftadd_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);

#ifdef __ARM_NEON
extern void q4_matvec_shiftadd_neon(int32_t*, const int8_t*, const uint8_t*, int, int);
#endif

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static uint32_t xorshift(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

int main(int argc, char** argv) {
    int N = 1024;
    int K = 1024;
    int iters = 500;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
    }
    
    /* Round K to multiple of 32 for NEON */
    K = (K / 32) * 32;
    if (K < 32) K = 32;
    
    int K_packed = K / 2;
    
    printf("=== Q4 Shift-Add vs Multiply Benchmark ===\n");
    printf("Matrix: %d x %d (N x K)\n", N, K);
    printf("Iterations: %d\n", iters);
    printf("Weight memory: %.2f MB (Q4 packed)\n", (double)(N * K_packed) / (1024 * 1024));
    printf("\n");
    
    /* Allocate */
    uint8_t* weights = (uint8_t*)malloc(N * K_packed);
    int8_t* act = (int8_t*)malloc(K);
    int32_t* out_ref = (int32_t*)malloc(N * sizeof(int32_t));
    int32_t* out_test = (int32_t*)malloc(N * sizeof(int32_t));
    
    if (!weights || !act || !out_ref || !out_test) {
        printf("ERROR: Allocation failed\n");
        return 1;
    }
    
    /* Initialize */
    uint32_t rng = 42;
    for (int i = 0; i < N * K_packed; i++) {
        weights[i] = xorshift(&rng) & 0xFF;  /* Random Q4 packed */
    }
    for (int i = 0; i < K; i++) {
        act[i] = (int8_t)((xorshift(&rng) % 256) - 128);
    }
    
    /* Compute reference with multiply */
    printf("Computing reference (multiply)...\n");
    q4_matvec_multiply_ref(out_ref, act, weights, N, K);
    
    /* Verify shift-add reference */
    printf("Verifying shift-add reference...\n");
    q4_matvec_shiftadd_ref(out_test, act, weights, N, K);
    int errors_ref = 0;
    for (int i = 0; i < N; i++) {
        if (out_test[i] != out_ref[i]) {
            if (errors_ref < 5) {
                printf("  Mismatch at %d: expected %d, got %d\n", i, out_ref[i], out_test[i]);
            }
            errors_ref++;
        }
    }
    printf("  Shift-add ref: %s (%d errors)\n", errors_ref ? "FAILED" : "PASSED", errors_ref);
    
#ifdef __ARM_NEON
    /* Verify NEON */
    printf("Verifying NEON shift-add...\n");
    memset(out_test, 0, N * sizeof(int32_t));
    q4_matvec_shiftadd_neon(out_test, act, weights, N, K);
    int errors_neon = 0;
    for (int i = 0; i < N; i++) {
        if (out_test[i] != out_ref[i]) {
            if (errors_neon < 5) {
                printf("  Mismatch at %d: expected %d, got %d\n", i, out_ref[i], out_test[i]);
            }
            errors_neon++;
        }
    }
    printf("  NEON shift-add: %s (%d errors)\n", errors_neon ? "FAILED" : "PASSED", errors_neon);
#endif
    
    printf("\nBenchmarking:\n");
    
    uint64_t t0, t1;
    double ns, gops, gbps;
    double ops = 2.0 * N * K;  /* Counting as MACs * 2 */
    
    /* Warmup */
    for (int i = 0; i < 50; i++) {
        q4_matvec_multiply_ref(out_test, act, weights, N, K);
    }
    
    /* Multiply reference */
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        q4_matvec_multiply_ref(out_test, act, weights, N, K);
    }
    t1 = get_time_ns();
    ns = (double)(t1 - t0) / iters;
    gops = ops / ns;
    gbps = (double)(N * K_packed) / ns;
    printf("  Multiply (scalar): %.1f us, %.2f GOP/s, %.1f GB/s\n", ns / 1000, gops, gbps);
    
    /* Shift-add reference */
    for (int i = 0; i < 50; i++) {
        q4_matvec_shiftadd_ref(out_test, act, weights, N, K);
    }
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        q4_matvec_shiftadd_ref(out_test, act, weights, N, K);
    }
    t1 = get_time_ns();
    ns = (double)(t1 - t0) / iters;
    gops = ops / ns;
    gbps = (double)(N * K_packed) / ns;
    printf("  Shift-add (scalar): %.1f us, %.2f GOP/s, %.1f GB/s\n", ns / 1000, gops, gbps);
    
#ifdef __ARM_NEON
    /* NEON shift-add */
    for (int i = 0; i < 50; i++) {
        q4_matvec_shiftadd_neon(out_test, act, weights, N, K);
    }
    t0 = get_time_ns();
    for (int i = 0; i < iters; i++) {
        q4_matvec_shiftadd_neon(out_test, act, weights, N, K);
    }
    t1 = get_time_ns();
    ns = (double)(t1 - t0) / iters;
    gops = ops / ns;
    gbps = (double)(N * K_packed) / ns;
    printf("  Shift-add (NEON): %.1f us, %.2f GOP/s, %.1f GB/s\n", ns / 1000, gops, gbps);
#endif
    
    printf("\n");
    
    free(weights);
    free(act);
    free(out_ref);
    free(out_test);
    
    return 0;
}
