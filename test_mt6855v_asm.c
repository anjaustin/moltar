/*
 * test_mt6855v_asm.c - Test MT6855V Assembly Optimizations
 *
 * Validates the hand-tuned ARM assembly kernels for Motorola MT6855V
 * Measures performance improvement over C implementations
 */

#include "mt6855v_sdot_matvec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

// Test configuration
#define TEST_N          1024     // Must be multiple of 8
#define TEST_K          1024     // Must be multiple of 64
#define TEST_ITERATIONS 100
#define WARMUP_ITERATIONS 10

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static void generate_test_data(int8_t* weights, int8_t* activations, int N, int K) {
    // Generate realistic test data
    for (int i = 0; i < N * K / 8; i++) {  // K/8 for Q4_0 blocks
        weights[i] = (int8_t)((rand() % 16) - 8);  // -8 to +7 range for 4-bit
    }
    
    for (int i = 0; i < K; i++) {
        activations[i] = (int8_t)((rand() % 256) - 128);  // -128 to +127
    }
}

/* ============================================================================
 * Reference Implementation (for comparison)
 * ============================================================================ */

int reference_matvec_int8(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ weights,
    const int8_t* __restrict__ act,
    int N,
    int K
) {
    // Simple reference implementation
    for (int n = 0; n < N; n++) {
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += weights[n * K + k] * act[k];
        }
        out[n] = sum;
    }
    return 0;
}

/* ============================================================================
 * Benchmark Function
 * ============================================================================ */

double benchmark_matvec(
    const char* name,
    int (*matvec_func)(int32_t*, const int8_t*, const int8_t*, int, int),
    int32_t* out,
    const int8_t* weights,
    const int8_t* act,
    int N,
    int K,
    int iterations
) {
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        matvec_func(out, weights, act, N, K);
    }
    
    // Benchmark
    double start_time = get_time_ms();
    
    for (int i = 0; i < iterations; i++) {
        matvec_func(out, weights, act, N, K);
    }
    
    double end_time = get_time_ms();
    double elapsed_ms = end_time - start_time;
    
    // Calculate performance metrics
    double ops = (double)N * K * iterations;  // MAC operations
    double gops = ops / (elapsed_ms / 1000.0) / 1e9;  // GOPS
    double tokens_per_sec = (double)N * iterations / (elapsed_ms / 1000.0);
    double memory_bw_gb = ((double)N * K + N + K) * iterations * 1e-9 / (elapsed_ms / 1000.0);
    
    printf("  %-20s: %6.1f ms, %5.2f GOPS, %5.1f tok/s, %4.1f GB/s\n",
           name, elapsed_ms, gops, tokens_per_sec, memory_bw_gb);
    
    // Store metrics globally
    extern float g_last_tokens_per_sec;
    extern float g_last_memory_bw_gb;
    g_last_tokens_per_sec = tokens_per_sec;
    g_last_memory_bw_gb = memory_bw_gb;
    
    return tokens_per_sec;
}

/* ============================================================================
 * Main Test Function
 * ============================================================================ */

int main(int argc, char* argv[]) {
    printf("MT6855V Assembly Optimization Test\n");
    printf("==================================\n\n");
    
    // Check hardware
    if (!mt6855v_assembly_available()) {
        printf("❌ MT6855V assembly optimizations not available\n");
        printf("   Required: ARMv8.2-a+dotprod hardware\n");
        return 1;
    }
    
    printf("✅ MT6855V hardware detected\n");
    printf("   CPU Core: %d (%s)\n", 
           get_current_cpu_core(),
           get_current_cpu_core() >= 4 ? "Big A78" : "Little A55");
    printf("   Target: 26 tok/s → 35-38 tok/s (+35-46%%)\n\n");
    
    // Allocate test data
    int8_t* weights = (int8_t*)aligned_alloc(32, TEST_N * TEST_K * sizeof(int8_t));
    int8_t* activations = (int8_t*)aligned_alloc(32, TEST_K * sizeof(int8_t));
    int32_t* out = (int32_t*)aligned_alloc(32, TEST_N * sizeof(int32_t));
    int32_t* out_ref = (int32_t*)aligned_alloc(32, TEST_N * sizeof(int32_t));
    
    if (!weights || !activations || !out || !out_ref) {
        printf("❌ Memory allocation failed\n");
        return 1;
    }
    
    // Generate test data
    generate_test_data(weights, activations, TEST_N, TEST_K);
    
    printf("Configuration:\n");
    printf("   N=%d outputs, K=%d inputs\n", TEST_N, TEST_K);
    printf("   Iterations: %d (+%d warmup)\n\n", TEST_ITERATIONS, WARMUP_ITERATIONS);
    
    printf("Benchmark Results:\n");
    printf("Function              : Time     : GOPS  : Tok/s : GB/s\n");
    printf("----------------------:----------:-------:-------:------\n");
    
    // Test reference implementation
    double ref_tok_s = benchmark_matvec("Reference C", reference_matvec_int8,
                                        out_ref, weights, activations,
                                        TEST_N, TEST_K, TEST_ITERATIONS);
    
    // Test assembly implementation
    double asm_tok_s = benchmark_matvec("MT6855V Assembly", neon_int8_matvec_blocked8_k64_asm,
                                      out, weights, activations,
                                      TEST_N, TEST_K, TEST_ITERATIONS);
    
    // Calculate improvement
    double improvement = (asm_tok_s - ref_tok_s) / ref_tok_s * 100.0;
    
    printf("\nPerformance Summary:\n");
    printf("   Assembly vs C: %.1f%% improvement\n", improvement);
    printf("   Current: %.1f tok/s (target: 35-38 tok/s)\n", asm_tok_s);
    
    // Verify correctness
    int errors = 0;
    for (int i = 0; i < TEST_N; i++) {
        if (out[i] != out_ref[i]) {
            errors++;
            if (errors <= 5) {
                printf("❌ Output mismatch at %d: asm=%d, ref=%d\n", i, out[i], out_ref[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("✅ Output verification passed\n");
    } else {
        printf("❌ Output verification failed: %d errors\n", errors);
    }
    
    // Memory bandwidth check
    float memory_bw;
    int core_type;
    mt6855v_get_performance_metrics(NULL, &memory_bw, &core_type);
    
    printf("   Memory bandwidth: %.1f GB/s (target: 11+ GB/s)\n", memory_bw);
    
    // Target achievement check
    if (asm_tok_s >= 35.0) {
        printf("🎯 TARGET ACHIEVED: %.1f tok/s (≥35 tok/s target)\n", asm_tok_s);
    } else if (asm_tok_s >= 30.0) {
        printf("🎯 GOOD PROGRESS: %.1f tok/s (30-35 tok/s range)\n", asm_tok_s);
    } else {
        printf("🔄 NEEDS WORK: %.1f tok/s (<30 tok/s)\n", asm_tok_s);
    }
    
    // Cleanup
    free(weights);
    free(activations);
    free(out);
    free(out_ref);
    
    printf("\n🏁 Test completed\n");
    return (errors == 0 && asm_tok_s >= 30.0) ? 0 : 1;
}