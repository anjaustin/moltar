/*
 * LUT-Based Q4_0 Matrix-Vector Multiply
 *
 * The "frozen 8x8 matvec" insight: precompute all possible products.
 *
 * Q4_0 format:
 *   - Each weight is 4 bits: values 0-15, representing -8 to +7
 *   - Q8_0 activations are 8 bits: -128 to +127
 *   - Product: 4-bit * 8-bit = 12-bit signed
 *
 * Strategy:
 *   For each Q4_0 weight value (0-15), precompute product with all 256 Q8_0 values.
 *   That's 16 * 256 = 4096 entries, each 2 bytes = 8 KB lookup table.
 *
 *   Then: dot product becomes: sum of table[weight[i]][activation[i]]
 *   No multiplies, just table lookups and adds.
 *
 * But wait - the table lookup is probably slower than SDOT on ARM.
 * SDOT does 4 multiplies + accumulate in 1 cycle.
 * Table lookup = load from memory = cache miss risk.
 *
 * Better idea: Use the LUT for the scale multiplication.
 *
 * Current Q4_0 pipeline:
 *   1. SDOT: int32 = sum(q4_weight * q8_activation) for 32 elements
 *   2. Scale: float = int32 * scale_q4 * scale_q8
 *
 * The SDOT is fast. The scale multiply requires float conversion.
 *
 * What if we precompute scale combinations?
 *   - Quantize scales to 8-bit indices
 *   - LUT[scale_q4_idx][scale_q8_idx] = precomputed product
 *   - Then: result = int32 * LUT[idx1][idx2] (still a multiply, but int * int)
 *
 * Actually, the real bottleneck is memory bandwidth, not compute.
 * Let's verify the current state and think more carefully.
 *
 * Created: February 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ============================================================================
 * APPROACH 1: Full Product LUT
 *
 * Precompute w * a for all w in {-8..+7}, a in {-128..+127}
 * Total: 16 * 256 = 4096 entries * 2 bytes = 8 KB
 * ============================================================================ */

static int16_t g_product_lut[16][256];

void init_product_lut(void) {
    for (int w = 0; w < 16; w++) {
        int8_t ws = (int8_t)(w - 8);  // -8 to +7
        for (int a = 0; a < 256; a++) {
            int8_t as = (int8_t)a;  // -128 to +127
            g_product_lut[w][a] = (int16_t)ws * (int16_t)as;
        }
    }
}

/* Dot product using LUT - baseline, no SIMD */
int32_t dot_q4_q8_lut_scalar(const uint8_t* q4, const int8_t* q8, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i += 2) {
        uint8_t byte = q4[i / 2];
        uint8_t w0 = byte & 0xF;
        uint8_t w1 = byte >> 4;
        
        sum += g_product_lut[w0][(uint8_t)q8[i]];
        sum += g_product_lut[w1][(uint8_t)q8[i + 1]];
    }
    return sum;
}

/* ============================================================================
 * APPROACH 2: Ternary Decomposition + LUT
 *
 * Decompose each Q4_0 value into ternary:
 *   -8 = -8*1 = -4*2 = sum of ternary * power-of-2
 *   
 * Each Q4_0 value can be written as: v = sum(trit[b] * 2^b) for b=0..3
 * where trit[b] is {-1, 0, +1}
 *
 * Actually, this doesn't simplify - we still need multiplies.
 * ============================================================================ */

/* ============================================================================
 * APPROACH 3: Additive Decomposition
 *
 * Key insight: any Q4_0 value v can be computed as:
 *   v = a + b where a, b are from a smaller set
 *
 * For example, using base 3 (ternary):
 *   -8 = -9 + 1 = -3*3 + 1  -> trit: (-1, 0, -1) + correction
 *
 * This is getting complicated. Let's try a different angle.
 * ============================================================================ */

/* ============================================================================
 * APPROACH 4: Precomputed Partial Sums (8-element blocks)
 *
 * For 8 consecutive Q4_0 weights (4 bytes = 32 bits):
 *   - There are 2^32 possible weight combinations
 *   - Too many to precompute all dot products
 *
 * But for 4 consecutive weights (2 bytes = 16 bits):
 *   - 2^16 = 65536 possible combinations
 *   - For each 4-activation pattern... still too many
 *
 * Actually, the issue is that activations vary at runtime.
 * We can only precompute things that depend only on weights.
 * ============================================================================ */

/* ============================================================================
 * APPROACH 5: SDOT with Integer-Only Scale Accumulation
 *
 * This is what we tried before - accumulate SDOT results as integers,
 * apply scales at the end. The issue was the two-pass overhead.
 *
 * New idea: Quantize scales to 8-bit and use integer scale multiply.
 *
 * Current: float_result = int32_dot * fp16_scale_w * fp16_scale_a
 * New:     int32_result = int32_dot * int8_scale_w * int8_scale_a >> 16
 *
 * The scales need to be normalized so product fits in int32.
 * ============================================================================ */

/* 8-bit scale lookup table */
static int32_t g_scale_product[256][256];

void init_scale_lut(void) {
    /* Precompute all 256*256 = 64K scale products */
    for (int sw = 0; sw < 256; sw++) {
        for (int sa = 0; sa < 256; sa++) {
            /* Treat as unsigned 8-bit, result fits in 16 bits */
            g_scale_product[sw][sa] = sw * sa;
        }
    }
}

/* ============================================================================
 * BENCHMARK
 * ============================================================================ */

#define N_WEIGHTS 1024  // Match Q4_0 block count (32K weights)
#define ITERATIONS 1000000

int main(int argc, char** argv) {
    printf("LUT-based Q4_0 Matmul Analysis\n");
    printf("================================\n\n");
    
    /* Initialize LUTs */
    init_product_lut();
    init_scale_lut();
    
    printf("Product LUT: %zu KB\n", sizeof(g_product_lut) / 1024);
    printf("Scale LUT:   %zu KB\n", sizeof(g_scale_product) / 1024);
    
    /* Create test data */
    uint8_t* q4 = (uint8_t*)malloc(N_WEIGHTS / 2);  // 4 bits per weight
    int8_t* q8 = (int8_t*)malloc(N_WEIGHTS);
    
    srand(42);
    for (int i = 0; i < N_WEIGHTS / 2; i++) {
        q4[i] = rand() & 0xFF;
    }
    for (int i = 0; i < N_WEIGHTS; i++) {
        q8[i] = (rand() % 256) - 128;
    }
    
    /* Benchmark LUT-based dot product */
    volatile int32_t sink = 0;
    
    clock_t start = clock();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        sink += dot_q4_q8_lut_scalar(q4, q8, N_WEIGHTS);
    }
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double ops_per_sec = (double)ITERATIONS * N_WEIGHTS / elapsed;
    
    printf("\nLUT Scalar Benchmark:\n");
    printf("  %d iterations, %d weights each\n", ITERATIONS, N_WEIGHTS);
    printf("  Time: %.3f s\n", elapsed);
    printf("  Throughput: %.1f M dot-products/s\n", ops_per_sec / 1e6);
    printf("  (sink=%d to prevent optimization)\n", (int)sink);
    
#ifdef __ARM_NEON
    /* Compare with SDOT */
    printf("\nComparing with NEON SDOT...\n");
    
    /* Convert Q4 to Q8 for SDOT (wasteful but fair comparison) */
    int8_t* q8_weights = (int8_t*)malloc(N_WEIGHTS);
    for (int i = 0; i < N_WEIGHTS / 2; i++) {
        q8_weights[i * 2] = (int8_t)((q4[i] & 0xF) - 8);
        q8_weights[i * 2 + 1] = (int8_t)((q4[i] >> 4) - 8);
    }
    
    start = clock();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < N_WEIGHTS; i += 16) {
            int8x16_t vw = vld1q_s8(q8_weights + i);
            int8x16_t va = vld1q_s8(q8 + i);
            acc = vdotq_s32(acc, vw, va);
        }
        sink += vaddvq_s32(acc);
    }
    end = clock();
    
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    ops_per_sec = (double)ITERATIONS * N_WEIGHTS / elapsed;
    
    printf("SDOT Benchmark:\n");
    printf("  Time: %.3f s\n", elapsed);
    printf("  Throughput: %.1f M dot-products/s\n", ops_per_sec / 1e6);
    
    free(q8_weights);
#endif
    
    printf("\n");
    printf("Key Insight:\n");
    printf("============\n");
    printf("LUT-based approach trades compute for memory access.\n");
    printf("On memory-bound systems (like mobile), this is WORSE.\n");
    printf("SDOT is already optimal for the compute - the bottleneck\n");
    printf("is loading weights from DRAM, not computing products.\n");
    printf("\n");
    printf("The 'frozen matvec' idea works for small fixed matrices\n");
    printf("(like Yinsen's 8x8 CfC cells) but not for 350M param LLMs.\n");
    
    free(q4);
    free(q8);
    
    return 0;
}
