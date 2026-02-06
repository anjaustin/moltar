/*
 * Test program for batched GEMV kernel
 * Validates correctness and benchmarks performance.
 */

#define _GNU_SOURCE
#include "batched_gemv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

extern void* memalign(size_t alignment, size_t size);
#define aligned_alloc(align, size) memalign(align, size)

// Reference implementation for validation
static float fp16_to_fp32_ref(uint16_t h) {
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

void reference_gemv(
    size_t n_out, size_t n_in,
    const batched_q4_0_block* weights,
    const int8_t* activations,
    const float* act_scales,
    float* output
) {
    size_t nb = n_in / BATCHED_GEMV_BLOCK_SIZE;
    
    for (size_t row = 0; row < n_out; row++) {
        float sum = 0.0f;
        
        for (size_t b = 0; b < nb; b++) {
            const batched_q4_0_block* blk = &weights[row * nb + b];
            float w_scale = fp16_to_fp32_ref(blk->d);
            
            int32_t dot = 0;
            for (int i = 0; i < 16; i++) {
                int8_t lo = (blk->qs[i] & 0x0F) - 8;
                int8_t hi = (blk->qs[i] >> 4) - 8;
                dot += lo * activations[b * 32 + i];
                dot += hi * activations[b * 32 + 16 + i];
            }
            
            sum += (float)dot * w_scale * act_scales[b];
        }
        
        output[row] = sum;
    }
}

int main(int argc, char** argv) {
    int n_in = 4608;
    int n_out = 1024;
    int iters = 1000;
    
    if (argc > 1) n_in = atoi(argv[1]);
    if (argc > 2) n_out = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);
    
    int nb = n_in / BATCHED_GEMV_BLOCK_SIZE;
    
    printf("Batched GEMV Test\n");
    printf("=================\n");
    printf("Matrix: %d x %d (%d blocks/row), %d iterations\n\n", n_out, n_in, nb, iters);
    
    // Allocate
    batched_q4_0_block* weights = aligned_alloc(64, n_out * nb * sizeof(batched_q4_0_block));
    int8_t* activations = aligned_alloc(64, n_in);
    float* act_scales = aligned_alloc(64, nb * sizeof(float));
    float* combined_scales = aligned_alloc(64, n_out * nb * sizeof(float));
    float* output_ref = aligned_alloc(64, n_out * sizeof(float));
    float* output_batched = aligned_alloc(64, n_out * sizeof(float));
    float* output_inline = aligned_alloc(64, n_out * sizeof(float));
    
    // Initialize with random data
    srand(42);
    for (int i = 0; i < n_out * nb; i++) {
        weights[i].d = 0x3000 + (rand() % 0x400);  // Small positive FP16
        for (int j = 0; j < 16; j++) {
            weights[i].qs[j] = rand() & 0xFF;
        }
    }
    for (int b = 0; b < nb; b++) {
        act_scales[b] = 0.01f + (rand() % 50) * 0.001f;
    }
    for (int i = 0; i < n_in; i++) {
        activations[i] = (rand() % 256) - 128;
    }
    
    // Initialize scales
    batched_gemv_scales_t weight_scales;
    if (batched_gemv_init_scales(&weight_scales, weights, n_out, n_in) != 0) {
        printf("Failed to init scales\n");
        return 1;
    }
    
    // Precompute combined scales
    batched_gemv_combine_scales(&weight_scales, act_scales, combined_scales);
    
    // Compute reference
    reference_gemv(n_out, n_in, weights, activations, act_scales, output_ref);
    
    // Compute batched
    batched_gemv_q4_0(n_out, n_in, weights, activations, combined_scales, output_batched);
    
    // Compute inline
    batched_gemv_q4_0_inline(n_out, n_in, weights, activations, act_scales, output_inline);
    
    // Validate
    float max_err_batched = 0, max_err_inline = 0;
    float max_rel_batched = 0, max_rel_inline = 0;
    
    for (int i = 0; i < n_out; i++) {
        float err_b = fabsf(output_ref[i] - output_batched[i]);
        float err_i = fabsf(output_ref[i] - output_inline[i]);
        float rel_b = (fabsf(output_ref[i]) > 1e-6f) ? err_b / fabsf(output_ref[i]) : 0;
        float rel_i = (fabsf(output_ref[i]) > 1e-6f) ? err_i / fabsf(output_ref[i]) : 0;
        
        if (err_b > max_err_batched) max_err_batched = err_b;
        if (err_i > max_err_inline) max_err_inline = err_i;
        if (rel_b > max_rel_batched) max_rel_batched = rel_b;
        if (rel_i > max_rel_inline) max_rel_inline = rel_i;
    }
    
    printf("Validation:\n");
    printf("  Batched: max abs err = %.2e, max rel err = %.4f%%\n", 
           max_err_batched, max_rel_batched * 100);
    printf("  Inline:  max abs err = %.2e, max rel err = %.4f%%\n\n",
           max_err_inline, max_rel_inline * 100);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        reference_gemv(n_out, n_in, weights, activations, act_scales, output_ref);
        batched_gemv_q4_0(n_out, n_in, weights, activations, combined_scales, output_batched);
    }
    
    // Benchmark
    struct timespec start, end;
    
    // Reference
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        reference_gemv(n_out, n_in, weights, activations, act_scales, output_ref);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double ref_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    
    // Batched with precomputed scales
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        batched_gemv_q4_0(n_out, n_in, weights, activations, combined_scales, output_batched);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double batched_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    
    // Inline scales
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        batched_gemv_q4_0_inline(n_out, n_in, weights, activations, act_scales, output_inline);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double inline_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    
    printf("Performance:\n");
    printf("  Reference:          %8.1f us (1.00x)\n", ref_us);
    printf("  Batched (precomp):  %8.1f us (%.2fx)\n", batched_us, ref_us / batched_us);
    printf("  Batched (inline):   %8.1f us (%.2fx)\n", inline_us, ref_us / inline_us);
    
    printf("\nMemory overhead:\n");
    size_t weight_bytes = n_out * nb * sizeof(batched_q4_0_block);
    size_t scale_bytes = n_out * nb * sizeof(float);
    printf("  Weights: %.2f MB\n", weight_bytes / (1024.0 * 1024.0));
    printf("  Weight scales: %.2f MB (+%.1f%% of weights)\n",
           scale_bytes / (1024.0 * 1024.0),
           100.0 * scale_bytes / weight_bytes);
    printf("  Combined scales: %.2f MB (can reuse weight scales buffer)\n",
           scale_bytes / (1024.0 * 1024.0));
    
    // Cleanup
    batched_gemv_free_scales(&weight_scales);
    free(weights);
    free(activations);
    free(act_scales);
    free(combined_scales);
    free(output_ref);
    free(output_batched);
    free(output_inline);
    
    return 0;
}
