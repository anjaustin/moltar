/*
 * Batched GEMV Kernel for Q4_0 weights
 * 
 * Optimized for Arm Cortex-A78 (Dimensity 930)
 * Uses batched SCVTF conversion to hide float latency.
 * 
 * Achieves ~2x speedup over per-block float conversion.
 */

#ifndef BATCHED_GEMV_H
#define BATCHED_GEMV_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BATCHED_GEMV_BLOCK_SIZE 32

// Q4_0 block structure (matches GGUF layout)
typedef struct {
    uint16_t d;        // FP16 scale
    uint8_t qs[16];    // 32 x 4-bit weights packed
} batched_q4_0_block;

// Precomputed scales structure
// For each weight tensor, we store w_scale as float32 (extracted from FP16)
// At inference time, we multiply by activation scales
typedef struct {
    float* scales;      // [n_out * nb] precomputed weight scales (FP32)
    size_t n_out;       // Number of output rows
    size_t n_in;        // Input dimension
    size_t nb;          // Number of blocks per row (n_in / 32)
} batched_gemv_scales_t;

// Initialize scales from Q4_0 weight tensor
// Returns 0 on success, -1 on error
int batched_gemv_init_scales(
    batched_gemv_scales_t* scales,
    const batched_q4_0_block* weights,
    size_t n_out,
    size_t n_in
);

// Free scales
void batched_gemv_free_scales(batched_gemv_scales_t* scales);

// Compute combined scales (weight_scale * act_scale) for a forward pass
// combined_out must have n_out * nb elements
void batched_gemv_combine_scales(
    const batched_gemv_scales_t* weight_scales,
    const float* act_scales,      // [nb] activation scales per block
    float* combined_out           // [n_out * nb] output
);

// Main GEMV kernel with batched-8 conversion
// - weights: Q4_0 weight tensor [n_out, n_in] in block format
// - activations: quantized int8 activations [n_in]
// - combined_scales: precomputed (w_scale * a_scale) [n_out * nb]
// - output: float32 output [n_out]
void batched_gemv_q4_0(
    size_t n_out,
    size_t n_in,
    const batched_q4_0_block* weights,
    const int8_t* activations,
    const float* combined_scales,
    float* output
);

// Variant that takes separate weight and activation scales
// Computes combined scales internally (slower, no precomputation benefit)
void batched_gemv_q4_0_inline(
    size_t n_out,
    size_t n_in,
    const batched_q4_0_block* weights,
    const int8_t* activations,
    const float* act_scales,
    float* output
);

// Multi-threaded version
// thread_id: 0 to n_threads-1
// n_threads: total thread count
void batched_gemv_q4_0_threaded(
    size_t n_out,
    size_t n_in,
    const batched_q4_0_block* weights,
    const int8_t* activations,
    const float* combined_scales,
    float* output,
    int thread_id,
    int n_threads
);

#ifdef __cplusplus
}
#endif

#endif // BATCHED_GEMV_H
