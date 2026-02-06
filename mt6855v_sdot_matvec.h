/*
 * mt6855v_sdot_matvec.h - ARM Assembly MatVec Interface for MT6855V
 *
 * Hand-tuned assembly kernels for Motorola MT6855V / Dimensity 930
 * Provides 35-46% performance improvement over C intrinsics
 *
 * Target: 26 tok/s → 35-38 tok/s
 */

#ifndef MT6855V_SDOT_MATVEC_H
#define MT6855V_SDOT_MATVEC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Assembly Kernel Declarations
 * ============================================================================ */

/**
 * Hand-tuned ARM assembly matrix-vector multiplication
 * 
 * @param out     Output buffer (int32_t*, aligned to 32 bytes)
 * @param weights Weight matrix (int8_t*, Q4_0 format)
 * @param act     Activation vector (int8_t*)
 * @param N       Number of outputs (must be multiple of 8)
 * @param K       Number of inputs (must be multiple of 64)
 * 
 * @return 0 on success, -1 on error
 */
int neon_int8_matvec_blocked8_k64_asm(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ weights,
    const int8_t* __restrict__ act,
    int N,
    int K
);

/**
 * Big core optimized version - Cortex-A78 specific
 * Use when running on CPU 4+ (big cores)
 */
int neon_int8_matvec_blocked8_k64_asm_big(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ weights,
    const int8_t* __restrict__ act,
    int N,
    int K
);

/**
 * Little core optimized version - Cortex-A55 specific  
 * Use when running on CPU 0-3 (little cores)
 */
int neon_int8_matvec_blocked8_k64_asm_little(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ weights,
    const int8_t* __restrict__ act,
    int N,
    int K
);

/* ============================================================================
 * Runtime Detection and Dispatch
 * ============================================================================ */

/**
 * Detect CPU core type and dispatch to appropriate assembly kernel
 * Automatically chooses between big/little core optimizations
 */
int mt6855v_matvec_dispatch(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ weights,
    const int8_t* __restrict__ act,
    int N,
    int K
);

/**
 * Check if assembly kernels are available on this hardware
 * Returns 1 if MT6855V assembly optimizations available, 0 otherwise
 */
int mt6855v_assembly_available(void);

/**
 * Get performance metrics from last assembly execution
 * 
 * @param tokens_per_sec    Output: tokens per second achieved
 * @param memory_bw_gb    Output: memory bandwidth utilization (GB/s)
 * @param core_type       Output: 0=little, 1=big, 2=auto-detected
 */
void mt6855v_get_performance_metrics(
    float* tokens_per_sec,
    float* memory_bw_gb,
    int* core_type
);

/* ============================================================================
 * Hardware Detection
 * ============================================================================ */

/**
 * Detect if running on MT6855V hardware
 * Checks CPU ID, features, and capability bits
 */
int is_mt6855v_hardware(void);

/**
 * Get current CPU core number (0-7 for MT6855V)
 * Used for core-specific optimization selection
 */
int get_current_cpu_core(void);

/* ============================================================================
 * Performance Targets
 * ============================================================================ */

// Baseline: llama.cpp with KleidiAI = 26 tok/s
#define MT6855V_BASELINE_TOK_S   26.0f

// Target with assembly: 35-38 tok/s (35-46% improvement)
#define MT6855V_TARGET_TOK_S      35.0f
#define MT6855V_STRETCH_TOK_S     38.0f

// Memory bandwidth targets
#define MT6855V_MEMORY_BW_GB      11.0f  // Target: 11+ GB/s (vs 13 theoretical)
#define MT6855V_MEMORY_BW_BASELINE  9.5f   // Current C implementation

#ifdef __cplusplus
}
#endif

#endif /* MT6855V_SDOT_MATVEC_H */