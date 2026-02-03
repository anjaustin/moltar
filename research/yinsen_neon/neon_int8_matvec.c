/*
 * neon_int8_matvec.c - Yinsen NEON Int8 MatVec Kernels for Android
 *
 * Ported from ~/Projects/trix/trix.research/yinsen/neon/neon_ternary.c
 * 
 * These kernels use SDOT (available on Cortex-A78/A55) for int8 matrix-vector
 * multiplication. The key optimizations are:
 *   1. Blocked-8: 8 output channels processed together for cache efficiency
 *   2. K-unrolling: Process 64 K-elements per iteration to reduce loop overhead
 *   3. Prefetching: Software prefetch 2 blocks ahead
 *
 * Target: Dimensity 7020 (Cortex-A78 + A55) with SDOT support
 *
 * Copyright 2026 Trix Research - Ported for Motorola LFM2 project
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ============================================================================
 * WEIGHT PACKING FUNCTIONS
 * ============================================================================ */

/*
 * pack_weights_int8_blocked8_k64 - Pack Int8 weights into Blocked-8 format
 *
 * Block size: 8 rows × 64 cols = 512 bytes = 8 cache lines
 * Layout: Groups of 8 output channels stored together for each 64-element K-block
 *
 * This ensures all 8 weight row loads in the inner loop hit sequential memory.
 */
void pack_weights_int8_blocked8_k64(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,  /* Must be multiple of 8 */
    int K   /* Must be multiple of 64 */
) {
    const int K_blocks = K / 64;
    const int N_blocks = N / 8;
    const int block_stride = 8 * 64;  /* 512 bytes per block */
    
    for (int nb = 0; nb < N_blocks; nb++) {
        for (int kb = 0; kb < K_blocks; kb++) {
            int8_t* dst = packed + (nb * K_blocks + kb) * block_stride;
            
            for (int row = 0; row < 8; row++) {
                int n = nb * 8 + row;
                int k_start = kb * 64;
                
                for (int kk = 0; kk < 64; kk++) {
                    dst[row * 64 + kk] = weights[n * K + k_start + kk];
                }
            }
        }
    }
}

/*
 * pack_weights_int8_rowmajor - Simple row-major copy (baseline)
 */
void pack_weights_int8_rowmajor(
    int8_t* __restrict__ packed,
    const int8_t* __restrict__ weights,
    int N,
    int K
) {
    memcpy(packed, weights, N * K);
}

/* ============================================================================
 * NEON SDOT KERNELS
 * ============================================================================ */

#if defined(__ARM_FEATURE_DOTPROD)

/*
 * neon_int8_matvec_8oc - Basic 8 output channel kernel
 *
 * Process 8 output channels simultaneously to hide SDOT latency (~3 cycles).
 * Uses row-major weight layout.
 */
void neon_int8_matvec_8oc(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  /* Int8 row-major [N][K] */
    int N,  /* Output channels (must be multiple of 8) */
    int K   /* Input channels (must be multiple of 16) */
) {
    for (int n = 0; n < N; n += 8) {
        /* 8 Accumulators */
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        
        /* Row pointers for 8 output channels */
        const int8_t* w0_ptr = wgt + (n + 0) * K;
        const int8_t* w1_ptr = wgt + (n + 1) * K;
        const int8_t* w2_ptr = wgt + (n + 2) * K;
        const int8_t* w3_ptr = wgt + (n + 3) * K;
        const int8_t* w4_ptr = wgt + (n + 4) * K;
        const int8_t* w5_ptr = wgt + (n + 5) * K;
        const int8_t* w6_ptr = wgt + (n + 6) * K;
        const int8_t* w7_ptr = wgt + (n + 7) * K;
        
        /* Process 16 K-elements per iteration */
        for (int k = 0; k < K; k += 16) {
            /* Prefetch ahead */
            if (k + 32 < K) {
                __builtin_prefetch(a_ptr + 32, 0, 3);
                __builtin_prefetch(w0_ptr + 32, 0, 3);
                __builtin_prefetch(w4_ptr + 32, 0, 3);
            }
            
            /* Load 16 activations */
            int8x16_t a = vld1q_s8(a_ptr);
            a_ptr += 16;
            
            /* Load 16 weights for each output channel and SDOT */
            int8x16_t w0 = vld1q_s8(w0_ptr); w0_ptr += 16;
            int8x16_t w1 = vld1q_s8(w1_ptr); w1_ptr += 16;
            int8x16_t w2 = vld1q_s8(w2_ptr); w2_ptr += 16;
            int8x16_t w3 = vld1q_s8(w3_ptr); w3_ptr += 16;
            int8x16_t w4 = vld1q_s8(w4_ptr); w4_ptr += 16;
            int8x16_t w5 = vld1q_s8(w5_ptr); w5_ptr += 16;
            int8x16_t w6 = vld1q_s8(w6_ptr); w6_ptr += 16;
            int8x16_t w7 = vld1q_s8(w7_ptr); w7_ptr += 16;
            
            /* SDOT: 4 dot products of 4 elements each = 16 elements total */
            acc0 = vdotq_s32(acc0, w0, a);
            acc1 = vdotq_s32(acc1, w1, a);
            acc2 = vdotq_s32(acc2, w2, a);
            acc3 = vdotq_s32(acc3, w3, a);
            acc4 = vdotq_s32(acc4, w4, a);
            acc5 = vdotq_s32(acc5, w5, a);
            acc6 = vdotq_s32(acc6, w6, a);
            acc7 = vdotq_s32(acc7, w7, a);
        }
        
        /* Final horizontal reduction and store */
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
    }
}

/*
 * neon_int8_matvec_blocked8_k64 - Cache-optimized blocked kernel
 *
 * Uses Blocked-8 weight layout where 8 rows × 64 cols are stored together.
 * All 8 loads hit sequential memory (512 bytes = 8 cache lines).
 *
 * This is the BEST kernel for memory-bound scenarios.
 */
void neon_int8_matvec_blocked8_k64(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,  /* Blocked-8-K64 format */
    int N,  /* Output channels (must be multiple of 8) */
    int K   /* Input channels (must be multiple of 64) */
) {
    const int K_blocks = K / 64;
    const int block_stride = 8 * 64;  /* 512 bytes per block */
    
    for (int n = 0; n < N; n += 8) {
        int nb = n / 8;
        
        /* 8 accumulators */
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const int8_t* w_base = wgt + nb * K_blocks * block_stride;
        
        for (int kb = 0; kb < K_blocks; kb++) {
            /* Prefetch 2 blocks ahead */
            if (kb + 2 < K_blocks) {
                __builtin_prefetch(a_ptr + 128, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride + 256, 0, 3);
            }
            
            /* Load 64 activations (4 × 16) */
            int8x16_t a0 = vld1q_s8(a_ptr);
            int8x16_t a1 = vld1q_s8(a_ptr + 16);
            int8x16_t a2 = vld1q_s8(a_ptr + 32);
            int8x16_t a3 = vld1q_s8(a_ptr + 48);
            a_ptr += 64;
            
            const int8_t* w_block = w_base + kb * block_stride;
            
            /* Process all 8 rows, 64 elements each */
            #define PROCESS_ROW_K64(ACC, ROW) { \
                int8x16_t w0 = vld1q_s8(w_block + ROW * 64); \
                int8x16_t w1 = vld1q_s8(w_block + ROW * 64 + 16); \
                int8x16_t w2 = vld1q_s8(w_block + ROW * 64 + 32); \
                int8x16_t w3 = vld1q_s8(w_block + ROW * 64 + 48); \
                ACC = vdotq_s32(ACC, w0, a0); \
                ACC = vdotq_s32(ACC, w1, a1); \
                ACC = vdotq_s32(ACC, w2, a2); \
                ACC = vdotq_s32(ACC, w3, a3); \
            }
            
            PROCESS_ROW_K64(acc0, 0);
            PROCESS_ROW_K64(acc1, 1);
            PROCESS_ROW_K64(acc2, 2);
            PROCESS_ROW_K64(acc3, 3);
            PROCESS_ROW_K64(acc4, 4);
            PROCESS_ROW_K64(acc5, 5);
            PROCESS_ROW_K64(acc6, 6);
            PROCESS_ROW_K64(acc7, 7);
            
            #undef PROCESS_ROW_K64
        }
        
        out[n + 0] = vaddvq_s32(acc0);
        out[n + 1] = vaddvq_s32(acc1);
        out[n + 2] = vaddvq_s32(acc2);
        out[n + 3] = vaddvq_s32(acc3);
        out[n + 4] = vaddvq_s32(acc4);
        out[n + 5] = vaddvq_s32(acc5);
        out[n + 6] = vaddvq_s32(acc6);
        out[n + 7] = vaddvq_s32(acc7);
    }
}

/* ============================================================================
 * MULTI-THREADED VERSION
 * ============================================================================ */

#include <pthread.h>

typedef struct {
    int32_t* out;
    const int8_t* act;
    const int8_t* wgt;
    int N;
    int K;
    int n_start;
    int n_end;
} mt_args_t;

static void* matvec_thread_func(void* arg) {
    mt_args_t* a = (mt_args_t*)arg;
    const int K_blocks = a->K / 64;
    const int block_stride = 8 * 64;
    
    for (int n = a->n_start; n < a->n_end; n += 8) {
        int nb = n / 8;
        
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);
        
        const int8_t* a_ptr = a->act;
        const int8_t* w_base = a->wgt + nb * K_blocks * block_stride;
        
        for (int kb = 0; kb < K_blocks; kb++) {
            if (kb + 2 < K_blocks) {
                __builtin_prefetch(a_ptr + 128, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride + 256, 0, 3);
            }
            
            int8x16_t a0 = vld1q_s8(a_ptr);
            int8x16_t a1 = vld1q_s8(a_ptr + 16);
            int8x16_t a2 = vld1q_s8(a_ptr + 32);
            int8x16_t a3 = vld1q_s8(a_ptr + 48);
            a_ptr += 64;
            
            const int8_t* w_block = w_base + kb * block_stride;
            
            #define PROCESS_ROW_K64_MT(ACC, ROW) { \
                int8x16_t w0 = vld1q_s8(w_block + ROW * 64); \
                int8x16_t w1 = vld1q_s8(w_block + ROW * 64 + 16); \
                int8x16_t w2 = vld1q_s8(w_block + ROW * 64 + 32); \
                int8x16_t w3 = vld1q_s8(w_block + ROW * 64 + 48); \
                ACC = vdotq_s32(ACC, w0, a0); \
                ACC = vdotq_s32(ACC, w1, a1); \
                ACC = vdotq_s32(ACC, w2, a2); \
                ACC = vdotq_s32(ACC, w3, a3); \
            }
            
            PROCESS_ROW_K64_MT(acc0, 0);
            PROCESS_ROW_K64_MT(acc1, 1);
            PROCESS_ROW_K64_MT(acc2, 2);
            PROCESS_ROW_K64_MT(acc3, 3);
            PROCESS_ROW_K64_MT(acc4, 4);
            PROCESS_ROW_K64_MT(acc5, 5);
            PROCESS_ROW_K64_MT(acc6, 6);
            PROCESS_ROW_K64_MT(acc7, 7);
            
            #undef PROCESS_ROW_K64_MT
        }
        
        a->out[n + 0] = vaddvq_s32(acc0);
        a->out[n + 1] = vaddvq_s32(acc1);
        a->out[n + 2] = vaddvq_s32(acc2);
        a->out[n + 3] = vaddvq_s32(acc3);
        a->out[n + 4] = vaddvq_s32(acc4);
        a->out[n + 5] = vaddvq_s32(acc5);
        a->out[n + 6] = vaddvq_s32(acc6);
        a->out[n + 7] = vaddvq_s32(acc7);
    }
    return NULL;
}

/*
 * neon_int8_matvec_blocked8_k64_mt - Multi-threaded blocked kernel
 *
 * Splits work across threads by output channels.
 */
void neon_int8_matvec_blocked8_k64_mt(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,
    int N,
    int K,
    int num_threads
) {
    if (num_threads <= 1) {
        neon_int8_matvec_blocked8_k64(out, act, wgt, N, K);
        return;
    }
    
    pthread_t threads[8];
    mt_args_t args[8];
    
    if (num_threads > 8) num_threads = 8;
    
    int rows_per_thread = ((N / 8) / num_threads) * 8;
    if (rows_per_thread < 8) rows_per_thread = 8;
    
    int n_pos = 0;
    int actual_threads = 0;
    
    for (int t = 0; t < num_threads && n_pos < N; t++) {
        args[t].out = out;
        args[t].act = act;
        args[t].wgt = wgt;
        args[t].N = N;
        args[t].K = K;
        args[t].n_start = n_pos;
        
        if (t == num_threads - 1) {
            args[t].n_end = N;
        } else {
            args[t].n_end = n_pos + rows_per_thread;
            if (args[t].n_end > N) args[t].n_end = N;
        }
        
        n_pos = args[t].n_end;
        
        if (t == 0) {
            /* Main thread handles first chunk */
            actual_threads = 1;
        } else {
            pthread_create(&threads[t], NULL, matvec_thread_func, &args[t]);
            actual_threads++;
        }
    }
    
    /* Main thread does its work */
    matvec_thread_func(&args[0]);
    
    /* Join worker threads */
    for (int t = 1; t < actual_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

#endif /* __ARM_FEATURE_DOTPROD */

/* ============================================================================
 * REFERENCE IMPLEMENTATION (for verification)
 * ============================================================================ */

void int8_matvec_ref(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const int8_t* __restrict__ wgt,
    int N,
    int K
) {
    for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++) {
            acc += (int32_t)act[k] * (int32_t)wgt[n * K + k];
        }
        out[n] = acc;
    }
}
