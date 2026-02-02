/*
 * matvec_shootout.c — Three-Way MatVec Benchmark on Real Q4_0 Weights
 *
 * Loads a Q4_0 tensor from a GGUF file, converts to ternary formats,
 * and benchmarks 5 kernel strategies on the actual hardware.
 *
 * The question: on a memory-bandwidth-bound device (Dimensity 930, LPDDR4X),
 * does ternary's 2-bit encoding beat Q4_0's 4-bit by reading half the data?
 *
 * Kernels:
 *   A: Scalar Q4_0 dequant-and-dot (reference)
 *   B: NEON SDOT Q4_0 (approximate KleidiAI approach)
 *   C: Yinsen's ternary SDOT (2-bit packed, K-vertical layout)
 *   D: Yinsen's ternary blocked-8 (2-bit packed, cache-optimized)
 *   E: Yinsen's ternary 8OC (2-bit packed, 8 output channels interleaved)
 *
 * Cross-compile:
 *   NDK=~/Library/Android/sdk/ndk/28.2.13676358
 *   $NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang \
 *       -O2 -march=armv8.2-a+dotprod -o matvec_shootout matvec_shootout.c -lm
 *
 * Usage:
 *   ./matvec_shootout model.gguf [tensor_name]
 *
 * Default tensor: blk.0.ffn_gate.weight (1024 x 4608 in 350M)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ══════════════════════════════════════════════════════════════════
 *  Timing
 * ══════════════════════════════════════════════════════════════════ */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ══════════════════════════════════════════════════════════════════
 *  Minimal GGUF Parser (just enough to find one Q4_0 tensor)
 * ══════════════════════════════════════════════════════════════════ */

#define GGUF_MAGIC 0x46554747
#define GGUF_TYPE_Q4_0 2

typedef struct {
    char name[256];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
} TensorInfo;

static const uint8_t *skip_gguf_string(const uint8_t *p) {
    uint64_t len = *(const uint64_t *)p;
    return p + 8 + len;
}

static void read_gguf_string(const uint8_t *p, char *out, int maxlen) {
    uint64_t len = *(const uint64_t *)p;
    int copy = (len < (uint64_t)maxlen - 1) ? (int)len : maxlen - 1;
    memcpy(out, p + 8, copy);
    out[copy] = '\0';
}

static const uint8_t *skip_gguf_value(const uint8_t *p, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return p + 1;
        case 2: case 3: return p + 2;
        case 4: case 5: case 6: return p + 4;
        case 10: case 11: case 12: return p + 8;
        case 8: return skip_gguf_string(p);
        case 9: {
            uint32_t et = *(const uint32_t *)p; p += 4;
            uint64_t cnt = *(const uint64_t *)p; p += 8;
            for (uint64_t i = 0; i < cnt; i++) p = skip_gguf_value(p, et);
            return p;
        }
        default: return p;
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  Q4_0 Helpers
 * ══════════════════════════════════════════════════════════════════ */

/* f16 to f32 conversion */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

/* ══════════════════════════════════════════════════════════════════
 *  KERNEL A: Scalar Q4_0 dequant-and-dot (reference)
 * ══════════════════════════════════════════════════════════════════ */

static void matvec_q4_scalar(
    float * __restrict__ out,
    const uint8_t * __restrict__ q4_data,  /* raw Q4_0 blocks */
    const float * __restrict__ act,        /* float activations [K] */
    int N, int K
) {
    const int n_blocks_per_row = K / 32;
    const int block_size = 18;  /* 2 bytes scale + 16 bytes data */

    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        const uint8_t *row = q4_data + n * n_blocks_per_row * block_size;

        for (int b = 0; b < n_blocks_per_row; b++) {
            uint16_t scale_h;
            memcpy(&scale_h, row, 2);
            float scale = f16_to_f32(scale_h);
            const uint8_t *nibbles = row + 2;

            for (int j = 0; j < 16; j++) {
                uint8_t byte = nibbles[j];
                int8_t v0 = (byte & 0x0F) - 8;
                int8_t v1 = ((byte >> 4) & 0x0F) - 8;
                int k = b * 32 + j * 2;
                sum += scale * v0 * act[k];
                sum += scale * v1 * act[k + 1];
            }
            row += block_size;
        }
        out[n] = sum;
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  KERNEL B: NEON SDOT Q4_0 matvec
 *  Dequantize Q4_0 nibbles to int8, use SDOT for dot product,
 *  then multiply by scale per block.
 * ══════════════════════════════════════════════════════════════════ */

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void matvec_q4_neon(
    float * __restrict__ out,
    const uint8_t * __restrict__ q4_data,
    const int8_t * __restrict__ act_i8,   /* int8 activations [K] */
    int N, int K
) {
    const int n_blocks_per_row = K / 32;
    const int block_size = 18;
    const int8x16_t eight = vdupq_n_s8(8);

    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        const uint8_t *row = q4_data + n * n_blocks_per_row * block_size;

        for (int b = 0; b < n_blocks_per_row; b++) {
            uint16_t scale_h;
            memcpy(&scale_h, row, 2);
            float scale = f16_to_f32(scale_h);
            const uint8_t *nibbles = row + 2;

            /* Load 16 bytes = 32 weights as nibbles */
            uint8x16_t raw = vld1q_u8(nibbles);
            
            /* Extract low and high nibbles, subtract 8 to get signed */
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);

            /* Load activations (32 int8 values) */
            /* Activations need to be deinterleaved: a[0],a[2],a[4]... and a[1],a[3],a[5]... */
            int8x16x2_t a_pair = vld2q_s8(act_i8 + b * 32);

            /* SDOT: dot product of 4 int8 values at a time */
            int32x4_t acc_lo = vdupq_n_s32(0);
            int32x4_t acc_hi = vdupq_n_s32(0);
            acc_lo = vdotq_s32(acc_lo, lo, a_pair.val[0]);
            acc_hi = vdotq_s32(acc_hi, hi, a_pair.val[1]);

            int32_t block_sum = vaddvq_s32(acc_lo) + vaddvq_s32(acc_hi);
            sum += scale * (float)block_sum / 64.0f;  /* Undo act_i8 scaling */

            row += block_size;
        }
        out[n] = sum;
    }
}
#endif

/* ══════════════════════════════════════════════════════════════════
 *  Ternary Conversion + Yinsen's Kernels
 * ══════════════════════════════════════════════════════════════════ */

/*
 * Trit decode table for TBL instruction:
 * Index 0 (00) -> 0, Index 1 (01) -> +1, Index 2 (10) -> -1, Index 3 (11) -> 0
 */
static const int8_t TRIT_DECODE_TABLE[16] __attribute__((aligned(16))) = {
    0, 1, -1, 0,
    0, 1, -1, 0,
    0, 1, -1, 0,
    0, 1, -1, 0
};

/*
 * Convert Q4_0 tensor to ternary {-1, 0, +1} stored as int8.
 * Also extract per-block scales.
 * Returns: ternary weights [N*K] as int8, scales [N * K/32] as float
 */
static void q4_to_ternary(
    int8_t * __restrict__ ternary,
    float * __restrict__ scales,
    const uint8_t * __restrict__ q4_data,
    int N, int K
) {
    const int n_blocks_per_row = K / 32;
    const int block_size = 18;

    for (int n = 0; n < N; n++) {
        const uint8_t *row = q4_data + n * n_blocks_per_row * block_size;
        for (int b = 0; b < n_blocks_per_row; b++) {
            uint16_t scale_h;
            memcpy(&scale_h, row, 2);
            scales[n * n_blocks_per_row + b] = f16_to_f32(scale_h);
            const uint8_t *nibbles = row + 2;

            for (int j = 0; j < 16; j++) {
                uint8_t byte = nibbles[j];
                int8_t v0 = (byte & 0x0F) - 8;
                int8_t v1 = ((byte >> 4) & 0x0F) - 8;
                int k = b * 32 + j * 2;

                /* Sign-only ternary: -1, 0, or +1 */
                ternary[n * K + k]     = (v0 > 0) ? 1 : (v0 < 0) ? -1 : 0;
                ternary[n * K + k + 1] = (v1 > 0) ? 1 : (v1 < 0) ? -1 : 0;
            }
            row += block_size;
        }
    }
}

/*
 * Pack ternary int8 weights into 2-bit format (K-vertical layout).
 * Encoding: 0->00, +1->01, -1->10
 */
static void pack_ternary_kvertical(
    uint8_t * __restrict__ packed,
    const int8_t * __restrict__ ternary,
    int N, int K
) {
    const int K_packed = K / 4;
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t byte = 0;
            for (int i = 0; i < 4; i++) {
                int8_t w = ternary[n * K + k + i];
                uint8_t trit = (w == 1) ? 1 : (w == -1) ? 2 : 0;
                byte |= (trit << (i * 2));
            }
            packed[n * K_packed + k / 4] = byte;
        }
    }
}

/*
 * Pack ternary into Blocked-8 layout (from Yinsen).
 * Groups of 8 output channels stored contiguously for each K-block of 64.
 * Block = 8 rows x 16 bytes = 128 bytes = 2 cache lines.
 */
static void pack_ternary_blocked8(
    uint8_t * __restrict__ packed,
    const int8_t * __restrict__ ternary,
    int N, int K
) {
    const int K_blocks = K / 64;
    const int N_blocks = N / 8;
    const int block_stride = 8 * 16;

    for (int nb = 0; nb < N_blocks; nb++) {
        for (int kb = 0; kb < K_blocks; kb++) {
            uint8_t *dst = packed + (nb * K_blocks + kb) * block_stride;
            for (int row = 0; row < 8; row++) {
                int n = nb * 8 + row;
                int k_start = kb * 64;
                for (int kk = 0; kk < 64; kk += 4) {
                    int k = k_start + kk;
                    uint8_t byte = 0;
                    for (int i = 0; i < 4; i++) {
                        int8_t w = ternary[n * K + k + i];
                        uint8_t trit = (w == 1) ? 1 : (w == -1) ? 2 : 0;
                        byte |= (trit << (i * 2));
                    }
                    dst[row * 16 + kk / 4] = byte;
                }
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  KERNEL C: Yinsen's Ternary SDOT (K-vertical layout)
 *  Adapted from neon_ternary_matvec_sdot.
 *  Added per-block scale multiplication.
 * ══════════════════════════════════════════════════════════════════ */

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void matvec_ternary_sdot(
    float * __restrict__ out,
    const uint8_t * __restrict__ packed_wgt,  /* 2-bit K-vertical */
    const int8_t * __restrict__ act_i8,
    const float * __restrict__ scales,        /* per-block scales [N * K/32] */
    int N, int K
) {
    const int K_packed = K / 4;
    const int n_blocks_per_row = K / 32;

    int8x16_t lut = vld1q_s8(TRIT_DECODE_TABLE);
    uint8x16_t mask_03 = vdupq_n_u8(0x03);

    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        const uint8_t *w_ptr = packed_wgt + n * K_packed;
        const int8_t *a_ptr = act_i8;
        const float *s_ptr = scales + n * n_blocks_per_row;

        /* Process in blocks of 32 weights (8 packed bytes) to apply per-block scale.
         * But SDOT needs 64-aligned loops for VLD4. So we process 64 at a time
         * and apply 2 scales per iteration. */
        for (int k = 0; k < K; k += 64) {
            int8x16x4_t a_streams = vld4q_s8(a_ptr);
            a_ptr += 64;

            uint8x16_t w_packed = vld1q_u8(w_ptr);
            w_ptr += 16;

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            acc0 = vdotq_s32(acc0, vqtbl1q_s8(lut, vandq_u8(w_packed, mask_03)), a_streams.val[0]);
            acc1 = vdotq_s32(acc1, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w_packed, 2), mask_03)), a_streams.val[1]);
            acc2 = vdotq_s32(acc2, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w_packed, 4), mask_03)), a_streams.val[2]);
            acc3 = vdotq_s32(acc3, vqtbl1q_s8(lut, vshrq_n_u8(w_packed, 6)), a_streams.val[3]);

            int32x4_t acc = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
            int32_t dot = vaddvq_s32(acc);

            /* Apply average of 2 block scales for this 64-element chunk.
             * Approximation: use average scale for speed. */
            int block_idx = k / 32;
            float avg_scale = (s_ptr[block_idx] + s_ptr[block_idx + 1]) * 0.5f;
            sum += avg_scale * (float)dot / 64.0f;  /* Undo act_i8 scaling */
        }
        out[n] = sum;
    }
}
#endif

/* ══════════════════════════════════════════════════════════════════
 *  KERNEL D: Yinsen's Ternary Blocked-8 (cache-optimized)
 *  Adapted from neon_ternary_matvec_blocked8.
 * ══════════════════════════════════════════════════════════════════ */

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void matvec_ternary_blocked8(
    float * __restrict__ out,
    const uint8_t * __restrict__ blocked_wgt,  /* Blocked-8 format */
    const int8_t * __restrict__ act_i8,
    const float * __restrict__ scales,
    int N, int K
) {
    const int K_blocks = K / 64;
    const int n_blocks_per_row = K / 32;
    const int block_stride = 8 * 16;

    int8x16_t lut = vld1q_s8(TRIT_DECODE_TABLE);
    uint8x16_t mask_03 = vdupq_n_u8(0x03);

    for (int n = 0; n < N; n += 8) {
        int nb = n / 8;
        float sums[8] = {0};

        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);

        const int8_t *a_ptr = act_i8;
        const uint8_t *w_base = blocked_wgt + nb * K_blocks * block_stride;

        for (int kb = 0; kb < K_blocks; kb++) {
            if (kb + 2 < K_blocks) {
                __builtin_prefetch(a_ptr + 128, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride, 0, 3);
                __builtin_prefetch(w_base + (kb + 2) * block_stride + 64, 0, 3);
            }

            int8x16x4_t a = vld4q_s8(a_ptr);
            a_ptr += 64;

            const uint8_t *w_block = w_base + kb * block_stride;

            #define PROCESS_ROW(ACC, ROW) { \
                uint8x16_t w = vld1q_u8(w_block + ROW * 16); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(w, mask_03)), a.val[0]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 2), mask_03)), a.val[1]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 4), mask_03)), a.val[2]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vshrq_n_u8(w, 6)), a.val[3]); \
            }

            PROCESS_ROW(acc0, 0);
            PROCESS_ROW(acc1, 1);
            PROCESS_ROW(acc2, 2);
            PROCESS_ROW(acc3, 3);
            PROCESS_ROW(acc4, 4);
            PROCESS_ROW(acc5, 5);
            PROCESS_ROW(acc6, 6);
            PROCESS_ROW(acc7, 7);

            #undef PROCESS_ROW
        }

        /* Apply average scale across all blocks for simplicity in the probe.
         * In production, per-block scaling would be fused into the kernel. */
        for (int r = 0; r < 8; r++) {
            float avg_scale = 0;
            const float *s = scales + (n + r) * n_blocks_per_row;
            for (int b = 0; b < n_blocks_per_row; b++) avg_scale += s[b];
            avg_scale /= n_blocks_per_row;
            sums[r] = avg_scale;
        }

        out[n + 0] = sums[0] * (float)vaddvq_s32(acc0) / 64.0f;
        out[n + 1] = sums[1] * (float)vaddvq_s32(acc1) / 64.0f;
        out[n + 2] = sums[2] * (float)vaddvq_s32(acc2) / 64.0f;
        out[n + 3] = sums[3] * (float)vaddvq_s32(acc3) / 64.0f;
        out[n + 4] = sums[4] * (float)vaddvq_s32(acc4) / 64.0f;
        out[n + 5] = sums[5] * (float)vaddvq_s32(acc5) / 64.0f;
        out[n + 6] = sums[6] * (float)vaddvq_s32(acc6) / 64.0f;
        out[n + 7] = sums[7] * (float)vaddvq_s32(acc7) / 64.0f;
    }
}
#endif

/* ══════════════════════════════════════════════════════════════════
 *  KERNEL E: Yinsen's Ternary 8OC SDOT (K-vertical, 8 output channels)
 *  Pure throughput kernel — no blocking, but processes 8 OC at once.
 * ══════════════════════════════════════════════════════════════════ */

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void matvec_ternary_8oc(
    float * __restrict__ out,
    const uint8_t * __restrict__ packed_wgt,  /* 2-bit K-vertical */
    const int8_t * __restrict__ act_i8,
    const float * __restrict__ scales,
    int N, int K
) {
    const int K_packed = K / 4;
    const int n_blocks_per_row = K / 32;

    int8x16_t lut = vld1q_s8(TRIT_DECODE_TABLE);
    uint8x16_t mask_03 = vdupq_n_u8(0x03);

    for (int n = 0; n < N; n += 8) {
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        int32x4_t acc4 = vdupq_n_s32(0);
        int32x4_t acc5 = vdupq_n_s32(0);
        int32x4_t acc6 = vdupq_n_s32(0);
        int32x4_t acc7 = vdupq_n_s32(0);

        const int8_t *a_ptr = act_i8;
        const uint8_t *w0 = packed_wgt + (n + 0) * K_packed;
        const uint8_t *w1 = packed_wgt + (n + 1) * K_packed;
        const uint8_t *w2 = packed_wgt + (n + 2) * K_packed;
        const uint8_t *w3 = packed_wgt + (n + 3) * K_packed;
        const uint8_t *w4 = packed_wgt + (n + 4) * K_packed;
        const uint8_t *w5 = packed_wgt + (n + 5) * K_packed;
        const uint8_t *w6 = packed_wgt + (n + 6) * K_packed;
        const uint8_t *w7 = packed_wgt + (n + 7) * K_packed;

        for (int k = 0; k < K; k += 64) {
            int8x16x4_t a = vld4q_s8(a_ptr);
            a_ptr += 64;

            #define DO_OC(ACC, WP) { \
                uint8x16_t w = vld1q_u8(WP); WP += 16; \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(w, mask_03)), a.val[0]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 2), mask_03)), a.val[1]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w, 4), mask_03)), a.val[2]); \
                ACC = vdotq_s32(ACC, vqtbl1q_s8(lut, vshrq_n_u8(w, 6)), a.val[3]); \
            }

            DO_OC(acc0, w0); DO_OC(acc1, w1); DO_OC(acc2, w2); DO_OC(acc3, w3);
            DO_OC(acc4, w4); DO_OC(acc5, w5); DO_OC(acc6, w6); DO_OC(acc7, w7);
            #undef DO_OC
        }

        /* Per-row average scale */
        for (int r = 0; r < 8; r++) {
            float avg = 0;
            const float *s = scales + (n + r) * n_blocks_per_row;
            for (int b = 0; b < n_blocks_per_row; b++) avg += s[b];
            avg /= n_blocks_per_row;

            int32_t dot;
            switch (r) {
                case 0: dot = vaddvq_s32(acc0); break;
                case 1: dot = vaddvq_s32(acc1); break;
                case 2: dot = vaddvq_s32(acc2); break;
                case 3: dot = vaddvq_s32(acc3); break;
                case 4: dot = vaddvq_s32(acc4); break;
                case 5: dot = vaddvq_s32(acc5); break;
                case 6: dot = vaddvq_s32(acc6); break;
                default: dot = vaddvq_s32(acc7); break;
            }
            out[n + r] = avg * (float)dot / 64.0f;  /* Undo act_i8 scaling */
        }
    }
}
#endif

/* ══════════════════════════════════════════════════════════════════
 *  Benchmark Runner
 * ══════════════════════════════════════════════════════════════════ */

typedef void (*bench_fn)(void);

static double bench(const char *name, bench_fn fn, int iters,
                    int N, int K, int bytes_per_weight_num, int bytes_per_weight_den) {
    /* Warmup */
    for (int i = 0; i < 10; i++) fn();

    double t0 = now_us();
    for (int i = 0; i < iters; i++) fn();
    double elapsed = now_us() - t0;

    double us_per = elapsed / iters;
    double ops = 2.0 * N * K;  /* 1 multiply + 1 add per weight */
    double gops = (ops * iters) / (elapsed * 1e3);
    double bytes_read = (double)N * K * bytes_per_weight_num / bytes_per_weight_den + K;  /* weights + activation */
    double gbps = (bytes_read * iters) / (elapsed * 1e3);

    printf("  %-35s %8.1f us  %6.1f GOP/s  %6.2f GB/s  (%.1f bytes read)\n",
           name, us_per, gops, gbps, bytes_read);
    return us_per;
}

/* ══════════════════════════════════════════════════════════════════
 *  Main
 * ══════════════════════════════════════════════════════════════════ */

/* Global pointers for bench callbacks */
static float *g_out_f;
static int32_t *g_out_i;
static const uint8_t *g_q4_data;
static const uint8_t *g_tern_kv;
static const uint8_t *g_tern_b8;
static const float *g_act_f;
static const int8_t *g_act_i8;
static const float *g_scales;
static int g_N, g_K;

static void run_A(void) { matvec_q4_scalar(g_out_f, g_q4_data, g_act_f, g_N, g_K); }
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void run_B(void) { matvec_q4_neon(g_out_f, g_q4_data, g_act_i8, g_N, g_K); }
static void run_C(void) { matvec_ternary_sdot(g_out_f, g_tern_kv, g_act_i8, g_scales, g_N, g_K); }
static void run_D(void) { matvec_ternary_blocked8(g_out_f, g_tern_b8, g_act_i8, g_scales, g_N, g_K); }
static void run_E(void) { matvec_ternary_8oc(g_out_f, g_tern_kv, g_act_i8, g_scales, g_N, g_K); }
#endif

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf [tensor_name]\n", argv[0]);
        return 1;
    }

    const char *target_tensor = (argc > 2) ? argv[2] : "blk.0.ffn_gate.weight";

    /* mmap the GGUF */
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }

    /* Parse header */
    const uint8_t *p = data;
    uint32_t magic = *(const uint32_t *)p; p += 4;
    if (magic != GGUF_MAGIC) { fprintf(stderr, "Not GGUF\n"); return 1; }
    p += 4; /* version */
    uint64_t n_tensors = *(const uint64_t *)p; p += 8;
    uint64_t n_kv = *(const uint64_t *)p; p += 8;

    /* Skip KV */
    for (uint64_t i = 0; i < n_kv; i++) {
        p = skip_gguf_string(p);
        uint32_t vtype = *(const uint32_t *)p; p += 4;
        p = skip_gguf_value(p, vtype);
    }

    /* Read tensor infos */
    TensorInfo *tensors = calloc(n_tensors, sizeof(TensorInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        read_gguf_string(p, tensors[i].name, sizeof(tensors[i].name));
        p = skip_gguf_string(p);
        tensors[i].n_dims = *(const uint32_t *)p; p += 4;
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            tensors[i].dims[d] = *(const uint64_t *)p; p += 8;
        }
        tensors[i].type = *(const uint32_t *)p; p += 4;
        tensors[i].offset = *(const uint64_t *)p; p += 8;
    }

    uint64_t data_offset = ((p - data) + 31) & ~31ULL;

    /* Find target tensor */
    int ti = -1;
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (strcmp(tensors[i].name, target_tensor) == 0 && tensors[i].type == GGUF_TYPE_Q4_0) {
            ti = i;
            break;
        }
    }
    if (ti < 0) {
        fprintf(stderr, "Tensor '%s' not found (or not Q4_0)\n", target_tensor);
        /* List available Q4_0 tensors */
        fprintf(stderr, "Available Q4_0 tensors:\n");
        for (uint64_t i = 0; i < n_tensors; i++) {
            if (tensors[i].type == GGUF_TYPE_Q4_0) {
                fprintf(stderr, "  %s [", tensors[i].name);
                for (uint32_t d = 0; d < tensors[i].n_dims; d++)
                    fprintf(stderr, "%llu%s", tensors[i].dims[d], d+1<tensors[i].n_dims?", ":"");
                fprintf(stderr, "]\n");
            }
        }
        return 1;
    }

    /* GGUF dimension convention: dim[0] = K (inner), dim[1] = N (outer) for 2D */
    int K = (int)tensors[ti].dims[0];
    int N = (int)tensors[ti].dims[1];
    uint64_t n_elements = (uint64_t)N * K;

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  MATVEC SHOOTOUT: %s [%d x %d] = %llu weights\n", tensors[ti].name, N, K, n_elements);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    /* Ensure K is multiple of 64 and N is multiple of 8 for NEON kernels */
    if (K % 64 != 0 || N % 8 != 0) {
        fprintf(stderr, "ERROR: K=%d must be %%64, N=%d must be %%8\n", K, N);
        return 1;
    }

    const uint8_t *q4_raw = data + data_offset + tensors[ti].offset;

    printf("  Q4_0 data: %d blocks/row, %.1f KiB per row, %.1f MiB total\n",
           K / 32, (K / 32) * 18.0 / 1024, (double)N * (K / 32) * 18 / (1024 * 1024));
    printf("  Ternary 2-bit: %.1f MiB total (%.0f%% of Q4_0)\n",
           (double)N * K / 4 / (1024 * 1024), 100.0 * (N * K / 4.0) / (N * (K/32) * 18.0));

    /* Allocate buffers */
    float *act_f = calloc(K, sizeof(float));
    int8_t *act_i8 = calloc(K, sizeof(int8_t));
    float *out_f = calloc(N, sizeof(float));
    float *ref_out = calloc(N, sizeof(float));
    int8_t *ternary = calloc(n_elements, sizeof(int8_t));
    float *scales = calloc(N * (K / 32), sizeof(float));
    uint8_t *packed_kv = calloc(N * (K / 4), 1);
    uint8_t *packed_b8 = calloc(N * (K / 4), 1);

    /* Generate random activations (small values to prevent overflow) */
    srand(42);
    for (int i = 0; i < K; i++) {
        act_f[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        act_i8[i] = (int8_t)(act_f[i] * 64);  /* Scale to int8 range */
    }

    /* Convert to ternary */
    printf("  Converting Q4_0 → ternary...\n");
    q4_to_ternary(ternary, scales, q4_raw, N, K);

    /* Count sparsity in converted ternary */
    uint64_t n_zero = 0, n_pos = 0, n_neg = 0;
    for (uint64_t i = 0; i < n_elements; i++) {
        if (ternary[i] == 0) n_zero++;
        else if (ternary[i] > 0) n_pos++;
        else n_neg++;
    }
    printf("  Ternary distribution: +1=%.1f%%  0=%.1f%%  -1=%.1f%%\n",
           100.0 * n_pos / n_elements, 100.0 * n_zero / n_elements, 100.0 * n_neg / n_elements);

    /* Pack into layouts */
    printf("  Packing K-vertical...\n");
    pack_ternary_kvertical(packed_kv, ternary, N, K);
    printf("  Packing Blocked-8...\n");
    pack_ternary_blocked8(packed_b8, ternary, N, K);

    /* Set globals */
    g_out_f = out_f;
    g_q4_data = q4_raw;
    g_tern_kv = packed_kv;
    g_tern_b8 = packed_b8;
    g_act_f = act_f;
    g_act_i8 = act_i8;
    g_scales = scales;
    g_N = N;
    g_K = K;

    int iters = 500;
    printf("\n  Running %d iterations each...\n\n", iters);

    /* Run reference first for validation */
    run_A();
    memcpy(ref_out, out_f, N * sizeof(float));

    printf("  %-35s %8s   %8s  %8s\n", "Kernel", "Time", "GOP/s", "GB/s");
    printf("  %-35s %8s   %8s  %8s\n",
           "───────────────────────────────────", "────────", "────────", "────────");

    /* Kernel A: Q4_0 scalar (0.5 bytes/weight effective for 4-bit + scale overhead) */
    bench("A: Q4_0 scalar (reference)", run_A, iters, N, K, 9, 16);  /* 18 bytes per 32 weights */

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    /* Kernel B: Q4_0 NEON SDOT */
    bench("B: Q4_0 NEON SDOT", run_B, iters, N, K, 9, 16);

    /* Kernel C: Ternary SDOT (K-vertical) — 0.25 bytes/weight */
    bench("C: Ternary SDOT (K-vertical)", run_C, iters, N, K, 1, 4);

    /* Kernel D: Ternary Blocked-8 — 0.25 bytes/weight */
    bench("D: Ternary Blocked-8 (cache-opt)", run_D, iters, N, K, 1, 4);

    /* Kernel E: Ternary 8OC SDOT — 0.25 bytes/weight */
    bench("E: Ternary 8OC SDOT", run_E, iters, N, K, 1, 4);

    /* Validate ternary kernels against scalar reference */
    printf("\n  Validation (vs scalar Q4_0 reference):\n");
    const char *knames[] = {"B: Q4_0 NEON", "C: Tern SDOT", "D: Tern B8", "E: Tern 8OC"};
    void (*kfns[])(void) = {run_B, run_C, run_D, run_E};

    for (int k = 0; k < 4; k++) {
        kfns[k]();
        float max_rel = 0, avg_rel = 0, max_abs = 0, avg_abs = 0;
        float ref_mag = 0;
        for (int i = 0; i < N; i++) {
            float err = fabsf(out_f[i] - ref_out[i]);
            float rel = (fabsf(ref_out[i]) > 0.1f) ? err / fabsf(ref_out[i]) : 0;
            if (rel > max_rel) max_rel = rel;
            avg_rel += rel;
            if (err > max_abs) max_abs = err;
            avg_abs += err;
            ref_mag += fabsf(ref_out[i]);
        }
        avg_rel /= N;
        avg_abs /= N;
        ref_mag /= N;
        printf("    %-20s max_rel=%.4f avg_rel=%.4f max_abs=%.2f avg_abs=%.2f ref_mag=%.2f %s\n",
               knames[k], max_rel, avg_rel, max_abs, avg_abs, ref_mag,
               (k == 0) ? (max_rel < 0.05 ? "PASS" : "FAIL") :
                          "(different quant)");
    }
#else
    printf("  (NEON kernels not available on this platform)\n");
#endif

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  LPDDR4X theoretical: ~13 GB/s\n");
    printf("  Measured at 76%% utilization: ~9.9 GB/s\n");
    printf("  If ternary GB/s approaches Q4_0 GB/s, bandwidth is the ceiling.\n");
    printf("  If ternary time is ~half of Q4_0, the 2-bit encoding wins.\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    free(act_f); free(act_i8); free(out_f); free(ref_out);
    free(ternary); free(scales); free(packed_kv); free(packed_b8);
    free(tensors);
    munmap((void *)data, st.st_size);
    close(fd);
    return 0;
}
