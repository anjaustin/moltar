/*
 * LFM2_TRIX — Frozen TriX Chip for LFM2-350M Inference
 *
 * Decomposes the LFM2 forward pass into compositions of 5 atomic primes:
 *   ADD, MUL, EXP, MAX, CONST
 *
 * Architecture:
 *   - 16 sequential layers, each = operator_block + FFN
 *   - 10 ShortConv layers (recurrent, layers 0,1,3,4,6,7,9,11,13,15)
 *   -  6 Attention layers  (causal,    layers 2,5,8,10,12,14)
 *   - All layers use SwiGLU FFN: down(silu(gate) * up)
 *
 * Target: Motorola moto g power 5G (2023)
 *   - MediaTek Dimensity 930 (6x A55 + 2x A76)
 *   - PowerVR BXM-8-256 (Vulkan 1.1, 16KB shared mem, 512 invocations)
 *   - 3.5 GB LPDDR5 (UMA, shared CPU/GPU)
 *
 * Design principle:
 *   Batch=1 token generation is memory-bandwidth bound (~190 MB weight
 *   reads per token at Q4_0). The chip does NOT split work across CPU
 *   and GPU (they share the same memory bus). Instead it:
 *
 *   1. Runs a PERSISTENT Vulkan compute kernel on the PowerVR GPU
 *   2. Fuses entire layers (operator + FFN) to minimize dispatch overhead
 *   3. Uses ION coherent memory for zero-copy CPU<->GPU token streaming
 *   4. CPU handles only tokenization and sampling (trivial cost)
 *
 * The goal: close the gap between measured 29ms/tok and theoretical
 * minimum 3.7ms/tok by eliminating the ~7.8x overhead from dispatch,
 * synchronization, and memory access patterns.
 *
 * Created by: Tripp + Claude
 * Date: February 1, 2026
 */

#ifndef LFM2_TRIX_H
#define LFM2_TRIX_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * LFM2-350M Model Constants
 * ═══════════════════════════════════════════════════════════════════════════ */

#define LFM2_D_MODEL     1024
#define LFM2_N_LAYERS    16
#define LFM2_N_HEADS     16
#define LFM2_N_KV_HEADS  8
#define LFM2_D_HEAD      64     /* D_MODEL / N_HEADS */
#define LFM2_FFN_HIDDEN  4608
#define LFM2_VOCAB_SIZE  65536
#define LFM2_L_CACHE     3      /* shortconv.l_cache from GGUF metadata — conv kernel width */
#define LFM2_D_CONV      2      /* conv state depth = l_cache - 1 */
#define LFM2_ROPE_BASE   1000000.0f
#define LFM2_NORM_EPS    1e-5f

/* Layer type map: 1 = attention, 0 = shortconv */
#define LFM2_IS_ATTN(il) ((il)==2||(il)==5||(il)==8||(il)==10||(il)==12||(il)==14)

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight Layout
 *
 * All weights are Q4_0 quantized: 32 values packed into 18 bytes
 * (2-byte scale + 16 bytes of 4-bit values).
 *
 * Q4_0 block:
 *   struct { float16 scale; uint8_t qs[16]; } // 32 values per block
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════════
 * FP16 -> FP32 conversion (handles subnormals correctly)
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float trix_fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;

    if (expo == 0) {
        if (mant == 0) {
            float r; uint32_t f = sign << 31;
            __builtin_memcpy(&r, &f, 4);
            return r;
        }
        /* Subnormal: (-1)^sign * 2^(-14) * (mant / 1024) */
        float r = (mant / 1024.0f) * (1.0f / 16384.0f);
        return sign ? -r : r;
    } else if (expo == 31) {
        float r; uint32_t f = (sign << 31) | 0x7F800000 | (mant << 13);
        __builtin_memcpy(&r, &f, 4);
        return r;
    } else {
        float r; uint32_t f = (sign << 31) | ((expo + 112) << 23) | (mant << 13);
        __builtin_memcpy(&r, &f, 4);
        return r;
    }
}

/* Q4_0 block: 32 values packed into 18 bytes */
#define Q4_0_BLOCK_SIZE  32
#define Q4_0_BLOCK_BYTES 18  /* sizeof(float16) + 16 */

typedef struct {
    uint16_t scale_f16;       /* FP16 scale factor */
    uint8_t  qs[Q4_0_BLOCK_SIZE / 2]; /* 4-bit quantized values */
} q4_0_block_t;

/* Q6_K block: 256 values packed into 210 bytes
 * Used for embedding tables that need higher precision */
#define Q6_K_BLOCK_SIZE  256
#define Q6_K_BLOCK_BYTES 210

typedef struct {
    uint8_t  ql[Q6_K_BLOCK_SIZE / 2]; /* lower 4 bits of quantized values */
    uint8_t  qh[Q6_K_BLOCK_SIZE / 4]; /* upper 2 bits of quantized values */
    int8_t   scales[Q6_K_BLOCK_SIZE / 16]; /* scales (16 groups of 16) */
    uint16_t d_f16;                   /* super-block scale (FP16) */
} q6_k_block_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * Per-Layer Weight Structures
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ShortConv layer weights */
typedef struct {
    const q4_0_block_t *in_proj;     /* [D_MODEL, 3*D_MODEL] — Q4_0 */
    const float        *conv_kernel; /* [D_CONV, D_MODEL]    — depthwise, always FP32 */
    const q4_0_block_t *out_proj;    /* [D_MODEL, D_MODEL]   — Q4_0 */
    const float        *norm_weight; /* [D_MODEL]            — RMSNorm gamma, FP32 */
} trix_shortconv_weights_t;

/* Attention layer weights */
typedef struct {
    const q4_0_block_t *wq;          /* [D_MODEL, D_MODEL]         */
    const q4_0_block_t *wk;          /* [D_MODEL, D_MODEL/2] (GQA) */
    const q4_0_block_t *wv;          /* [D_MODEL, D_MODEL/2] (GQA) */
    const q4_0_block_t *wo;          /* [D_MODEL, D_MODEL]         */
    const float        *q_norm;      /* [D_HEAD] — per-head RMSNorm */
    const float        *k_norm;      /* [D_HEAD] — per-head RMSNorm */
    const float        *norm_weight; /* [D_MODEL] — layer RMSNorm  */
} trix_attention_weights_t;

/* FFN weights (shared by both layer types) */
typedef struct {
    const q4_0_block_t *gate;        /* [D_MODEL, FFN_HIDDEN] */
    const q4_0_block_t *up;          /* [D_MODEL, FFN_HIDDEN] */
    const q4_0_block_t *down;        /* [FFN_HIDDEN, D_MODEL] */
    const float        *norm_weight; /* [D_MODEL] — FFN RMSNorm */
} trix_ffn_weights_t;

/* Quantization type tag for flexible tensor handling */
typedef enum {
    TRIX_QTYPE_F32  = 0,
    TRIX_QTYPE_Q4_0 = 2,
    TRIX_QTYPE_Q6_K = 14,
} trix_qtype_t;

/* Complete model weights */
typedef struct {
    /* Embedding and output head
     * token_embd may be Q6_K (common in "Q4_0" GGUFs).
     * output may be NULL if weight-tied to token_embd. */
    const void  *tok_embd;           /* [VOCAB, D_MODEL] — Q6_K or Q4_0 */
    trix_qtype_t tok_embd_type;
    const void  *output;             /* [D_MODEL, VOCAB] — Q6_K, Q4_0, or NULL (tied) */
    trix_qtype_t output_type;
    int          output_tied;        /* 1 if output == tok_embd (weight tying) */
    const float *output_norm;        /* [D_MODEL]        */

    /* Per-layer weights */
    trix_shortconv_weights_t shortconv[10]; /* 10 conv layers */
    trix_attention_weights_t attention[6];  /* 6 attn layers  */
    trix_ffn_weights_t       ffn[16];       /* all 16 layers  */
} trix_lfm2_weights_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * Inference State (mutable, persists across tokens)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* ShortConv recurrent state: [10 layers][D_CONV][D_MODEL] */
    float conv_state[10][LFM2_D_CONV][LFM2_D_MODEL];

    /* KV cache: [6 attn layers][max_ctx][n_kv_heads][d_head] */
    float *k_cache; /* allocated dynamically based on max_ctx */
    float *v_cache;
    uint32_t max_ctx;

    /* Current sequence position */
    uint32_t pos;
} trix_lfm2_state_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * ION Channel — Zero-Copy CPU<->GPU Communication
 *
 * The channel is a coherent memory region visible to both CPU and GPU.
 * CPU writes token_id + increments version.
 * GPU reads version, runs forward pass, writes logits + increments version.
 * No syscalls. No copies. Just atomic loads/stores on shared memory.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* CPU -> GPU */
    volatile uint32_t input_version;
    uint32_t          token_id;
    uint32_t          seq_pos;
    uint32_t          _pad0;

    /* GPU -> CPU */
    volatile uint32_t output_version;
    uint32_t          output_token;   /* argmax result */
    float             top_logit;      /* for diagnostics */
    uint32_t          _pad1;

    /* Shared workspace — the activation buffer.
     * GPU owns this during forward pass. CPU must not touch. */
    float activations[LFM2_D_MODEL];

    /* Logits output (vocab-sized) — written by GPU, read by CPU for sampling */
    float logits[LFM2_VOCAB_SIZE];
} trix_ion_channel_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * TriX Atomic Primes — The 5 Frozen Primitives
 *
 * Every operation in LFM2 decomposes to these:
 *   ADD(a, b) = a + b
 *   MUL(a, b) = a * b
 *   EXP(x)    = e^x
 *   MAX(a, b) = max(a, b)
 *   CONST(v)  = v
 *
 * Derived operations used in LFM2:
 *   MATVEC(W, x)     = SUM_k(MUL(W[i,k], x[k]))      — ADD + MUL
 *   RMSNORM(x, g)    = MUL(g, MUL(x, RSQRT(MEAN(MUL(x,x))))) — MUL + ADD
 *   SILU(x)          = MUL(x, SIGMOID(x))
 *                     = MUL(x, 1/(ADD(1, EXP(MUL(-1, x))))) — MUL + ADD + EXP + CONST
 *   SOFTMAX(x)       = MUL(EXP(ADD(x, MUL(-1, MAX(x)))), 1/SUM(EXP(...)))
 *                                                       — all 5 primes
 *   ROPE(x, pos)     = ADD(MUL(x, COS(pos*freq)), MUL(x_rot, SIN(pos*freq)))
 *                                                       — MUL + ADD + CONST
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════════
 * Chip API — The Frozen Interface
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * trix_lfm2_init — Allocate and initialize inference state
 * @max_ctx: Maximum context length (determines KV cache size)
 * @return: Allocated state, or NULL on failure
 */
trix_lfm2_state_t *trix_lfm2_init(uint32_t max_ctx);

/**
 * trix_lfm2_free — Release inference state
 */
void trix_lfm2_free(trix_lfm2_state_t *state);

/**
 * trix_lfm2_forward — Run one token through the full 16-layer model
 *
 * This is the frozen chip. Given a token embedding, it produces logits.
 * Every operation is a composition of the 5 primes.
 *
 * @state:   Mutable inference state (conv_state, KV cache, position)
 * @weights: Frozen model weights (immutable)
 * @x:       Input embedding [D_MODEL] (modified in-place as working buffer)
 * @logits:  Output logits [VOCAB_SIZE]
 */
void trix_lfm2_forward(
    trix_lfm2_state_t       *state,
    const trix_lfm2_weights_t *weights,
    float                    *x,
    float                    *logits
);

/* ═══════════════════════════════════════════════════════════════════════════
 * Sub-Chips — Fused Layer Primitives
 *
 * Each is a frozen composition of primes that processes one layer.
 * The forward() function calls these sequentially.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * CHIP: RMSNorm
 * Primes: MUL, ADD, CONST(eps)
 * out[i] = gamma[i] * x[i] * rsqrt(mean(x^2) + eps)
 */
void trix_rmsnorm(const float *x, const float *gamma, float *out, int n);

/**
 * CHIP: Q4_0 Matrix-Vector Multiply (the hot path)
 * Primes: MUL, ADD
 * y[i] = SUM_k(dequant(W[i,k]) * x[k])
 */
void trix_matvec_q4_0(
    const q4_0_block_t *W,
    const float        *x,
    float              *y,
    int M, int K
);

/**
 * CHIP: Q6_K Matrix-Vector Multiply (for embedding/output head)
 * Primes: MUL, ADD
 * y[i] = SUM_k(dequant(W[i,k]) * x[k])
 */
void trix_matvec_q6_k(
    const q6_k_block_t *W,
    const float        *x,
    float              *y,
    int M, int K
);

/**
 * CHIP: Fused ShortConv Block
 * Primes: MUL, ADD (via MATVEC), plus state shift
 *
 * 1. bcx = in_proj @ x                    — MATVEC (MUL+ADD)
 * 2. b, c, x_proj = split(bcx)            — view (free)
 * 3. bx = b * x_proj                      — MUL
 * 4. conv_out = depthwise_conv(bx, state)  — MUL+ADD (dot product per channel)
 * 5. y = c * conv_out                      — MUL
 * 6. out = out_proj @ y                    — MATVEC (MUL+ADD)
 */
void trix_shortconv_chip(
    const trix_shortconv_weights_t *w,
    float conv_state[LFM2_D_CONV][LFM2_D_MODEL],
    const float *x,
    float       *out
);

/**
 * CHIP: Fused Attention Block
 * Primes: all 5 (MUL+ADD for projections, EXP+MAX for softmax, CONST for scale)
 *
 * 1. q = wq @ x, k = wk @ x, v = wv @ x  — 3x MATVEC
 * 2. q, k = rmsnorm(q), rmsnorm(k)         — RMSNORM
 * 3. q, k = rope(q, k, pos)                — MUL+ADD+CONST
 * 4. k_cache[pos] = k, v_cache[pos] = v    — store
 * 5. scores = q @ k_cache^T / sqrt(d_head)  — MATVEC + MUL(CONST)
 * 6. weights = softmax(scores)              — MAX+EXP+ADD+MUL
 * 7. attn_out = weights @ v_cache            — MATVEC
 * 8. out = wo @ attn_out                     — MATVEC
 */
void trix_attention_chip(
    const trix_attention_weights_t *w,
    float *k_cache,
    float *v_cache,
    uint32_t pos,
    uint32_t max_ctx,
    const float *x,
    float       *out
);

/**
 * CHIP: Fused SwiGLU FFN
 * Primes: MUL, ADD, EXP, CONST
 *
 * 1. gate_out = gate @ x                   — MATVEC (MUL+ADD)
 * 2. up_out = up @ x                       — MATVEC (MUL+ADD)
 * 3. hidden = silu(gate_out) * up_out       — MUL+EXP+ADD+CONST (silu) + MUL
 * 4. out = down @ hidden                    — MATVEC (MUL+ADD)
 */
void trix_ffn_chip(
    const trix_ffn_weights_t *w,
    const float *x,
    float       *out
);

/**
 * CHIP: RoPE (Rotary Position Embedding)
 * Primes: MUL, ADD, CONST
 *
 * For each pair (x[2i], x[2i+1]):
 *   freq = 1.0 / (base ^ (2i / d_head))
 *   cos_val = cos(pos * freq)
 *   sin_val = sin(pos * freq)
 *   out[2i]   = x[2i] * cos_val - x[2i+1] * sin_val
 *   out[2i+1] = x[2i] * sin_val + x[2i+1] * cos_val
 */
void trix_rope(float *x, int dim, uint32_t pos, float base);

/* ═══════════════════════════════════════════════════════════════════════════
 * GGUF Loader API
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Opaque handle — holds mmap'd file + weight pointers */
typedef struct trix_gguf_model trix_gguf_model_t;

/**
 * trix_load_gguf — Load a Q4_0 GGUF file and map all tensors
 * @path: Path to .gguf file
 * @return: Model handle, or NULL on failure (prints errors to stderr)
 *
 * The returned handle keeps the file mmap'd. All weight pointers in
 * trix_lfm2_weights_t point directly into the mmap region (zero-copy).
 */
trix_gguf_model_t *trix_load_gguf(const char *path);

/**
 * trix_unload_gguf — Release model handle and munmap the file
 */
void trix_unload_gguf(trix_gguf_model_t *model);

/**
 * trix_get_weights — Get the weight structure from a loaded model
 */
const trix_lfm2_weights_t *trix_get_weights(const trix_gguf_model_t *model);

/**
 * trix_embed_token — Dequantize one row of the Q4_0 embedding table
 * @w:        Model weights
 * @token_id: Token to look up
 * @out:      Output buffer [D_MODEL]
 */
void trix_embed_token(
    const trix_lfm2_weights_t *w,
    uint32_t token_id,
    float *out
);

#ifdef __cplusplus
}
#endif

#endif /* LFM2_TRIX_H */
