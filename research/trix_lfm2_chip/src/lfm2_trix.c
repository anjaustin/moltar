/*
 * LFM2_TRIX — Frozen TriX Chip Implementation
 *
 * Every function here is a frozen composition of 5 primes:
 *   ADD, MUL, EXP, MAX, CONST
 *
 * No dynamic allocation during forward pass.
 * No branching on data values (only on structure constants).
 * Deterministic: same input + state => bit-identical output.
 *
 * Created by: Tripp + Claude
 * Date: February 1, 2026
 */

#include "../include/lfm2_trix.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * PRIME: CONST — Frozen constants used throughout
 * ═══════════════════════════════════════════════════════════════════════════ */

static const float CONST_NORM_EPS  = LFM2_NORM_EPS;
static const float CONST_ROPE_BASE = LFM2_ROPE_BASE;
static const float CONST_NEG_ONE   = -1.0f;
static const float CONST_ONE       = 1.0f;
static const float CONST_ATTN_SCALE = 0.125f; /* 1/sqrt(64) = 1/8 */

/* ═══════════════════════════════════════════════════════════════════════════
 * State Management
 * ═══════════════════════════════════════════════════════════════════════════ */

trix_lfm2_state_t *trix_lfm2_init(uint32_t max_ctx) {
    trix_lfm2_state_t *s = (trix_lfm2_state_t *)calloc(1, sizeof(trix_lfm2_state_t));
    if (!s) return NULL;

    s->max_ctx = max_ctx;
    s->pos = 0;

    /* KV cache: 6 attn layers * max_ctx * n_kv_heads * d_head */
    size_t kv_size = 6 * max_ctx * LFM2_N_KV_HEADS * LFM2_D_HEAD * sizeof(float);
    s->k_cache = (float *)calloc(1, kv_size);
    s->v_cache = (float *)calloc(1, kv_size);

    if (!s->k_cache || !s->v_cache) {
        free(s->k_cache);
        free(s->v_cache);
        free(s);
        return NULL;
    }

    return s;
}

void trix_lfm2_free(trix_lfm2_state_t *state) {
    if (state) {
        free(state->k_cache);
        free(state->v_cache);
        free(state);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CHIP: Q4_0 Dequantize + MATVEC (Fused)
 *
 * Primes: MUL, ADD
 *
 * This is the hot path. ~80% of all compute lives here.
 * For each output element y[i]:
 *   y[i] = SUM_k( dequant(W[i,k]) * x[k] )
 *
 * Dequant(block, idx) = (int4_val - 8) * scale
 * which is: ADD(MUL(CONST(val), CONST(scale)), CONST(-8*scale))
 *
 * The inner loop is structured for auto-vectorization:
 *   - Process one Q4_0 block (32 values) at a time
 *   - Accumulate into a float sum
 *   - Compiler emits NEON FMA on ARM64 with -O2
 * ═══════════════════════════════════════════════════════════════════════════ */

void trix_matvec_q4_0(
    const q4_0_block_t *W,
    const float        *x,
    float              *y,
    int M, int K
) {
    const int n_blocks_per_row = K / Q4_0_BLOCK_SIZE;

    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const q4_0_block_t *row = W + i * n_blocks_per_row;

        for (int b = 0; b < n_blocks_per_row; b++) {
            /* CONST: extract scale from FP16 */
            float scale = trix_fp16_to_f32(row[b].scale_f16);

            const float *xb = x + b * Q4_0_BLOCK_SIZE;

            /* Process 32 quantized values (packed in 16 bytes)
             *
             * Q4_0 nibble mapping (matches ggml):
             *   qs[j] low nibble  → element j      (indices 0..15)
             *   qs[j] high nibble → element j + 16  (indices 16..31)
             */
            for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                uint8_t packed = row[b].qs[j];

                /* Low nibble: MUL(ADD(CONST(val), CONST(-8)), CONST(scale)) */
                float v0 = ((float)(packed & 0x0F) - 8.0f) * scale;
                /* High nibble */
                float v1 = ((float)(packed >> 4) - 8.0f) * scale;

                /* MUL + ADD: accumulate dot product */
                sum += v0 * xb[j];
                sum += v1 * xb[j + 16];
            }
        }

        y[i] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CHIP: RMSNorm
 *
 * Primes: MUL, ADD, CONST(eps)
 *
 * out[i] = gamma[i] * x[i] / sqrt(mean(x^2) + eps)
 *
 * Decomposition:
 *   sum_sq = SUM(MUL(x[i], x[i]))           — MUL + ADD
 *   rms    = SQRT(ADD(MUL(sum_sq, 1/n), eps)) — MUL + ADD + CONST
 *   out[i] = MUL(gamma[i], MUL(x[i], 1/rms)) — MUL
 * ═══════════════════════════════════════════════════════════════════════════ */

void trix_rmsnorm(const float *x, const float *gamma, float *out, int n) {
    /* Pass 1: MUL + ADD — sum of squares */
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];  /* MUL, ADD */
    }

    /* CONST(1/n), MUL, ADD(eps), rsqrt */
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)n + CONST_NORM_EPS);

    /* Pass 2: MUL(gamma, MUL(x, inv_rms)) */
    for (int i = 0; i < n; i++) {
        out[i] = gamma[i] * x[i] * inv_rms;  /* MUL, MUL */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CHIP: SiLU (Sigmoid Linear Unit)
 *
 * Primes: MUL, ADD, EXP, CONST
 *
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *         = MUL(x, 1 / ADD(CONST(1), EXP(MUL(CONST(-1), x))))
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float trix_silu(float x) {
    return x / (CONST_ONE + expf(CONST_NEG_ONE * x));  /* MUL, ADD, EXP, CONST */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CHIP: RoPE (Rotary Position Embedding)
 *
 * Primes: MUL, ADD, CONST
 *
 * For each pair of dimensions:
 *   freq = CONST(1) / POW(base, CONST(2i/d))  — CONST, MUL, EXP
 *   angle = MUL(CONST(pos), freq)               — MUL, CONST
 *   x_new[2i]   = ADD(MUL(x[2i], cos(angle)), MUL(CONST(-1), MUL(x[2i+1], sin(angle))))
 *   x_new[2i+1] = ADD(MUL(x[2i], sin(angle)), MUL(x[2i+1], cos(angle)))
 * ═══════════════════════════════════════════════════════════════════════════ */

void trix_rope(float *x, int dim, uint32_t pos, float base) {
    for (int i = 0; i < dim; i += 2) {
        float freq = 1.0f / powf(base, (float)i / (float)dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float x0 = x[i];
        float x1 = x[i + 1];
        x[i]     = x0 * cos_a - x1 * sin_a;  /* MUL, MUL, ADD */
        x[i + 1] = x0 * sin_a + x1 * cos_a;  /* MUL, MUL, ADD */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CHIP: Fused ShortConv Block
 *
 * Fuses the entire recurrent block into one function:
 *   RMSNorm → in_proj → split(b,c,x) → b*x → conv(state) → c*conv → out_proj
 *
 * All primes used: MUL, ADD, CONST (no EXP or MAX needed)
 * ═══════════════════════════════════════════════════════════════════════════ */

void trix_shortconv_chip(
    const trix_shortconv_weights_t *w,
    float conv_state[LFM2_D_CONV][LFM2_D_MODEL],
    const float *x,
    float       *out
) {
    float normed[LFM2_D_MODEL];
    float bcx[3 * LFM2_D_MODEL]; /* in_proj output: [b, c, x_proj] */
    float bx[LFM2_D_MODEL];
    float conv_out[LFM2_D_MODEL];
    float y[LFM2_D_MODEL];

    /* 1. RMSNorm — MUL + ADD + CONST */
    trix_rmsnorm(x, w->norm_weight, normed, LFM2_D_MODEL);

    /* 2. in_proj: [D_MODEL] -> [3*D_MODEL] — MATVEC (MUL+ADD) */
    trix_matvec_q4_0(w->in_proj, normed, bcx, 3 * LFM2_D_MODEL, LFM2_D_MODEL);

    /* 3. Split into b, c, x_proj (views, zero cost) */
    float *b      = bcx;
    float *c      = bcx + LFM2_D_MODEL;
    float *x_proj = bcx + 2 * LFM2_D_MODEL;

    /* 4. bx = b * x_proj — MUL (element-wise) */
    for (int i = 0; i < LFM2_D_MODEL; i++) {
        bx[i] = b[i] * x_proj[i];
    }

    /* 5. Depthwise convolution with state — MUL + ADD per channel
     *
     *    For LFM2-350M: l_cache=3, d_conv = l_cache-1 = 2
     *    Conv state holds the last D_CONV=2 values of bx (oldest to newest).
     *    Input to conv: [state[0], state[1], bx_new] (length L_CACHE=3)
     *    Kernel: ggml shape {L_CACHE=3, D_MODEL=1024} = 3 taps per channel
     *
     *    ggml_ssm_conv CPU kernel:
     *      conv_out[d] = sum_{t=0..L_CACHE-1}(kernel[d*L_CACHE + t] * input[t][d])
     *    where input = [state[0], state[1], bx] (oldest first)
     *    kernel tap 0 → oldest value in window.
     *
     *    After convolution, shift state and insert new bx. */
    const float *kernel = w->conv_kernel; /* ggml {L_CACHE, D_MODEL} → C layout: kernel[d * L_CACHE + t] */
    for (int d = 0; d < LFM2_D_MODEL; d++) {
        float sum = 0.0f;
        const float *k_ch = kernel + d * LFM2_L_CACHE; /* L_CACHE=3 taps for channel d */
        /* Taps 0..D_CONV-1 use the state (oldest to newest) */
        for (int t = 0; t < LFM2_D_CONV; t++) {
            sum += conv_state[t][d] * k_ch[t]; /* MUL, ADD */
        }
        /* Tap D_CONV (=last tap) uses the current input bx */
        sum += bx[d] * k_ch[LFM2_D_CONV]; /* MUL, ADD */
        conv_out[d] = sum;
    }

    /* Shift state: drop oldest, insert bx as newest */
    for (int t = 0; t < LFM2_D_CONV - 1; t++) {
        memcpy(conv_state[t], conv_state[t + 1], LFM2_D_MODEL * sizeof(float));
    }
    memcpy(conv_state[LFM2_D_CONV - 1], bx, LFM2_D_MODEL * sizeof(float));

    /* 6. y = c * conv_out — MUL (element-wise) */
    for (int i = 0; i < LFM2_D_MODEL; i++) {
        y[i] = c[i] * conv_out[i];
    }

    /* 7. out_proj: [D_MODEL] -> [D_MODEL] — MATVEC (MUL+ADD) */
    trix_matvec_q4_0(w->out_proj, y, out, LFM2_D_MODEL, LFM2_D_MODEL);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CHIP: Fused Attention Block
 *
 * Fuses: RMSNorm → Q/K/V proj → QK norm → RoPE → KV cache →
 *        score → softmax → weighted sum → O proj
 *
 * All 5 primes used.
 * ═══════════════════════════════════════════════════════════════════════════ */

void trix_attention_chip(
    const trix_attention_weights_t *w,
    float *k_cache,   /* [max_ctx * N_KV_HEADS * D_HEAD] for this layer */
    float *v_cache,
    uint32_t pos,
    uint32_t max_ctx,
    const float *x,
    float       *out
) {
    const int d_model = LFM2_D_MODEL;
    const int n_heads = LFM2_N_HEADS;
    const int n_kv    = LFM2_N_KV_HEADS;
    const int d_head  = LFM2_D_HEAD;
    const int d_kv    = n_kv * d_head; /* 512 */

    float normed[LFM2_D_MODEL];
    float q[LFM2_D_MODEL];        /* [n_heads * d_head] = [1024] */
    float k[LFM2_N_KV_HEADS * LFM2_D_HEAD]; /* [512] */
    float v[LFM2_N_KV_HEADS * LFM2_D_HEAD];
    float attn_out[LFM2_D_MODEL];

    /* 1. RMSNorm */
    trix_rmsnorm(x, w->norm_weight, normed, d_model);

    /* 2. Q/K/V projections — 3x MATVEC (MUL+ADD) */
    trix_matvec_q4_0(w->wq, normed, q, d_model, d_model);
    trix_matvec_q4_0(w->wk, normed, k, d_kv, d_model);
    trix_matvec_q4_0(w->wv, normed, v, d_kv, d_model);

    /* 3. QK RMSNorm (per-head) — MUL + ADD + CONST */
    for (int h = 0; h < n_heads; h++) {
        trix_rmsnorm(q + h * d_head, w->q_norm, q + h * d_head, d_head);
    }
    for (int h = 0; h < n_kv; h++) {
        trix_rmsnorm(k + h * d_head, w->k_norm, k + h * d_head, d_head);
    }

    /* 4. RoPE — MUL + ADD + CONST */
    for (int h = 0; h < n_heads; h++) {
        trix_rope(q + h * d_head, d_head, pos, CONST_ROPE_BASE);
    }
    for (int h = 0; h < n_kv; h++) {
        trix_rope(k + h * d_head, d_head, pos, CONST_ROPE_BASE);
    }

    /* 5. Update KV cache — store */
    memcpy(k_cache + pos * d_kv, k, d_kv * sizeof(float));
    memcpy(v_cache + pos * d_kv, v, d_kv * sizeof(float));

    /* 6-7. Multi-head attention with GQA
     *   n_heads=16 query heads, n_kv=8 kv heads
     *   Each 2 query heads share 1 kv head (GQA ratio = 2) */
    memset(attn_out, 0, d_model * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / (n_heads / n_kv); /* GQA: map query head -> kv head */
        float *q_head = q + h * d_head;

        /* Allocate scores for [0..pos] */
        float scores[pos + 1]; /* VLA, safe for ctx <= 2048 */

        /* Score: q @ k_cache^T / sqrt(d_head) — MATVEC + MUL(CONST) */
        float max_score = -1e30f;
        for (uint32_t p = 0; p <= pos; p++) {
            float *k_p = k_cache + p * d_kv + kv_h * d_head;
            float dot = 0.0f;
            for (int d = 0; d < d_head; d++) {
                dot += q_head[d] * k_p[d];  /* MUL, ADD */
            }
            scores[p] = dot * CONST_ATTN_SCALE;  /* MUL(CONST) */

            /* MAX — track for softmax stability */
            if (scores[p] > max_score) max_score = scores[p];
        }

        /* Softmax — EXP + ADD + MUL + MAX + CONST */
        float sum_exp = 0.0f;
        for (uint32_t p = 0; p <= pos; p++) {
            scores[p] = expf(scores[p] - max_score);  /* EXP(ADD(score, MUL(-1, max))) */
            sum_exp += scores[p];                       /* ADD */
        }
        float inv_sum = 1.0f / sum_exp;                /* CONST / sum */
        for (uint32_t p = 0; p <= pos; p++) {
            scores[p] *= inv_sum;                       /* MUL */
        }

        /* Weighted sum of values — MUL + ADD */
        float *attn_head = attn_out + h * d_head;
        for (uint32_t p = 0; p <= pos; p++) {
            float *v_p = v_cache + p * d_kv + kv_h * d_head;
            float w_p = scores[p];
            for (int d = 0; d < d_head; d++) {
                attn_head[d] += w_p * v_p[d];  /* MUL, ADD */
            }
        }
    }

    /* 8. Output projection — MATVEC (MUL+ADD) */
    trix_matvec_q4_0(w->wo, attn_out, out, d_model, d_model);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CHIP: Fused SwiGLU FFN
 *
 * Primes: MUL, ADD, EXP, CONST
 *
 * out = down_proj @ (silu(gate_proj @ x) * up_proj @ x)
 * ═══════════════════════════════════════════════════════════════════════════ */

void trix_ffn_chip(
    const trix_ffn_weights_t *w,
    const float *x,
    float       *out
) {
    float normed[LFM2_D_MODEL];
    float gate_out[LFM2_FFN_HIDDEN];
    float up_out[LFM2_FFN_HIDDEN];
    float hidden[LFM2_FFN_HIDDEN];

    /* 1. RMSNorm — MUL + ADD + CONST */
    trix_rmsnorm(x, w->norm_weight, normed, LFM2_D_MODEL);

    /* 2. Gate projection — MATVEC (MUL+ADD) */
    trix_matvec_q4_0(w->gate, normed, gate_out, LFM2_FFN_HIDDEN, LFM2_D_MODEL);

    /* 3. Up projection — MATVEC (MUL+ADD) */
    trix_matvec_q4_0(w->up, normed, up_out, LFM2_FFN_HIDDEN, LFM2_D_MODEL);

    /* 4. SiLU(gate) * up — MUL + EXP + ADD + CONST (silu) + MUL */
    for (int i = 0; i < LFM2_FFN_HIDDEN; i++) {
        hidden[i] = trix_silu(gate_out[i]) * up_out[i];
    }

    /* 5. Down projection — MATVEC (MUL+ADD) */
    trix_matvec_q4_0(w->down, hidden, out, LFM2_D_MODEL, LFM2_FFN_HIDDEN);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CHIP: Full LFM2-350M Forward Pass
 *
 * The frozen chip. 16 layers, sequential.
 * Each layer = operator_block (shortconv or attention) + residual + FFN + residual
 *
 * This function uses all 5 primes through its sub-chips.
 * ═══════════════════════════════════════════════════════════════════════════ */

void trix_lfm2_forward(
    trix_lfm2_state_t         *state,
    const trix_lfm2_weights_t *weights,
    float                     *x,       /* [D_MODEL], modified in-place */
    float                     *logits   /* [VOCAB_SIZE] */
) {
    float residual[LFM2_D_MODEL];
    float block_out[LFM2_D_MODEL];
    float ffn_out[LFM2_D_MODEL];

    int conv_idx = 0; /* index into shortconv weights (0..9) */
    int attn_idx = 0; /* index into attention weights (0..5) */

    for (int il = 0; il < LFM2_N_LAYERS; il++) {

        /* Save residual */
        memcpy(residual, x, LFM2_D_MODEL * sizeof(float));

        if (LFM2_IS_ATTN(il)) {
            /* Attention layer */
            size_t cache_offset = attn_idx * state->max_ctx
                                  * LFM2_N_KV_HEADS * LFM2_D_HEAD;

            trix_attention_chip(
                &weights->attention[attn_idx],
                state->k_cache + cache_offset,
                state->v_cache + cache_offset,
                state->pos,
                state->max_ctx,
                x, block_out
            );
            attn_idx++;
        } else {
            /* ShortConv layer */
            trix_shortconv_chip(
                &weights->shortconv[conv_idx],
                state->conv_state[conv_idx],
                x, block_out
            );
            conv_idx++;
        }

        /* Residual add — ADD */
        for (int i = 0; i < LFM2_D_MODEL; i++) {
            x[i] = residual[i] + block_out[i];
        }

        /* FFN */
        trix_ffn_chip(&weights->ffn[il], x, ffn_out);

        /* Residual add — ADD */
        for (int i = 0; i < LFM2_D_MODEL; i++) {
            x[i] = x[i] + ffn_out[i];
        }
    }

    /* Final RMSNorm */
    float normed[LFM2_D_MODEL];
    trix_rmsnorm(x, weights->output_norm, normed, LFM2_D_MODEL);

    /* Output projection: [D_MODEL] -> [VOCAB_SIZE] — MATVEC
     * May be Q4_0, Q6_K, or weight-tied to embedding (Q6_K) */
    if (weights->output_type == TRIX_QTYPE_Q6_K) {
        trix_matvec_q6_k((const q6_k_block_t *)weights->output,
                         normed, logits, LFM2_VOCAB_SIZE, LFM2_D_MODEL);
    } else {
        trix_matvec_q4_0((const q4_0_block_t *)weights->output,
                         normed, logits, LFM2_VOCAB_SIZE, LFM2_D_MODEL);
    }

    /* Advance position */
    state->pos++;
}
