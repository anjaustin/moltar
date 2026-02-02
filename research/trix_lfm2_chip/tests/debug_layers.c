/*
 * debug_layers — Layer-by-layer activation dump for debugging
 * Runs a single token through each layer and dumps intermediate values.
 */

#include "../include/lfm2_trix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void print_vec(const char *name, const float *v, int n, int show) {
    float norm = 0, mn = v[0], mx = v[0];
    for (int i = 0; i < n; i++) {
        norm += v[i] * v[i];
        if (v[i] < mn) mn = v[i];
        if (v[i] > mx) mx = v[i];
    }
    norm = sqrtf(norm);
    printf("  %-30s L2=%.6f  min=%.6f  max=%.6f  first%d=[", name, norm, mn, mx, show);
    for (int i = 0; i < show && i < n; i++) printf("%.4f%s", v[i], i < show-1 ? "," : "");
    printf("]\n");
}

/* Expose internal functions for debugging: reimplements forward pass with instrumentation */
extern void trix_rmsnorm(const float *x, const float *gamma, float *out, int n);
extern void trix_matvec_q4_0(const q4_0_block_t *W, const float *x, float *y, int M, int K);
extern void trix_matvec_q6_k(const q6_k_block_t *W, const float *x, float *y, int M, int K);
extern void trix_rope(float *x, int dim, uint32_t pos, float base);

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf> [token_id]\n", argv[0]); return 1; }

    uint32_t token_id = argc >= 3 ? (uint32_t)atoi(argv[2]) : 1;

    trix_gguf_model_t *model = trix_load_gguf(argv[1]);
    if (!model) { fprintf(stderr, "FATAL: failed to load model\n"); return 1; }
    const trix_lfm2_weights_t *w = trix_get_weights(model);

    printf("\n=== Debugging token %u ===\n\n", token_id);

    /* Embedding */
    float x[LFM2_D_MODEL];
    trix_embed_token(w, token_id, x);
    print_vec("embedding", x, LFM2_D_MODEL, 4);

    /* Layer 0: ShortConv */
    printf("\n--- Layer 0 (ShortConv) ---\n");
    {
        const trix_shortconv_weights_t *sw = &w->shortconv[0];
        float conv_state[LFM2_D_CONV][LFM2_D_MODEL];
        memset(conv_state, 0, sizeof(conv_state));

        /* Step 1: RMSNorm */
        float normed[LFM2_D_MODEL];
        trix_rmsnorm(x, sw->norm_weight, normed, LFM2_D_MODEL);
        print_vec("after_norm", normed, LFM2_D_MODEL, 4);

        /* Step 2: in_proj */
        float bcx[3 * LFM2_D_MODEL];
        trix_matvec_q4_0(sw->in_proj, normed, bcx, 3 * LFM2_D_MODEL, LFM2_D_MODEL);
        print_vec("in_proj_out[0..D] (b)", bcx, LFM2_D_MODEL, 4);
        print_vec("in_proj_out[D..2D] (c)", bcx + LFM2_D_MODEL, LFM2_D_MODEL, 4);
        print_vec("in_proj_out[2D..3D] (x)", bcx + 2*LFM2_D_MODEL, LFM2_D_MODEL, 4);

        /* Step 3: b*x */
        float bx[LFM2_D_MODEL];
        for (int i = 0; i < LFM2_D_MODEL; i++) bx[i] = bcx[i] * bcx[2*LFM2_D_MODEL + i];
        print_vec("bx = b*x_proj", bx, LFM2_D_MODEL, 4);

        /* Step 4: Conv (with zero state) */
        float conv_out[LFM2_D_MODEL];
        const float *kernel = sw->conv_kernel;
        for (int d = 0; d < LFM2_D_MODEL; d++) {
            float sum = 0.0f;
            const float *k_ch = kernel + d * LFM2_L_CACHE;
            for (int t = 0; t < LFM2_D_CONV; t++) {
                sum += conv_state[t][d] * k_ch[t];
            }
            sum += bx[d] * k_ch[LFM2_D_CONV];
            conv_out[d] = sum;
        }
        print_vec("conv_out", conv_out, LFM2_D_MODEL, 4);

        /* Step 5: c * conv_out */
        float y[LFM2_D_MODEL];
        for (int i = 0; i < LFM2_D_MODEL; i++) y[i] = bcx[LFM2_D_MODEL + i] * conv_out[i];
        print_vec("y = c * conv_out", y, LFM2_D_MODEL, 4);

        /* Step 6: out_proj */
        float block_out[LFM2_D_MODEL];
        trix_matvec_q4_0(sw->out_proj, y, block_out, LFM2_D_MODEL, LFM2_D_MODEL);
        print_vec("block_out (out_proj)", block_out, LFM2_D_MODEL, 4);

        /* Residual */
        float residual[LFM2_D_MODEL];
        for (int i = 0; i < LFM2_D_MODEL; i++) residual[i] = x[i] + block_out[i];
        print_vec("after_residual", residual, LFM2_D_MODEL, 4);

        /* FFN */
        float ffn_out[LFM2_D_MODEL];
        trix_ffn_chip(&w->ffn[0], residual, ffn_out);
        print_vec("ffn_out", ffn_out, LFM2_D_MODEL, 4);

        /* FFN residual */
        for (int i = 0; i < LFM2_D_MODEL; i++) x[i] = residual[i] + ffn_out[i];
        print_vec("after_layer_0", x, LFM2_D_MODEL, 4);
    }

    /* Now run full forward pass and compare */
    printf("\n--- Full forward pass ---\n");
    {
        trix_lfm2_state_t *state = trix_lfm2_init(2048);
        float xf[LFM2_D_MODEL];
        float logits[LFM2_VOCAB_SIZE];
        trix_embed_token(w, token_id, xf);
        trix_lfm2_forward(state, w, xf, logits);
        print_vec("final_hidden", xf, LFM2_D_MODEL, 4);
        print_vec("logits (first 8)", logits, 8, 8);

        /* Top 5 */
        int top[5] = {0}; float topv[5] = {-1e30,-1e30,-1e30,-1e30,-1e30};
        for (int i = 0; i < LFM2_VOCAB_SIZE; i++) {
            if (logits[i] > topv[4]) {
                for (int k = 0; k < 5; k++) {
                    if (logits[i] > topv[k]) {
                        for (int j = 4; j > k; j--) { top[j]=top[j-1]; topv[j]=topv[j-1]; }
                        top[k]=i; topv[k]=logits[i]; break;
                    }
                }
            }
        }
        printf("  Top 5:\n");
        for (int k = 0; k < 5; k++) printf("    token=%d  logit=%.4f\n", top[k], topv[k]);

        trix_lfm2_free(state);
    }

    trix_unload_gguf(model);
    return 0;
}
