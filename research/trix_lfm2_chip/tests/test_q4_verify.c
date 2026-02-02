/*
 * test_q4_verify — Verify Q4_0 dequantization and matvec against manual computation
 */
#include "../include/lfm2_trix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }

    trix_gguf_model_t *model = trix_load_gguf(argv[1]);
    if (!model) return 1;
    const trix_lfm2_weights_t *w = trix_get_weights(model);

    /* Test 1: Verify Q4_0 block structure by dequantizing first block of attn_norm weight layer 0 */
    /* attn_norm is F32, so let's use the in_proj weight which is Q4_0 */
    printf("=== Q4_0 Dequantization Verification ===\n\n");

    /* First, let's dequant the first block of blk.0.shortconv.in_proj.weight */
    const q4_0_block_t *in_proj = w->shortconv[0].in_proj;
    float scale = trix_fp16_to_f32(in_proj[0].scale_f16);
    printf("First Q4_0 block of in_proj:\n");
    printf("  scale_f16 = 0x%04X  -> %.6f\n", in_proj[0].scale_f16, scale);
    printf("  First 8 nibbles (raw): ");
    for (int j = 0; j < 4; j++) {
        uint8_t packed = in_proj[0].qs[j];
        printf("[%d,%d] ", packed & 0xF, packed >> 4);
    }
    printf("\n  Dequantized: ");
    for (int j = 0; j < 4; j++) {
        uint8_t packed = in_proj[0].qs[j];
        float v0 = ((float)(packed & 0xF) - 8.0f) * scale;
        float v1 = ((float)(packed >> 4) - 8.0f) * scale;
        printf("%.6f %.6f ", v0, v1);
    }
    printf("\n\n");

    /* Test 2: Verify embedding by dequantizing token 1 manually */
    printf("=== Q6_K Embedding Verification (token 1) ===\n");
    float emb_auto[LFM2_D_MODEL];
    trix_embed_token(w, 1, emb_auto);
    printf("  First 4 via trix_embed_token: %.6f %.6f %.6f %.6f\n",
           emb_auto[0], emb_auto[1], emb_auto[2], emb_auto[3]);

    /* Test 3: Verify matvec with a simple known input */
    printf("\n=== Matvec Verification ===\n");
    /* Create an identity-like input: all zeros except x[0]=1.0 */
    float x_test[LFM2_D_MODEL];
    memset(x_test, 0, sizeof(x_test));
    x_test[0] = 1.0f;

    /* Compute in_proj @ x_test — this should give us the first column of in_proj (dequantized) */
    float y_test[3 * LFM2_D_MODEL];
    trix_matvec_q4_0(w->shortconv[0].in_proj, x_test, y_test, 3 * LFM2_D_MODEL, LFM2_D_MODEL);
    printf("  in_proj @ e_0 (first 4): %.6f %.6f %.6f %.6f\n",
           y_test[0], y_test[1], y_test[2], y_test[3]);

    /* Compare: manually dequant row 0, element 0 of in_proj
     * This should equal y_test[0] since x=[1,0,0,...] */
    float manual_val = ((float)(in_proj[0].qs[0] & 0xF) - 8.0f) * scale;
    printf("  Manual dequant row0[0]: %.6f  (match: %s)\n",
           manual_val, fabsf(manual_val - y_test[0]) < 1e-6 ? "YES" : "NO");

    /* Test 4: Verify RMSNorm */
    printf("\n=== RMSNorm Verification ===\n");
    float x_norm_test[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gamma_test[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out_norm[4];
    trix_rmsnorm(x_norm_test, gamma_test, out_norm, 4);
    /* RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.7386 */
    /* inv_rms = 1/2.7386 = 0.3651 */
    float rms = sqrtf((1+4+9+16)/4.0f);
    printf("  Input: [1, 2, 3, 4], gamma=[1,1,1,1]\n");
    printf("  Expected RMS: %.6f  inv_rms: %.6f\n", rms, 1.0f/rms);
    printf("  Expected output: [%.6f, %.6f, %.6f, %.6f]\n",
           1.0f/rms, 2.0f/rms, 3.0f/rms, 4.0f/rms);
    printf("  Actual output:   [%.6f, %.6f, %.6f, %.6f]\n",
           out_norm[0], out_norm[1], out_norm[2], out_norm[3]);

    /* Test 5: Check if the Q4_0 matvec is doing the right thing for the first attention layer */
    printf("\n=== Attention Layer Q/K/V Check ===\n");
    float emb[LFM2_D_MODEL];
    trix_embed_token(w, 1, emb);

    /* Apply layer-0 shortconv + FFN to get the input to layer 2 (first attn) */
    /* For simplicity, just check the raw embedding norm and projection output norms */
    float normed[LFM2_D_MODEL];
    trix_rmsnorm(emb, w->attention[0].norm_weight, normed, LFM2_D_MODEL);

    float q[LFM2_D_MODEL];
    float k[LFM2_N_KV_HEADS * LFM2_D_HEAD];
    float v[LFM2_N_KV_HEADS * LFM2_D_HEAD];

    trix_matvec_q4_0(w->attention[0].wq, normed, q, LFM2_D_MODEL, LFM2_D_MODEL);
    trix_matvec_q4_0(w->attention[0].wk, normed, k, LFM2_N_KV_HEADS * LFM2_D_HEAD, LFM2_D_MODEL);
    trix_matvec_q4_0(w->attention[0].wv, normed, v, LFM2_N_KV_HEADS * LFM2_D_HEAD, LFM2_D_MODEL);

    float q_norm = 0, k_norm = 0, v_norm = 0;
    for (int i = 0; i < LFM2_D_MODEL; i++) q_norm += q[i]*q[i];
    for (int i = 0; i < LFM2_N_KV_HEADS * LFM2_D_HEAD; i++) k_norm += k[i]*k[i];
    for (int i = 0; i < LFM2_N_KV_HEADS * LFM2_D_HEAD; i++) v_norm += v[i]*v[i];

    printf("  Attn layer 0 (using raw embedding as input, not correct but diagnostic):\n");
    printf("  Q L2=%.4f  first 4: %.4f %.4f %.4f %.4f\n", sqrtf(q_norm), q[0], q[1], q[2], q[3]);
    printf("  K L2=%.4f  first 4: %.4f %.4f %.4f %.4f\n", sqrtf(k_norm), k[0], k[1], k[2], k[3]);
    printf("  V L2=%.4f  first 4: %.4f %.4f %.4f %.4f\n", sqrtf(v_norm), v[0], v[1], v[2], v[3]);

    /* Test 6: Check QK-norm behavior */
    printf("\n=== QK-Norm Check ===\n");
    printf("  q_norm weights: [%.6f, %.6f, %.6f, %.6f, ...]\n",
           w->attention[0].q_norm[0], w->attention[0].q_norm[1],
           w->attention[0].q_norm[2], w->attention[0].q_norm[3]);
    printf("  k_norm weights: [%.6f, %.6f, %.6f, %.6f, ...]\n",
           w->attention[0].k_norm[0], w->attention[0].k_norm[1],
           w->attention[0].k_norm[2], w->attention[0].k_norm[3]);

    /* Apply QK-norm to first head of Q */
    float q_head0[LFM2_D_HEAD];
    memcpy(q_head0, q, LFM2_D_HEAD * sizeof(float));
    float q_normed[LFM2_D_HEAD];
    trix_rmsnorm(q_head0, w->attention[0].q_norm, q_normed, LFM2_D_HEAD);
    printf("  Q head0 after norm: [%.6f, %.6f, %.6f, %.6f, ...]\n",
           q_normed[0], q_normed[1], q_normed[2], q_normed[3]);

    trix_unload_gguf(model);
    return 0;
}
