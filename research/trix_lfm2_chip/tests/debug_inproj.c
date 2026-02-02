/*
 * debug_inproj — Focused test: manually compute in_proj row 0 dot product
 * and compare against our trix_matvec_q4_0.
 *
 * This will tell us if the Q4_0 dequant + matvec is correct.
 */

#include "../include/lfm2_trix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void trix_rmsnorm(const float *x, const float *gamma, float *out, int n);
extern void trix_matvec_q4_0(const q4_0_block_t *W, const float *x, float *y, int M, int K);

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }

    trix_gguf_model_t *model = trix_load_gguf(argv[1]);
    if (!model) { fprintf(stderr, "FATAL: failed to load model\n"); return 1; }
    const trix_lfm2_weights_t *w = trix_get_weights(model);

    /* Step 1: Get embedding and norm */
    float emb[LFM2_D_MODEL];
    trix_embed_token(w, 1, emb);

    float normed[LFM2_D_MODEL];
    trix_rmsnorm(emb, w->shortconv[0].norm_weight, normed, LFM2_D_MODEL);

    printf("normed first 8: ");
    for (int i = 0; i < 8; i++) printf("%.6f ", normed[i]);
    printf("\n\n");

    /* Step 1b: Verify tensor offset */
    printf("\nVerifying tensor pointer...\n");
    /* Get the mmap base from the model by checking embedding position */
    /* We know token_embd offset should be 0 for the first tensor */

    /* Step 2: Our matvec */
    float bcx[3 * LFM2_D_MODEL];
    trix_matvec_q4_0(w->shortconv[0].in_proj, normed, bcx, 3 * LFM2_D_MODEL, LFM2_D_MODEL);
    printf("Our matvec first 8: ");
    for (int i = 0; i < 8; i++) printf("%.6f ", bcx[i]);
    printf("\n");

    /* Step 3: Manual row-0 dot product */
    const q4_0_block_t *in_proj = w->shortconv[0].in_proj;
    const int n_blocks = LFM2_D_MODEL / Q4_0_BLOCK_SIZE; /* 32 */

    /* Dequantize row 0 fully */
    float row0_dequant[LFM2_D_MODEL];
    for (int b = 0; b < n_blocks; b++) {
        const q4_0_block_t *blk = &in_proj[b]; /* row 0, block b */
        float scale = trix_fp16_to_f32(blk->scale_f16);
        for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
            uint8_t packed = blk->qs[j];
            row0_dequant[b * Q4_0_BLOCK_SIZE + 2*j]     = ((float)(packed & 0x0F) - 8.0f) * scale;
            row0_dequant[b * Q4_0_BLOCK_SIZE + 2*j + 1]  = ((float)(packed >> 4)   - 8.0f) * scale;
        }
    }

    printf("\nRow 0 dequant first 8: ");
    for (int i = 0; i < 8; i++) printf("%.6f ", row0_dequant[i]);
    printf("\n");

    float manual_dot = 0.0f;
    for (int k = 0; k < LFM2_D_MODEL; k++) {
        manual_dot += row0_dequant[k] * normed[k];
    }
    printf("Manual dot product row 0: %.6f\n", manual_dot);
    printf("Our bcx[0]:               %.6f\n", bcx[0]);

    /* Step 4: Check a few more rows */
    printf("\nManual vs matvec for first 8 outputs:\n");
    for (int row = 0; row < 8; row++) {
        float dequant[LFM2_D_MODEL];
        for (int b = 0; b < n_blocks; b++) {
            const q4_0_block_t *blk = &in_proj[row * n_blocks + b];
            float scale = trix_fp16_to_f32(blk->scale_f16);
            for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                uint8_t packed = blk->qs[j];
                dequant[b * Q4_0_BLOCK_SIZE + 2*j]     = ((float)(packed & 0x0F) - 8.0f) * scale;
                dequant[b * Q4_0_BLOCK_SIZE + 2*j + 1]  = ((float)(packed >> 4)   - 8.0f) * scale;
            }
        }
        float dot = 0.0f;
        for (int k = 0; k < LFM2_D_MODEL; k++) dot += dequant[k] * normed[k];
        printf("  [%d] manual=%.6f  matvec=%.6f  diff=%.2e\n", row, dot, bcx[row], dot - bcx[row]);
    }

    /* Step 5: Print raw block data for row 0 first block */
    printf("\nRaw in_proj block 0 (row 0, first 32 weights):\n");
    printf("  scale_f16 = 0x%04x  (%.6f)\n",
           in_proj[0].scale_f16, trix_fp16_to_f32(in_proj[0].scale_f16));
    printf("  qs bytes: ");
    for (int j = 0; j < 16; j++) printf("%02x ", in_proj[0].qs[j]);
    printf("\n");

    /* Step 6: What llama.cpp expects for in_proj[0] */
    printf("\nExpected (llama.cpp) in_proj-0 first 8: -0.000834 0.124981 0.267852 -0.535283 0.069673 -0.038454 0.018493 0.399625\n");
    printf("Our bcx first 8:                       ");
    for (int i = 0; i < 8; i++) printf("%.6f ", bcx[i]);
    printf("\n");

    /* Step 7: L2 norms for comparison */
    float l2_our = 0, l2_manual = 0;
    for (int i = 0; i < 3 * LFM2_D_MODEL; i++) l2_our += bcx[i] * bcx[i];
    printf("\nOur full in_proj L2 = %.6f (expected: 15.786986)\n", sqrtf(l2_our));

    /* Step 8: Try TRANSPOSED interpretation
     * What if llama.cpp's mul_mat interprets the weight as transposed?
     * i.e., what if output[i] = dot(column_i, x) instead of dot(row_i, x)?
     * For Q4_0 with ne[0]=1024, ne[1]=3072:
     *   Row-major: row_i = blocks[i*32 .. i*32+31]
     *   Column-major: col_i = blocks[(i%32)*3072 + (i/32)] -- doesn't make sense for Q4_0
     *
     * Actually, let's try: what if the in_proj weight shape ne[0]=1024 means
     * there are 1024 output rows (not 3072)?
     * GGML mul_mat: output_ne[0] = src1->ne[1] = 3072, wait no...
     *
     * Let me try computing with M=1024, K=3072 (swapped) to see if that matches
     */
    printf("\n=== TESTING ALTERNATIVE INTERPRETATIONS ===\n");

    /* Alt 1: Compute column-wise dot products */
    /* For ne[0]=1024, ne[1]=3072 in GGML, each "row" has ne[0]=1024 elements.
     * There are ne[1]=3072 such rows.
     * In mul_mat(W,x): output[i] = dot(W_row_i, x) for i=0..ne[1]-1=3071
     * That's what we already do. So let's try the opposite... */

    /* Alt 2: What if the weight is actually {3072, 1024} (ne[0]=3072)?
     * Then each row would have 3072 elements = 96 Q4_0 blocks.
     * And there would be 1024 rows -> 1024 outputs.
     * But the GGUF says ne[0]=1024, ne[1]=3072. So this shouldn't apply. */

    /* Alt 3: Verify by matching llama.cpp's first output value (-0.000834).
     * Search which row of the weight matrix produces this value when dotted with normed. */
    printf("\nSearching for row that produces -0.000834...\n");
    float target = -0.000834f;
    float best_diff = 1e30f;
    int best_row = -1;
    for (int row = 0; row < 100; row++) {  /* check first 100 rows */
        float dequant[LFM2_D_MODEL];
        for (int b = 0; b < n_blocks; b++) {
            const q4_0_block_t *blk = &in_proj[row * n_blocks + b];
            float scale = trix_fp16_to_f32(blk->scale_f16);
            for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                uint8_t packed = blk->qs[j];
                dequant[b * Q4_0_BLOCK_SIZE + 2*j]     = ((float)(packed & 0x0F) - 8.0f) * scale;
                dequant[b * Q4_0_BLOCK_SIZE + 2*j + 1]  = ((float)(packed >> 4)   - 8.0f) * scale;
            }
        }
        float dot = 0.0f;
        for (int k = 0; k < LFM2_D_MODEL; k++) dot += dequant[k] * normed[k];
        float diff = fabsf(dot - target);
        if (diff < best_diff) {
            best_diff = diff;
            best_row = row;
        }
        if (diff < 0.001f) {
            printf("  Row %d produces %.6f (diff=%.2e) *** MATCH\n", row, dot, diff);
        }
    }
    printf("  Best match in first 100 rows: row %d (diff=%.2e)\n", best_row, best_diff);

    /* Also search rows 1024..1124 and 2048..2148 */
    for (int start = 1024; start <= 2048; start += 1024) {
        best_diff = 1e30f; best_row = -1;
        for (int row = start; row < start + 100 && row < 3072; row++) {
            float dequant[LFM2_D_MODEL];
            for (int b = 0; b < n_blocks; b++) {
                const q4_0_block_t *blk = &in_proj[row * n_blocks + b];
                float scale = trix_fp16_to_f32(blk->scale_f16);
                for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                    uint8_t packed = blk->qs[j];
                    dequant[b * Q4_0_BLOCK_SIZE + 2*j]     = ((float)(packed & 0x0F) - 8.0f) * scale;
                    dequant[b * Q4_0_BLOCK_SIZE + 2*j + 1]  = ((float)(packed >> 4)   - 8.0f) * scale;
                }
            }
            float dot = 0.0f;
            for (int k = 0; k < LFM2_D_MODEL; k++) dot += dequant[k] * normed[k];
            float diff = fabsf(dot - target);
            if (diff < best_diff) {
                best_diff = diff;
                best_row = row;
            }
            if (diff < 0.001f) {
                printf("  Row %d produces %.6f (diff=%.2e) *** MATCH\n", row, dot, diff);
            }
        }
        printf("  Best match in rows %d..%d: row %d (diff=%.2e)\n",
               start, start+99, best_row, best_diff);
    }

    /* Step 8b: Try Q4_0 x Q8_0 style dot product.
     * First quantize normed[] to Q8_0, then compute integer dot products. */
    {
        printf("\n=== Q4_0 x Q8_0 DOT PRODUCT (mimicking llama.cpp) ===\n");

        /* Q8_0 block: 32 values, { float16 d; int8_t qs[32]; }
         * Quantize: d = max(abs(x[i])) / 127, qs[i] = round(x[i] / d) */
        typedef struct { float d; int8_t qs[32]; } q8_0_block_t;

        const int q8_blocks = LFM2_D_MODEL / 32;
        q8_0_block_t q8_normed[q8_blocks];

        for (int b = 0; b < q8_blocks; b++) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = fabsf(normed[b * 32 + j]);
                if (a > amax) amax = a;
            }
            float d = amax / 127.0f;
            q8_normed[b].d = d;
            float id = d > 0 ? 1.0f / d : 0.0f;
            for (int j = 0; j < 32; j++) {
                float v = normed[b * 32 + j] * id;
                q8_normed[b].qs[j] = (int8_t)(v > 0 ? v + 0.5f : v - 0.5f);
            }
        }

        /* Verify Q8_0 quantization accuracy */
        printf("Q8_0 reconst first 8: ");
        for (int i = 0; i < 8; i++) {
            int b = i / 32;
            int j = i % 32;
            printf("%.6f ", q8_normed[b].d * q8_normed[b].qs[j]);
        }
        printf("\n");
        printf("Original    first 8: ");
        for (int i = 0; i < 8; i++) printf("%.6f ", normed[i]);
        printf("\n");

        /* Now compute Q4_0 x Q8_0 dot product for first 8 rows */
        printf("\nQ4_0 x Q8_0 matvec for first 8 outputs:\n");
        for (int row = 0; row < 8; row++) {
            float sum = 0.0f;
            for (int b = 0; b < n_blocks; b++) {
                const q4_0_block_t *w_blk = &in_proj[row * n_blocks + b];
                float w_scale = trix_fp16_to_f32(w_blk->scale_f16);

                /* Integer dot product for this block pair */
                int32_t isum = 0;
                for (int j = 0; j < 16; j++) {
                    uint8_t packed = w_blk->qs[j];
                    int8_t w0 = (int8_t)(packed & 0x0F) - 8;
                    int8_t w1 = (int8_t)(packed >> 4)    - 8;
                    isum += (int32_t)w0 * q8_normed[b].qs[2*j];
                    isum += (int32_t)w1 * q8_normed[b].qs[2*j + 1];
                }
                sum += w_scale * q8_normed[b].d * (float)isum;
            }
            printf("  [%d] q4xq8=%.6f  our_f32=%.6f  llama=", row, sum, bcx[row]);
            /* llama.cpp values for reference */
            const float llama_ref[] = {-0.000834f, 0.124981f, 0.267852f, -0.535283f,
                                       0.069673f, -0.038454f, 0.018493f, 0.399625f};
            printf("%.6f  diff_q8=%.2e\n", llama_ref[row], sum - llama_ref[row]);
        }
    }

    /* Step 9: What about the q4_0_4x8 repacking?
     * The repacking interleaves 4 rows. So rows [0,1,2,3] are interleaved.
     * Let me check if maybe the repacking changes which block feeds which output.
     *
     * Actually, let me try a completely different theory:
     * What if GGML stores this transposed relative to what we think?
     * i.e., the tensor stores {ne[0]=1024, ne[1]=3072} but the data layout
     * is actually 1024 groups of 3072 elements (instead of 3072 groups of 1024)?
     *
     * Let me compute output[0] = sum over k=0..1023 of W[k * 3072 + 0] * normed[k]
     * This would be a column-wise dot product.
     */
    printf("\n=== TRYING COLUMN-WISE DOT PRODUCT ===\n");
    printf("(i.e., treating weight as 1024 rows of 3072 elements)\n");

    /* Total Q4_0 blocks in in_proj = (1024 * 3072) / 32 = 98304 */
    /* If we think of it as 1024 rows of 3072 elements = 96 blocks per row */
    const int alt_rows = LFM2_D_MODEL;       /* 1024 */
    const int alt_cols = 3 * LFM2_D_MODEL;   /* 3072 */
    const int alt_blocks_per_row = alt_cols / Q4_0_BLOCK_SIZE; /* 96 */

    /* Dequantize the first "alt row" (3072 elements) */
    float alt_row0[3 * LFM2_D_MODEL];
    for (int b = 0; b < alt_blocks_per_row; b++) {
        const q4_0_block_t *blk = &in_proj[b];
        float scale = trix_fp16_to_f32(blk->scale_f16);
        for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
            uint8_t packed = blk->qs[j];
            alt_row0[b * Q4_0_BLOCK_SIZE + 2*j]     = ((float)(packed & 0x0F) - 8.0f) * scale;
            alt_row0[b * Q4_0_BLOCK_SIZE + 2*j + 1]  = ((float)(packed >> 4)   - 8.0f) * scale;
        }
    }
    printf("Alt row 0 first 8 (3072 el): ");
    for (int i = 0; i < 8; i++) printf("%.6f ", alt_row0[i]);
    printf("\n");

    /* Now compute column-wise: for output j (j=0..3071):
     * out[j] = sum_{k=0..1023} dequant(W[k * alt_blocks_per_row + j/32], j%32) * normed[k]
     * This is expensive but let's check first 8 outputs */
    printf("\nColumn-wise outputs (first 8):\n");
    for (int j = 0; j < 8; j++) {
        float sum = 0.0f;
        int blk_in_row = j / Q4_0_BLOCK_SIZE;   /* which block within a 3072-element row */
        int elem_in_blk = j % Q4_0_BLOCK_SIZE;  /* which element within that block */

        for (int k = 0; k < alt_rows; k++) {
            /* Block index in the linear array */
            const q4_0_block_t *blk = &in_proj[k * alt_blocks_per_row + blk_in_row];
            float scale = trix_fp16_to_f32(blk->scale_f16);
            int byte_idx = elem_in_blk / 2;
            uint8_t packed = blk->qs[byte_idx];
            float val;
            if (elem_in_blk % 2 == 0) {
                val = ((float)(packed & 0x0F) - 8.0f) * scale;
            } else {
                val = ((float)(packed >> 4)   - 8.0f) * scale;
            }
            sum += val * normed[k];
        }
        printf("  col[%d] = %.6f\n", j, sum);
    }

    trix_unload_gguf(model);
    return 0;
}
