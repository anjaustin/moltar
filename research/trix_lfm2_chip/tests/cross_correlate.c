/*
 * cross_correlate — THE definitive experiment.
 *
 * Computes ALL 3072 of our in_proj dot products, then matches each against
 * llama.cpp's 3072 outputs to determine if it's a permutation, wrong weights,
 * or something else.
 */

#include "../include/lfm2_trix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void trix_rmsnorm(const float *x, const float *gamma, float *out, int n);
extern void trix_matvec_q4_0(const q4_0_block_t *W, const float *x, float *y, int M, int K);

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <llama_inproj0.bin>\n", argv[0]);
        return 1;
    }

    /* Load model */
    trix_gguf_model_t *model = trix_load_gguf(argv[1]);
    if (!model) { fprintf(stderr, "FATAL: failed to load model\n"); return 1; }
    const trix_lfm2_weights_t *w = trix_get_weights(model);

    /* Load llama.cpp reference */
    float llama_out[3072];
    FILE *fp = fopen(argv[2], "rb");
    if (!fp) { fprintf(stderr, "FATAL: cannot open %s\n", argv[2]); return 1; }
    size_t nread = fread(llama_out, sizeof(float), 3072, fp);
    fclose(fp);
    if (nread != 3072) { fprintf(stderr, "FATAL: expected 3072 floats, got %zu\n", nread); return 1; }

    printf("Loaded llama.cpp in_proj-0 reference (%zu floats)\n", nread);
    printf("  llama first 4: %.6f %.6f %.6f %.6f\n", llama_out[0], llama_out[1], llama_out[2], llama_out[3]);

    /* Compute our in_proj output */
    float emb[LFM2_D_MODEL];
    trix_embed_token(w, 1, emb);

    float normed[LFM2_D_MODEL];
    trix_rmsnorm(emb, w->shortconv[0].norm_weight, normed, LFM2_D_MODEL);

    float our_out[3072];
    trix_matvec_q4_0(w->shortconv[0].in_proj, normed, our_out, 3072, LFM2_D_MODEL);

    printf("  ours  first 4: %.6f %.6f %.6f %.6f\n\n", our_out[0], our_out[1], our_out[2], our_out[3]);

    /* L2 norms */
    float l2_llama = 0, l2_ours = 0;
    for (int i = 0; i < 3072; i++) { l2_llama += llama_out[i] * llama_out[i]; l2_ours += our_out[i] * our_out[i]; }
    printf("L2 norms: llama=%.6f  ours=%.6f  ratio=%.6f\n\n", sqrtf(l2_llama), sqrtf(l2_ours), sqrtf(l2_ours)/sqrtf(l2_llama));

    /* EXPERIMENT 1: For each llama output[i], find the closest our output[j] */
    printf("=== MATCHING: for each llama[i], find closest ours[j] ===\n");
    int match_count = 0;
    int perm_map[3072];  /* perm_map[i] = j means llama[i] best matches our[j] */

    for (int i = 0; i < 3072; i++) {
        float best_diff = 1e30f;
        int best_j = -1;
        for (int j = 0; j < 3072; j++) {
            float diff = fabsf(llama_out[i] - our_out[j]);
            if (diff < best_diff) {
                best_diff = diff;
                best_j = j;
            }
        }
        perm_map[i] = best_j;
        if (best_diff < 0.01f) match_count++;
        if (i < 16) {
            printf("  llama[%4d]=%.6f  -> ours[%4d]=%.6f  diff=%.2e %s\n",
                   i, llama_out[i], best_j, our_out[best_j], best_diff,
                   best_diff < 0.01f ? "MATCH" : "");
        }
    }
    printf("  ... (%d more)\n", 3072 - 16);
    printf("  Matches within 0.01: %d / 3072 (%.1f%%)\n\n", match_count, 100.0f * match_count / 3072);

    /* Check if perm_map is a bijection (1-to-1) */
    int used[3072];
    memset(used, 0, sizeof(used));
    int collisions = 0;
    for (int i = 0; i < 3072; i++) {
        if (used[perm_map[i]]) collisions++;
        used[perm_map[i]] = 1;
    }
    printf("  Permutation collisions: %d (0 = valid permutation)\n\n", collisions);

    /* EXPERIMENT 2: Check if perm_map reveals a simple pattern */
    printf("=== PERMUTATION PATTERN (first 64) ===\n");
    for (int i = 0; i < 64; i++) {
        printf("  llama[%4d] -> ours[%4d]  (offset=%+5d)  our_row/32=%d  our_row%%32=%d\n",
               i, perm_map[i], perm_map[i] - i, perm_map[i]/32, perm_map[i]%32);
    }

    /* Dump full perm_map to file */
    {
        FILE *pf = fopen("build/perm_map.txt", "w");
        if (pf) {
            for (int i = 0; i < 3072; i++) {
                fprintf(pf, "%d -> %d  (offset=%+d)  blk32=%d,rem32=%d  blk96=%d,rem96=%d  blk1024=%d,rem1024=%d\n",
                        i, perm_map[i], perm_map[i] - i,
                        perm_map[i]/32, perm_map[i]%32,
                        perm_map[i]/96, perm_map[i]%96,
                        perm_map[i]/1024, perm_map[i]%1024);
            }
            fclose(pf);
            printf("\n  Full permutation map written to build/perm_map.txt\n");
        }
    }

    /* EXPERIMENT 2b: Analyze block patterns in perm_map */
    printf("\n=== BLOCK PATTERN ANALYSIS ===\n");
    /* For each block size, check if perm_map maps blocks to blocks */
    int block_sizes[] = {32, 96, 128, 256, 512, 1024, 0};
    for (int bi = 0; block_sizes[bi]; bi++) {
        int bs = block_sizes[bi];
        int n_blocks_out = 3072 / bs;
        if (3072 % bs != 0) continue;

        /* For each input block, find which output block it maps to */
        int block_consistent = 0;
        int block_inconsistent = 0;
        for (int blk = 0; blk < n_blocks_out; blk++) {
            int target_blk = perm_map[blk * bs] / bs;
            int consistent = 1;
            for (int off = 1; off < bs && off < 3072 - blk * bs; off++) {
                if (perm_map[blk * bs + off] / bs != target_blk) {
                    consistent = 0;
                    break;
                }
            }
            if (consistent) block_consistent++;
            else block_inconsistent++;
        }
        printf("  block_size=%4d: %d/%d blocks consistent (%.1f%%)\n",
               bs, block_consistent, n_blocks_out, 100.0f * block_consistent / n_blocks_out);
    }

    /* EXPERIMENT 2c: Check if the permutation is the identity within blocks of 1024 */
    printf("\n=== PER-CHUNK (1024) ANALYSIS ===\n");
    for (int chunk = 0; chunk < 3; chunk++) {
        int base = chunk * 1024;
        int min_target = 3072, max_target = 0;
        for (int i = base; i < base + 1024 && i < 3072; i++) {
            if (perm_map[i] < min_target) min_target = perm_map[i];
            if (perm_map[i] > max_target) max_target = perm_map[i];
        }
        printf("  llama[%d..%d] maps to ours[%d..%d] (range=%d, expected 1024)\n",
               base, base + 1023, min_target, max_target, max_target - min_target + 1);
    }

    /* EXPERIMENT 3: Element-wise diff (no permutation) */
    float sum_sq_diff = 0;
    for (int i = 0; i < 3072; i++) {
        float d = llama_out[i] - our_out[i];
        sum_sq_diff += d * d;
    }
    printf("\n=== DIRECT COMPARISON (no permutation) ===\n");
    printf("  RMS diff: %.6f\n", sqrtf(sum_sq_diff / 3072));
    printf("  If close to 0: outputs match directly (no permutation needed)\n");

    /* EXPERIMENT 4: Check if our output is a shifted version */
    printf("\n=== CIRCULAR SHIFT SEARCH ===\n");
    float best_corr = -1e30f;
    int best_shift = 0;
    for (int shift = 0; shift < 3072; shift++) {
        float corr = 0;
        for (int i = 0; i < 3072; i++) {
            corr += llama_out[i] * our_out[(i + shift) % 3072];
        }
        if (corr > best_corr) {
            best_corr = corr;
            best_shift = shift;
        }
    }
    printf("  Best circular shift: %d  correlation: %.6f\n", best_shift, best_corr);
    printf("  Self-correlation (llama with llama): ");
    float self_corr = 0;
    for (int i = 0; i < 3072; i++) self_corr += llama_out[i] * llama_out[i];
    printf("%.6f\n", self_corr);

    /* EXPERIMENT 5: Column-major block read.
     * Hypothesis: blocks in GGUF are stored column-major for this tensor.
     * Actual row r in the weight matrix has blocks at file positions:
     *   [j * 3072 + r for j in range(32)]
     * i.e., blocks 3072 apart, not contiguous.
     */
    printf("\n=== COLUMN-MAJOR MATVEC ===\n");
    {
        const q4_0_block_t *in_proj = w->shortconv[0].in_proj;
        const int M = 3072;
        const int K = LFM2_D_MODEL;
        const int nblocks_per_row = K / Q4_0_BLOCK_SIZE; /* 32 */

        float col_major_out[3072];
        for (int row = 0; row < M; row++) {
            float sum = 0.0f;
            for (int b = 0; b < nblocks_per_row; b++) {
                /* Column-major: block b of row 'row' is at file position b*M + row */
                const q4_0_block_t *blk = &in_proj[b * M + row];
                float scale = trix_fp16_to_f32(blk->scale_f16);
                const float *xb = normed + b * Q4_0_BLOCK_SIZE;
                for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                    uint8_t packed = blk->qs[j];
                    float v0 = ((float)(packed & 0x0F) - 8.0f) * scale;
                    float v1 = ((float)(packed >> 4)   - 8.0f) * scale;
                    sum += v0 * xb[2*j] + v1 * xb[2*j + 1];
                }
            }
            col_major_out[row] = sum;
        }

        printf("  col-major first 8: ");
        for (int i = 0; i < 8; i++) printf("%.6f ", col_major_out[i]);
        printf("\n  llama     first 8: ");
        for (int i = 0; i < 8; i++) printf("%.6f ", llama_out[i]);
        printf("\n");

        /* Check how many match */
        int cm_matches = 0;
        float cm_diff_sq = 0;
        for (int i = 0; i < 3072; i++) {
            float d = col_major_out[i] - llama_out[i];
            cm_diff_sq += d * d;
            if (fabsf(d) < 0.01f) cm_matches++;
        }
        printf("  RMS diff: %.6f\n", sqrtf(cm_diff_sq / 3072));
        printf("  Matches within 0.01: %d / 3072 (%.1f%%)\n", cm_matches, 100.0f * cm_matches / 3072);

        float cm_l2 = 0;
        for (int i = 0; i < 3072; i++) cm_l2 += col_major_out[i] * col_major_out[i];
        printf("  L2: %.6f (llama: %.6f)\n", sqrtf(cm_l2), sqrtf(l2_llama));
    }

    /* EXPERIMENT 6: Verify the match rate is not spurious.
     * Compare with RANDOM permutation of our outputs. If random permutation
     * also gives high match rate, then our 98.4% is meaningless. */
    printf("\n=== RANDOM PERMUTATION CONTROL ===\n");
    {
        /* Create a random permutation of our_out */
        float shuffled[3072];
        memcpy(shuffled, our_out, 3072 * sizeof(float));
        /* Simple Fisher-Yates shuffle with fixed seed */
        unsigned seed = 42;
        for (int i = 3071; i > 0; i--) {
            seed = seed * 1103515245 + 12345;
            int j = (seed >> 16) % (i + 1);
            float tmp = shuffled[i]; shuffled[i] = shuffled[j]; shuffled[j] = tmp;
        }

        int random_matches = 0;
        for (int i = 0; i < 3072; i++) {
            float best_diff = 1e30f;
            for (int j = 0; j < 3072; j++) {
                float diff = fabsf(llama_out[i] - shuffled[j]);
                if (diff < best_diff) best_diff = diff;
            }
            if (best_diff < 0.01f) random_matches++;
        }
        printf("  Shuffled match rate within 0.01: %d / 3072 (%.1f%%)\n", random_matches, 100.0f * random_matches / 3072);
        printf("  (If similar to 98.4%%, the original match is spurious)\n");
    }

    /* EXPERIMENT 7: Try much tighter tolerance */
    printf("\n=== TIGHTER TOLERANCE MATCHING ===\n");
    {
        float tolerances[] = {0.01f, 0.001f, 0.0001f, 0.00001f, 0.0f};
        for (int ti = 0; tolerances[ti] >= 0; ti++) {
            float tol = tolerances[ti];
            int matches = 0;
            for (int i = 0; i < 3072; i++) {
                float best_diff = 1e30f;
                for (int j = 0; j < 3072; j++) {
                    float diff = fabsf(llama_out[i] - our_out[j]);
                    if (diff < best_diff) best_diff = diff;
                }
                if (best_diff <= tol) matches++;
            }
            printf("  tol=%.1e: %d / 3072 (%.1f%%)\n", tol, matches, 100.0f * matches / 3072);
            if (tol == 0.0f) break;
        }
    }

    /* EXPERIMENT 8: Direct row-by-row weight comparison.
     * Dequantize the first few rows of in_proj using our loader,
     * compute dot products manually, verify they match our_out exactly. */
    printf("\n=== DIRECT ROW DEQUANT VERIFICATION ===\n");
    {
        const q4_0_block_t *in_proj = w->shortconv[0].in_proj;
        const int K = LFM2_D_MODEL;
        const int nblocks_per_row = K / Q4_0_BLOCK_SIZE;

        for (int row = 0; row < 4; row++) {
            /* Dequantize row manually */
            float dequant[1024];
            const q4_0_block_t *row_ptr = in_proj + row * nblocks_per_row;
            for (int b = 0; b < nblocks_per_row; b++) {
                float scale = trix_fp16_to_f32(row_ptr[b].scale_f16);
                for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                    uint8_t packed = row_ptr[b].qs[j];
                    dequant[b * Q4_0_BLOCK_SIZE + 2*j]     = ((float)(packed & 0x0F) - 8.0f) * scale;
                    dequant[b * Q4_0_BLOCK_SIZE + 2*j + 1] = ((float)(packed >> 4) - 8.0f) * scale;
                }
            }

            /* Compute dot product */
            float dot = 0;
            for (int k = 0; k < K; k++) dot += dequant[k] * normed[k];

            printf("  row %d: manual_dot=%.6f  our_out[%d]=%.6f  diff=%.2e\n",
                   row, dot, row, our_out[row], fabsf(dot - our_out[row]));

            /* Also print first 4 dequantized weights of each row */
            printf("    first 4 weights: %.6f %.6f %.6f %.6f\n",
                   dequant[0], dequant[1], dequant[2], dequant[3]);
        }
    }

    /* EXPERIMENT 9: Compute what llama.cpp's row 0 weights should be.
     * llama_out[0] = dot(W_row_0, normed). We know normed exactly.
     * If we could find which of our rows gives llama_out[0], that tells us the mapping. */
    printf("\n=== WHICH OF OUR ROWS GIVES LLAMA OUTPUT[0]? ===\n");
    {
        float target = llama_out[0]; /* -0.000834 */
        float best_diff = 1e30f;
        int best_row = -1;
        for (int row = 0; row < 3072; row++) {
            float diff = fabsf(our_out[row] - target);
            if (diff < best_diff) {
                best_diff = diff;
                best_row = row;
            }
        }
        printf("  llama_out[0] = %.6f\n", target);
        printf("  Closest our_out: row %d = %.6f (diff=%.2e)\n", best_row, our_out[best_row], best_diff);

        /* Now: if these are truly the same dot product, the raw weight bytes for
         * llama's row 0 should be the same as our row best_row.
         * But wait — maybe they're NOT the same, and 98.4% is spurious due to
         * similar distributions. Let's check. */
    }

    trix_unload_gguf(model);
    return 0;
}
