/*
 * compare_weights — The definitive weight comparison.
 *
 * Loads the GGUF file via our loader, manually reads the same bytes from
 * the GGUF file at the expected offset, and verifies they match.
 *
 * Then: dequantizes row 0 and computes its dot product with the normed
 * embedding vector. If the dot product matches our_out[0], our loader is correct.
 * If it doesn't match llama.cpp's output[0], the problem is elsewhere.
 *
 * KEY INSIGHT: llama.cpp repacks Q4_0 into q4_0_4x8 on Apple Silicon.
 * The repacked format interleaves 4 rows of Q4_0 blocks.
 * But the REPACK uses different kernel code — we can't just read the raw
 * repacked data and expect Q4_0 block format.
 *
 * The ORIGINAL (non-repacked) data should match our GGUF mmap exactly
 * since both load from the same file bytes.
 *
 * So the question is: does our dequant(row0_Q4_0) dot normed give -0.295
 * (our answer) or -0.000834 (llama.cpp's answer)?
 *
 * If our dequant gives -0.295 and llama.cpp gives -0.000834 from the SAME
 * bytes, then the Q4_0->Q8_0 quantization of the input OR the Q4_0_4x8
 * repacking is changing the semantics. But that would affect ALL Q4_0
 * models on Apple Silicon — which clearly work — so it's wrong.
 *
 * The only remaining hypothesis: we're pointing to the WRONG OFFSET in the file.
 */

#include "../include/lfm2_trix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

extern void trix_rmsnorm(const float *x, const float *gamma, float *out, int n);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    /* Load via our GGUF loader */
    trix_gguf_model_t *model = trix_load_gguf(argv[1]);
    if (!model) { fprintf(stderr, "FATAL: failed to load model\n"); return 1; }
    const trix_lfm2_weights_t *w = trix_get_weights(model);

    /* Also mmap the file independently */
    int fd = open(argv[1], O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    const uint8_t *file_data = (const uint8_t *)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    /* Get our pointer to in_proj weight */
    const uint8_t *our_inproj = (const uint8_t *)w->shortconv[0].in_proj;

    /* Compute offset of our in_proj pointer into the file */
    ptrdiff_t our_offset = our_inproj - file_data;
    printf("=== POINTER ANALYSIS ===\n");
    printf("  file_data (mmap base):     %p\n", (void*)file_data);
    printf("  in_proj pointer:           %p\n", (void*)our_inproj);
    printf("  offset from file start:    %td (0x%tx)\n", our_offset, our_offset);
    printf("  file size:                 %lld\n", (long long)st.st_size);

    /* The expected offset from dump_gguf_meta should be data_offset + tensor_offset */
    /* For in_proj: data_offset=2383808, tensor_offset=63037440 → abs=65421248 */
    printf("  Expected offset (from notes): 65421248 (0x3E5A800)\n");
    printf("  Difference: %td\n", our_offset - 65421248);

    /* Verify first 36 bytes (2 Q4_0 blocks) match between our loader and raw file */
    printf("\n=== RAW BYTE COMPARISON (first 36 bytes) ===\n");
    const uint8_t *raw_at_expected = file_data + 65421248;
    printf("  From our loader:   ");
    for (int i = 0; i < 36; i++) printf("%02x ", our_inproj[i]);
    printf("\n  From raw offset:   ");
    for (int i = 0; i < 36; i++) printf("%02x ", raw_at_expected[i]);
    printf("\n  Match: %s\n",
           memcmp(our_inproj, raw_at_expected, 36) == 0 ? "YES" : "NO");

    /* If our_offset != 65421248, also show what's at our_offset in the raw file */
    if (our_offset != 65421248) {
        printf("\n  From raw at OUR offset: ");
        for (int i = 0; i < 36; i++) printf("%02x ", file_data[our_offset + i]);
        printf("\n");
    }

    /* Compute embedding and normed vector */
    float emb[LFM2_D_MODEL];
    trix_embed_token(w, 1, emb);

    float normed[LFM2_D_MODEL];
    trix_rmsnorm(emb, w->shortconv[0].norm_weight, normed, LFM2_D_MODEL);
    printf("\n=== NORMED VECTOR (first 8) ===\n  ");
    for (int i = 0; i < 8; i++) printf("%.6f ", normed[i]);
    printf("\n");

    /* Dequantize row 0 of in_proj and compute dot product */
    printf("\n=== ROW 0 DEQUANT + DOT PRODUCT ===\n");
    {
        const q4_0_block_t *row0 = w->shortconv[0].in_proj;
        const int nblk = LFM2_D_MODEL / Q4_0_BLOCK_SIZE; /* 32 */
        float dot = 0;
        printf("  Block 0: scale_f16=0x%04x, scale=%.6f\n",
               row0[0].scale_f16, trix_fp16_to_f32(row0[0].scale_f16));
        printf("  Block 0 qs: ");
        for (int i = 0; i < 16; i++) printf("%02x ", row0[0].qs[i]);
        printf("\n");

        for (int b = 0; b < nblk; b++) {
            float scale = trix_fp16_to_f32(row0[b].scale_f16);
            for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                uint8_t packed = row0[b].qs[j];
                float v0 = ((float)(packed & 0x0F) - 8.0f) * scale;
                float v1 = ((float)(packed >> 4) - 8.0f) * scale;
                dot += v0 * normed[b * Q4_0_BLOCK_SIZE + 2*j];
                dot += v1 * normed[b * Q4_0_BLOCK_SIZE + 2*j + 1];
            }
        }
        printf("  Dot product of row 0 with normed: %.6f\n", dot);
        printf("  Expected (our matvec):    -0.295303\n");
        printf("  Expected (llama.cpp):     -0.000834\n");
    }

    /* Now try: what if the tensor at the GGUF offset is actually stored
     * in COLUMN-major order? i.e., the first 3072 blocks are column 0
     * of the weight matrix (all rows, first block position). */
    printf("\n=== HYPOTHESIS: What if ne[0]=1024 means 1024 ROWS, not 1024 columns? ===\n");
    printf("  i.e., the tensor shape [1024, 3072] means 1024 rows x 3072 cols\n");
    printf("  Then each row has 3072 elements = 96 Q4_0 blocks\n");
    printf("  And there are 1024 rows = 1024 * 96 = 98304 blocks total (same)\n");
    {
        const q4_0_block_t *in_proj = w->shortconv[0].in_proj;
        /* If ne[0]=1024 means rows, then W is 1024x3072
         * and mul_mat(W, x) gives output of size 1024 (not 3072).
         * But llama.cpp gives 3072 output. So this interpretation is wrong.
         * UNLESS: mul_mat transposes, and we need W^T * x. */
    }

    /* CRITICAL NEW HYPOTHESIS: What if GGML's ne[0]=1024, ne[1]=3072 means
     * the tensor is stored as 1024 "rows" of 3072 elements each?
     *
     * In memory: row 0 = elements [0..3071], row 1 = elements [3072..6143], etc.
     * Total elements = 1024 * 3072 = 3,145,728.
     *
     * Then ggml_mul_mat(W{ne0=1024,ne1=3072}, x{ne0=1024,ne1=1}):
     *   Output has ne0 = ne01 = 3072 (as we observe).
     *   Output[i] = dot(W[*, i], x) = dot(column i of W, x)
     *   i.e., it multiplies x by each COLUMN of W.
     *
     * But wait — GGML iterates: src0_row + ir0 * nb01
     * where ir0 ranges from 0 to ne01-1 = 3071.
     * nb01 = the stride between "rows" in dimension 1.
     *
     * GGML ne convention:
     *   ne[0] = innermost (fastest-varying) dimension
     *   ne[1] = next dimension
     *
     * For mul_mat: output[i] = dot product along dimension 0 of src0[i, :], and src1[:, 0]
     *
     * So "src0 row i" means: fix dimension 1 index = i, iterate over dimension 0.
     * In memory: src0_data + i * nb01, with nb01 = ne[0] * type_size (for quantized: ne[0]/block_size * block_bytes)
     *
     * For ne[0]=1024: each "row" (column of the mathematical matrix) has 1024 elements = 32 Q4_0 blocks.
     * The dot product computes output[i] = dot(W_row_i, x) where row_i has 1024 elements.
     *
     * This is EXACTLY what we do. So why is the output different?
     *
     * THE ONLY EXPLANATION: We are pointing to the wrong tensor data!
     * OR: our in_proj pointer is NOT where we think it is in the file.
     */

    /* Let me verify by computing the dot product using raw file bytes at offset 65421248 */
    printf("\n=== DOT PRODUCT FROM RAW FILE BYTES AT EXPECTED OFFSET ===\n");
    {
        const q4_0_block_t *raw_row0 = (const q4_0_block_t *)(file_data + 65421248);
        printf("  Block 0: scale_f16=0x%04x, scale=%.6f\n",
               raw_row0[0].scale_f16, trix_fp16_to_f32(raw_row0[0].scale_f16));
        printf("  Block 0 qs: ");
        for (int i = 0; i < 16; i++) printf("%02x ", raw_row0[0].qs[i]);
        printf("\n");

        float dot = 0;
        for (int b = 0; b < 32; b++) {
            float scale = trix_fp16_to_f32(raw_row0[b].scale_f16);
            for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                uint8_t packed = raw_row0[b].qs[j];
                float v0 = ((float)(packed & 0x0F) - 8.0f) * scale;
                float v1 = ((float)(packed >> 4) - 8.0f) * scale;
                dot += v0 * normed[b * Q4_0_BLOCK_SIZE + 2*j];
                dot += v1 * normed[b * Q4_0_BLOCK_SIZE + 2*j + 1];
            }
        }
        printf("  Dot product: %.6f\n", dot);
    }

    /* Let me also scan the ENTIRE tensor data for the first few blocks that,
     * when dotted with normed, give approximately -0.000834 (llama.cpp row 0 output) */
    printf("\n=== SCANNING FOR ROW THAT GIVES llama_out[0] = -0.000834 ===\n");
    {
        const q4_0_block_t *base = (const q4_0_block_t *)(file_data + 65421248);
        int total_blocks = 98304; /* 3072 * 32 */
        float target = -0.000834f;
        float best_diff = 1e30f;
        int best_row = -1;

        for (int row = 0; row < 3072; row++) {
            const q4_0_block_t *row_ptr = base + row * 32;
            float dot = 0;
            for (int b = 0; b < 32; b++) {
                float scale = trix_fp16_to_f32(row_ptr[b].scale_f16);
                for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                    uint8_t packed = row_ptr[b].qs[j];
                    float v0 = ((float)(packed & 0x0F) - 8.0f) * scale;
                    float v1 = ((float)(packed >> 4) - 8.0f) * scale;
                    dot += v0 * normed[b * Q4_0_BLOCK_SIZE + 2*j];
                    dot += v1 * normed[b * Q4_0_BLOCK_SIZE + 2*j + 1];
                }
            }
            float diff = fabsf(dot - target);
            if (diff < best_diff) {
                best_diff = diff;
                best_row = row;
            }
        }
        printf("  Best matching row: %d (dot=%.6f, diff=%.2e)\n",
               best_row, -1.0f, best_diff); /* placeholder, recompute */

        /* Recompute best row's dot product */
        if (best_row >= 0) {
            const q4_0_block_t *row_ptr = base + best_row * 32;
            float dot = 0;
            for (int b = 0; b < 32; b++) {
                float scale = trix_fp16_to_f32(row_ptr[b].scale_f16);
                for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                    uint8_t packed = row_ptr[b].qs[j];
                    float v0 = ((float)(packed & 0x0F) - 8.0f) * scale;
                    float v1 = ((float)(packed >> 4) - 8.0f) * scale;
                    dot += v0 * normed[b * Q4_0_BLOCK_SIZE + 2*j];
                    dot += v1 * normed[b * Q4_0_BLOCK_SIZE + 2*j + 1];
                }
            }
            printf("  Row %d: dot=%.6f (target=%.6f, diff=%.2e)\n",
                   best_row, dot, target, fabsf(dot - target));
        }
    }

    /* ALSO: What if llama.cpp quantizes the input to Q8_0 before the dot product?
     * Let's simulate that and see if it changes the result significantly. */
    printf("\n=== Q8_0 QUANTIZED INPUT DOT PRODUCT ===\n");
    {
        /* Quantize normed to Q8_0: blocks of 32, each with scale and int8 values */
        int8_t q8_vals[LFM2_D_MODEL];
        float q8_scales[LFM2_D_MODEL / 32];

        for (int b = 0; b < LFM2_D_MODEL / 32; b++) {
            float amax = 0;
            for (int j = 0; j < 32; j++) {
                float a = fabsf(normed[b * 32 + j]);
                if (a > amax) amax = a;
            }
            float d = amax / 127.0f;
            q8_scales[b] = d;
            float id = d != 0 ? 1.0f / d : 0;
            for (int j = 0; j < 32; j++) {
                float v = normed[b * 32 + j] * id;
                q8_vals[b * 32 + j] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : roundf(v)));
            }
        }

        /* Now compute Q4_0 x Q8_0 dot product for row 0 */
        const q4_0_block_t *row0 = w->shortconv[0].in_proj;
        float dot_q8 = 0;
        for (int b = 0; b < 32; b++) {
            float w_scale = trix_fp16_to_f32(row0[b].scale_f16);
            float x_scale = q8_scales[b];
            int sum = 0;
            for (int j = 0; j < 16; j++) {
                uint8_t packed = row0[b].qs[j];
                int w0 = (int)(packed & 0x0F) - 8;
                int w1 = (int)(packed >> 4) - 8;
                sum += w0 * (int)q8_vals[b * 32 + 2*j];
                sum += w1 * (int)q8_vals[b * 32 + 2*j + 1];
            }
            dot_q8 += w_scale * x_scale * (float)sum;
        }
        printf("  Row 0 with Q8_0 input: %.6f (vs F32 input: -0.295303)\n", dot_q8);
        printf("  Difference due to Q8_0: %.6f\n", fabsf(dot_q8 - (-0.295303f)));
    }

    munmap((void *)file_data, st.st_size);
    close(fd);
    trix_unload_gguf(model);
    return 0;
}
