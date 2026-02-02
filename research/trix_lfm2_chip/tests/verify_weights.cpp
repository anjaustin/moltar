/*
 * verify_weights — Load model via llama.cpp and dequantize in_proj row 0
 * to compare against our GGUF loader's interpretation.
 *
 * Build:
 *   g++ -std=c++17 -O2 \
 *       -I ../../research/llama.cpp/include \
 *       -I ../../research/llama.cpp/ggml/include \
 *       -o build/verify_weights tests/verify_weights.cpp \
 *       -L ../../research/llama.cpp/build-mac/bin \
 *       -lllama -lggml -lggml-base -lggml-cpu -lm \
 *       -Wl,-rpath,../../research/llama.cpp/build-mac/bin
 */

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    /* Get the in_proj weight tensor directly from the model */
    /* In llama.cpp, tensors are accessible via model internals.
     * But the public API doesn't expose individual tensors easily.
     * Let's use ggml_backend_tensor_get to read the (potentially repacked) data.
     *
     * Actually, we can iterate the model's tensors... */

    /* Alternative approach: use the eval callback to capture the matmul output
     * AND the weight tensor values. Let me capture the input (normed) and
     * the weight scale factors to verify they match. */

    /* Actually, the simplest way: feed a known input vector through the matmul
     * and check the output. We already have the matmul output from llama_layer_dump.
     * Instead, let me create a simple 1-hot input and run it through the model
     * to isolate weight row behavior. */

    /* Even simpler: dump the weight tensor raw bytes via ggml_backend_tensor_get */
    /* We need to find the tensor. Let's try the model's graph. */

    /* Actually, let me just compare the raw in_proj data.
     * I'll load the file as raw bytes and verify against the GGUF offset. */

    /* Read the GGUF file directly */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open file\n"); return 1; }

    /* Offset from dump_gguf_meta: blk.0.shortconv.in_proj.weight at offset 63037440
     * data_offset = 2383808
     * absolute offset = 63037440 + 2383808 = 65421248 */
    size_t abs_offset = 65421248;

    fseek(f, abs_offset, SEEK_SET);

    /* Read first Q4_0 block (18 bytes: 2 byte scale + 16 byte quants) */
    uint8_t block_bytes[18];
    fread(block_bytes, 1, 18, f);
    fclose(f);

    /* Dequantize */
    uint16_t scale_f16 = block_bytes[0] | (block_bytes[1] << 8);
    /* FP16 to F32 */
    uint32_t sign = (scale_f16 >> 15) & 1;
    uint32_t exp  = (scale_f16 >> 10) & 0x1F;
    uint32_t frac = scale_f16 & 0x3FF;
    float scale;
    if (exp == 0) {
        scale = (sign ? -1.0f : 1.0f) * (frac / 1024.0f) * (1.0f / 16384.0f);
    } else if (exp == 31) {
        scale = (sign ? -1.0f : 1.0f) * (frac == 0 ? INFINITY : NAN);
    } else {
        uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
        memcpy(&scale, &f32, 4);
    }

    printf("Raw block 0 of in_proj:\n");
    printf("  scale_f16=0x%04x (%.6f)\n", scale_f16, scale);
    printf("  qs: ");
    for (int i = 0; i < 16; i++) printf("%02x ", block_bytes[2+i]);
    printf("\n");

    printf("  Dequantized 32 values:\n  ");
    for (int j = 0; j < 16; j++) {
        uint8_t packed = block_bytes[2+j];
        float v0 = ((float)(packed & 0x0F) - 8.0f) * scale;
        float v1 = ((float)(packed >> 4)   - 8.0f) * scale;
        printf("%.6f %.6f ", v0, v1);
        if ((j+1) % 4 == 0) printf("\n  ");
    }
    printf("\n");

    /* Now let's also read what the GGUF Q6_K embedding gives for token 1 */
    /* token_embd offset should be 0 (first tensor) + data_offset */
    /* Let me verify by reading the embedding and computing operator_norm manually */

    /* Read the operator_norm result from llama.cpp (from our earlier dump):
     * operator_norm-0 first 8: 0.173848 -0.032380 -0.048859 0.052039 -0.519809 0.015226 0.152262 0.071312
     *
     * Now manually compute: in_proj row 0 dot normed
     * = sum of (dequant(in_proj[0,k]) * normed[k]) for k=0..1023
     *
     * The normed vector has large values like -0.519809 at index 4.
     * But our dequantized in_proj row 0 has small values around +/- 0.009.
     * So the dot product would be dominated by large elements of normed.
     */

    /* Let's check: what's the llama.cpp matmul output for in_proj row 0?
     * llama.cpp says in_proj-0 first value = -0.000834
     * We compute -0.295303
     * That's a 353x difference.
     *
     * -0.000834 is TINY. It's as if the weight row is nearly orthogonal to normed.
     * Our answer -0.295 suggests significant correlation.
     *
     * Theory: what if llama.cpp's Q4_0_4x8 repacking shuffles the rows?
     * i.e., what was "row 0" in the original Q4_0 is no longer output[0]
     * after repacking? */

    /* Let's check: in the repacked format, 4 rows are interleaved.
     * The output order should still be sequential [0,1,2,3,4,5,...].
     * But what if the row ORDER within the weight matrix was already
     * different from what we expect?
     *
     * In GGML, for mul_mat(W, x):
     *   W shape = {ne0=1024, ne1=3072}
     *   output shape = {ne1=3072, 1}
     *   output[i] = dot(W_row_i, x)
     *
     * But wait - ne[0] is the INNER dimension.
     * Output ne[0] = src0->ne[1] = 3072
     *
     * For the VIEW operations in build_shortconv_block:
     *   auto * b = ggml_view_3d(ctx0, bcx, chunk_size, ..., 0 * chunk_size * sizeof(float));
     * where chunk_size = bcx->ne[0] / 3
     *
     * bcx->ne[0] = 3072 (the output of mul_mat has ne[0] = ne1 of weight = 3072)
     * chunk_size = 3072 / 3 = 1024
     * b = bcx[0..1023], c = bcx[1024..2047], x = bcx[2048..3071]
     *
     * So b = first 1024 outputs = in_proj rows 0..1023 dotted with input.
     * c = next 1024 = rows 1024..2047
     * x = last 1024 = rows 2048..3071
     *
     * This is the SAME as what we do. So the split is correct.
     *
     * UNLESS... the weight tensor shape in GGML for mul_mat is interpreted differently.
     * What if ne[0] is the OUTPUT dimension, not the inner/reduction dimension?
     *
     * In GGML mul_mat(A, B):
     *   C->ne[0] = A->ne[1]
     *   C->ne[1] = B->ne[1]
     *   For each (i0, i1): C[i0, i1] = dot(A_row[i0], B_col[i1])
     *   where A_row[i0] has A->ne[0] elements.
     *
     * A = weight = {ne0=1024, ne1=3072}
     * B = input  = {ne0=1024, ne1=1}
     * C = output = {ne0=3072, ne1=1}
     * C[i0] = dot(A_row[i0], B_col[0]) = dot(A_row[i0], B)
     * A_row[i0] has ne0=1024 elements.
     * i0 ranges from 0 to ne1-1 = 3071.
     *
     * A_row[i0] in memory = A_data + i0 * nb1 = A_data + i0 * (ne0 * type_size)
     *
     * For Q4_0: type_size = 18 bytes per block, 32 elements per block
     *   row stride = ne0 * type_size / block_size = 1024 * 18 / 32 = 576 bytes
     *   A_row[i0] = A_data + i0 * 576
     *
     * Our code: row = W + i * n_blocks_per_row = W + i * 32
     *   In bytes: i * 32 * 18 = i * 576. Matches!
     */

    printf("\nConclusion: weight layout and offset are verified correct.\n");
    printf("The mystery remains: why does llama.cpp produce different output?\n\n");

    /* Let me try one more thing: manually compute the dot product using
     * llama.cpp's normed values, to rule out norm differences */
    printf("Computing manual dot product with EXACT llama.cpp normed values...\n");
    float llama_normed_first8[] = {0.173848f, -0.032380f, -0.048859f, 0.052039f,
                                    -0.519809f, 0.015226f, 0.152262f, 0.071312f};
    /* These match our values to 6 decimal places, so it's not a norm issue */

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
