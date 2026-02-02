/*
 * verify_inproj_weights — Definitive test: use eval callback to capture
 * the raw weight tensor AND the normed input, then compute the dot product
 * manually to prove whether the weight data matches.
 *
 * Strategy: In the eval callback, intercept both:
 * 1. The operator_norm output (our input x to the matmul)
 * 2. The in_proj output (result of the matmul) 
 * 3. ANY tensor whose data pointer we can get at
 *
 * We also try to read src[0] (weight) from the in_proj result tensor.
 *
 * Build against CPU-only no-repack:
 *   g++ -std=c++17 -O2 \
 *       -I ../../research/llama.cpp/include \
 *       -I ../../research/llama.cpp/ggml/include \
 *       -o build/verify_inproj_weights tests/verify_inproj_weights.cpp \
 *       -L ../../research/llama.cpp/build-mac-cpuonly/bin \
 *       -lllama -lggml -lggml-base -lggml-cpu -lm \
 *       -Wl,-rpath,../../research/llama.cpp/build-mac-cpuonly/bin
 */

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>
#include <map>

/* Q4_0 block structure — matches ggml's definition */
struct q4_0_block {
    uint16_t d;      /* FP16 scale */
    uint8_t  qs[16]; /* 32 x 4-bit quants packed into 16 bytes */
};
static_assert(sizeof(q4_0_block) == 18, "Q4_0 block must be 18 bytes");

/* FP16 to F32 conversion (handles subnormals) */
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, 4);
            return f;
        }
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        uint32_t result = sign | 0x7F800000 | (mant << 13);
        float f;
        memcpy(&f, &result, 4);
        return f;
    }
    
    uint32_t result = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
}

/* Global captures */
static std::vector<float> g_norm_output;
static bool g_norm_captured = false;

/* Weight tensor info — captured from the graph traversal */
static const void *g_weight_data = nullptr;
static int64_t g_weight_ne[4] = {0};
static size_t g_weight_nb[4] = {0};
static int g_weight_type = -1;

static bool eval_callback(struct ggml_tensor *t, bool ask, void *user_data) {
    (void)user_data;
    const char *name = ggml_get_name(t);
    
    if (ask) {
        /* We want the in_proj result — and from it we'll get src[0] (weight) */
        if (name && strcmp(name, "model.layers.{}.operator_norm-0") == 0) return true;
        if (name && strcmp(name, "model.layers.{}.conv.in_proj-0") == 0) return true;
        return false;
    }
    
    if (name && strcmp(name, "model.layers.{}.operator_norm-0") == 0) {
        if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
            int64_t n = ggml_nelements(t);
            g_norm_output.resize(n);
            ggml_backend_tensor_get(t, g_norm_output.data(), 0, n * sizeof(float));
            g_norm_captured = true;
            printf("[callback] Captured operator_norm-0: %lld elements\n", (long long)n);
        }
    }
    
    if (name && strcmp(name, "model.layers.{}.conv.in_proj-0") == 0) {
        /* This tensor is the result of ggml_mul_mat(weight, cur)
         * t->src[0] = weight tensor (Q4_0)
         * t->src[1] = cur (the normed input, F32) 
         *
         * We can read src[0] to get the weight data pointer! */
        printf("[callback] in_proj-0 tensor:\n");
        printf("  result type: %d  ne: [%lld, %lld, %lld, %lld]\n",
               t->type, (long long)t->ne[0], (long long)t->ne[1],
               (long long)t->ne[2], (long long)t->ne[3]);
        printf("  op: %d (MUL_MAT=%d)\n", t->op, GGML_OP_MUL_MAT);
        
        if (t->src[0]) {
            struct ggml_tensor *w = t->src[0];
            printf("  src[0] (weight): name='%s' type=%d ne=[%lld,%lld,%lld,%lld]\n",
                   ggml_get_name(w), w->type,
                   (long long)w->ne[0], (long long)w->ne[1],
                   (long long)w->ne[2], (long long)w->ne[3]);
            printf("  src[0] nb=[%zu,%zu,%zu,%zu]\n",
                   w->nb[0], w->nb[1], w->nb[2], w->nb[3]);
            printf("  src[0] data=%p\n", w->data);
            
            /* Try to get backend buffer info */
            struct ggml_backend_buffer *buf = w->buffer;
            printf("  src[0] buffer=%p\n", (void*)buf);
            
            /* Save for later analysis */
            g_weight_data = w->data;
            g_weight_type = w->type;
            for (int i = 0; i < 4; i++) {
                g_weight_ne[i] = w->ne[i];
                g_weight_nb[i] = w->nb[i];
            }
            
            /* If the data pointer is directly accessible (CPU buffer), 
             * try to read the first few bytes */
            if (w->data) {
                const uint8_t *bytes = (const uint8_t *)w->data;
                printf("  src[0] first 36 bytes: ");
                for (int i = 0; i < 36; i++) printf("%02x ", bytes[i]);
                printf("\n");
                
                /* Dequantize row 0 and compute dot product */
                if (g_norm_captured && w->type == GGML_TYPE_Q4_0) {
                    const int ne0 = (int)w->ne[0]; /* 1024 */
                    const int nblocks = ne0 / 32;
                    const size_t row_bytes = w->nb[1];
                    
                    printf("\n  === Manual dot product from llama.cpp weight data ===\n");
                    printf("  ne0=%d nblocks=%d row_bytes=%zu\n", ne0, nblocks, row_bytes);
                    
                    /* Test rows 0, 1, 2, 2619 */
                    int test_rows[] = {0, 1, 2, 2619, -1};
                    for (int ri = 0; test_rows[ri] >= 0; ri++) {
                        int row = test_rows[ri];
                        if (row >= (int)w->ne[1]) continue;
                        
                        const q4_0_block *blocks = (const q4_0_block *)(bytes + row * row_bytes);
                        float dot = 0.0f;
                        
                        for (int b = 0; b < nblocks; b++) {
                            float scale = fp16_to_f32(blocks[b].d);
                            for (int j = 0; j < 16; j++) {
                                uint8_t byte_val = blocks[b].qs[j];
                                float v0 = ((int)(byte_val & 0xF) - 8) * scale;
                                float v1 = ((int)(byte_val >> 4) - 8) * scale;
                                dot += v0 * g_norm_output[b * 32 + j];
                                dot += v1 * g_norm_output[b * 32 + j + 16];
                            }
                        }
                        
                        printf("  Row %4d: dot = %.6f\n", row, dot);
                        
                        if (row == 0) {
                            printf("           block[0] scale_fp16=0x%04x scale_f32=%.10f\n",
                                   blocks[0].d, fp16_to_f32(blocks[0].d));
                            printf("           block[0] qs[0..3]: %02x %02x %02x %02x\n",
                                   blocks[0].qs[0], blocks[0].qs[1], blocks[0].qs[2], blocks[0].qs[3]);
                        }
                    }
                    
                    /* Scan all rows for best match to llama's output[0] */
                    printf("\n  === Scanning all %lld rows ===\n", (long long)w->ne[1]);
                    float target = -0.000834f;
                    float best_diff = 1e30f;
                    int best_row = -1;
                    float best_dot = 0;
                    
                    for (int row = 0; row < (int)w->ne[1]; row++) {
                        const q4_0_block *blocks = (const q4_0_block *)(bytes + row * row_bytes);
                        float dot = 0.0f;
                        
                        for (int b = 0; b < nblocks; b++) {
                            float scale = fp16_to_f32(blocks[b].d);
                            for (int j = 0; j < 16; j++) {
                                uint8_t byte_val = blocks[b].qs[j];
                                float v0 = ((int)(byte_val & 0xF) - 8) * scale;
                                float v1 = ((int)(byte_val >> 4) - 8) * scale;
                                dot += v0 * g_norm_output[b * 32 + j];
                                dot += v1 * g_norm_output[b * 32 + j + 16];
                            }
                        }
                        
                        float diff = fabsf(dot - target);
                        if (diff < best_diff) {
                            best_diff = diff;
                            best_row = row;
                            best_dot = dot;
                        }
                    }
                    
                    printf("  Best match to %.6f: row %d, dot=%.6f, diff=%.6e\n",
                           target, best_row, best_dot, best_diff);
                }
            }
        }
        
        if (t->src[1]) {
            struct ggml_tensor *x = t->src[1];
            printf("\n  src[1] (input): name='%s' type=%d ne=[%lld,%lld,%lld,%lld]\n",
                   ggml_get_name(x), x->type,
                   (long long)x->ne[0], (long long)x->ne[1],
                   (long long)x->ne[2], (long long)x->ne[3]);
            printf("  src[1] data=%p\n", x->data);
            
            /* Read the actual input values that went into the matmul */
            if (x->type == GGML_TYPE_F32 && ggml_is_contiguous(x)) {
                int64_t n = ggml_nelements(x);
                std::vector<float> input_data(n);
                ggml_backend_tensor_get(x, input_data.data(), 0, n * sizeof(float));
                printf("  src[1] first 8: ");
                int show = n < 8 ? (int)n : 8;
                for (int i = 0; i < show; i++) printf("%.6f ", input_data[i]);
                printf("\n");
                
                /* Compare to our captured norm output */
                if (g_norm_captured) {
                    float max_diff = 0;
                    for (int i = 0; i < (int)g_norm_output.size() && i < (int)n; i++) {
                        float d = fabsf(input_data[i] - g_norm_output[i]);
                        if (d > max_diff) max_diff = d;
                    }
                    printf("  Max diff between src[1] and operator_norm-0: %.6e\n", max_diff);
                }
            }
        }
        
        /* Read the actual result values */
        if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
            int64_t n = ggml_nelements(t);
            std::vector<float> result(n);
            ggml_backend_tensor_get(t, result.data(), 0, n * sizeof(float));
            printf("\n  in_proj result first 8: ");
            int show = n < 8 ? (int)n : 8;
            for (int i = 0; i < show; i++) printf("%.6f ", result[i]);
            printf("\n");
        }
    }
    
    return true;
}

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
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;
    cparams.cb_eval = eval_callback;
    cparams.cb_eval_user_data = nullptr;
    
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    
    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    printf("BOS token: %d\n\n", bos);
    
    llama_batch batch = llama_batch_get_one(&bos, 1);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "llama_decode failed\n"); return 1;
    }
    
    printf("\n=== Done ===\n");
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
