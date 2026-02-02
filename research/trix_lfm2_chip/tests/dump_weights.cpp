/*
 * dump_weights — Load model via llama.cpp, extract raw in_proj weight bytes,
 * and dump them for comparison against our GGUF loader.
 *
 * Also dequantizes row 0 and row 1 of the weight and dumps the values.
 *
 * Build:
 *   g++ -std=c++17 -O2 \
 *       -I ../../research/llama.cpp/include \
 *       -I ../../research/llama.cpp/ggml/include \
 *       -o build/dump_weights tests/dump_weights.cpp \
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
#include <string>
#include <vector>
#include <map>

/* We'll use the eval callback to capture the weight tensor (src[0]) of the
 * in_proj matmul, as well as the input (src[1] = normed) and output. */

struct tensor_capture {
    std::string name;
    std::vector<uint8_t> raw_data;
    int64_t ne[4];
    int n_dims;
    int type;
};

static std::map<std::string, tensor_capture> g_captures;

static bool eval_callback(struct ggml_tensor *t, bool ask, void *user_data) {
    (void)user_data;
    const char *n = ggml_get_name(t);
    if (!n) return false;

    if (ask) {
        /* We want to capture the in_proj matmul output */
        if (strcmp(n, "model.layers.{}.conv.in_proj-0") == 0) return true;
        if (strcmp(n, "model.layers.{}.operator_norm-0") == 0) return true;
        return false;
    }

    /* When we get the in_proj output, also dump its src[0] (weight) and src[1] (input) */
    if (strcmp(n, "model.layers.{}.conv.in_proj-0") == 0) {
        /* Dump the matmul output */
        {
            tensor_capture cap;
            cap.name = "in_proj_output";
            cap.n_dims = ggml_n_dims(t);
            cap.type = t->type;
            int64_t total = 1;
            for (int i = 0; i < 4; i++) { cap.ne[i] = t->ne[i]; total *= t->ne[i]; }
            size_t nbytes = total * sizeof(float); /* output is F32 */
            cap.raw_data.resize(nbytes);
            ggml_backend_tensor_get(t, cap.raw_data.data(), 0, nbytes);
            g_captures["in_proj_output"] = std::move(cap);
        }

        /* Dump the weight tensor (src[0]) */
        if (t->src[0]) {
            struct ggml_tensor *w = t->src[0];
            tensor_capture cap;
            cap.name = "in_proj_weight";
            cap.n_dims = ggml_n_dims(w);
            cap.type = w->type;
            size_t nbytes = ggml_nbytes(w);
            for (int i = 0; i < 4; i++) cap.ne[i] = w->ne[i];
            printf("in_proj weight tensor:\n");
            printf("  name: %s\n", ggml_get_name(w));
            printf("  type: %d (%s)\n", w->type, ggml_type_name((ggml_type)w->type));
            printf("  ne: [%lld, %lld, %lld, %lld]\n",
                   (long long)w->ne[0], (long long)w->ne[1], (long long)w->ne[2], (long long)w->ne[3]);
            printf("  nb: [%zu, %zu, %zu, %zu]\n", w->nb[0], w->nb[1], w->nb[2], w->nb[3]);
            printf("  nbytes: %zu\n", nbytes);

            /* Read first 576 bytes (first row = 32 Q4_0 blocks) */
            size_t read_bytes = nbytes < 1152 ? nbytes : 1152; /* 2 rows */
            cap.raw_data.resize(read_bytes);
            ggml_backend_tensor_get(w, cap.raw_data.data(), 0, read_bytes);
            g_captures["in_proj_weight"] = std::move(cap);

            /* Print raw hex of first 2 Q4_0 blocks (36 bytes) */
            printf("  First 36 raw bytes (2 blocks): ");
            for (size_t i = 0; i < 36 && i < read_bytes; i++) {
                printf("%02x ", g_captures["in_proj_weight"].raw_data[i]);
            }
            printf("\n");

            /* Dequantize block 0 */
            const uint8_t *blk = g_captures["in_proj_weight"].raw_data.data();
            uint16_t scale_f16 = blk[0] | (blk[1] << 8);
            /* FP16 -> F32 */
            uint32_t sign = (scale_f16 >> 15) & 1;
            uint32_t exp  = (scale_f16 >> 10) & 0x1F;
            uint32_t frac = scale_f16 & 0x3FF;
            float scale;
            if (exp == 0) {
                scale = (sign ? -1.0f : 1.0f) * (frac / 1024.0f) * (1.0f / 16384.0f);
            } else if (exp == 31) {
                scale = 0.0f;
            } else {
                uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
                memcpy(&scale, &f32, 4);
            }
            printf("  Block 0: scale_f16=0x%04x scale=%.6f\n", scale_f16, scale);
            printf("  Block 0 dequant first 8: ");
            for (int j = 0; j < 4; j++) {
                uint8_t packed = blk[2+j];
                float v0 = ((float)(packed & 0x0F) - 8.0f) * scale;
                float v1 = ((float)(packed >> 4)   - 8.0f) * scale;
                printf("%.6f %.6f ", v0, v1);
            }
            printf("\n");
        }

        /* Dump the input tensor (src[1]) */
        if (t->src[1]) {
            struct ggml_tensor *inp = t->src[1];
            printf("\nin_proj input tensor:\n");
            printf("  name: %s\n", ggml_get_name(inp));
            printf("  type: %d (%s)\n", inp->type, ggml_type_name((ggml_type)inp->type));
            printf("  ne: [%lld, %lld, %lld, %lld]\n",
                   (long long)inp->ne[0], (long long)inp->ne[1], (long long)inp->ne[2], (long long)inp->ne[3]);

            /* Check if it's Q8_0 (quantized input) */
            if (inp->type == GGML_TYPE_Q8_0) {
                printf("  ** INPUT IS Q8_0 (quantized!) **\n");
                /* Read first Q8_0 block (34 bytes) */
                uint8_t q8_block[34];
                ggml_backend_tensor_get(inp, q8_block, 0, 34);
                printf("  First Q8_0 block raw: ");
                for (int i = 0; i < 34; i++) printf("%02x ", q8_block[i]);
                printf("\n");
                /* Dequantize */
                uint16_t d_f16 = q8_block[0] | (q8_block[1] << 8);
                uint32_t s = (d_f16 >> 15) & 1;
                uint32_t e = (d_f16 >> 10) & 0x1F;
                uint32_t f = d_f16 & 0x3FF;
                float d;
                if (e == 0) d = (s ? -1.0f : 1.0f) * (f / 1024.0f) * (1.0f / 16384.0f);
                else if (e == 31) d = 0.0f;
                else { uint32_t f32 = (s << 31) | ((e + 112) << 23) | (f << 13); memcpy(&d, &f32, 4); }
                printf("  Q8_0 block 0: d=%.6f\n", d);
                printf("  First 8 dequantized: ");
                for (int i = 0; i < 8; i++) {
                    printf("%.6f ", d * (float)(int8_t)q8_block[2+i]);
                }
                printf("\n");
            } else if (inp->type == GGML_TYPE_F32) {
                printf("  ** INPUT IS F32 **\n");
                float first8[8];
                ggml_backend_tensor_get(inp, first8, 0, 8 * sizeof(float));
                printf("  First 8: ");
                for (int i = 0; i < 8; i++) printf("%.6f ", first8[i]);
                printf("\n");
            }
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

    /* Feed BOS token */
    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    printf("BOS token: %d\n\n", bos);

    llama_batch batch = llama_batch_get_one(&bos, 1);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "llama_decode failed\n"); return 1;
    }

    /* Print in_proj output for comparison */
    if (g_captures.count("in_proj_output")) {
        auto &cap = g_captures["in_proj_output"];
        float *data = (float *)cap.raw_data.data();
        size_t n = cap.raw_data.size() / sizeof(float);
        printf("\n=== IN_PROJ OUTPUT (llama.cpp) ===\n");
        printf("  n_elem: %zu\n", n);
        printf("  first 8: ");
        for (size_t i = 0; i < 8 && i < n; i++) printf("%.6f ", data[i]);
        printf("\n");

        /* Write to binary file */
        FILE *fp = fopen("build/llama_inproj0.bin", "wb");
        if (fp) {
            fwrite(data, sizeof(float), n, fp);
            fclose(fp);
            printf("  Written to build/llama_inproj0.bin\n");
        }
    }

    /* Dump the weight bytes to a file for comparison */
    if (g_captures.count("in_proj_weight")) {
        auto &cap = g_captures["in_proj_weight"];
        FILE *fp = fopen("build/llama_inproj_weight_first2rows.bin", "wb");
        if (fp) {
            fwrite(cap.raw_data.data(), 1, cap.raw_data.size(), fp);
            fclose(fp);
            printf("\n  Weight first 2 rows (%zu bytes) written to build/llama_inproj_weight_first2rows.bin\n",
                   cap.raw_data.size());
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
