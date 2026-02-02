/*
 * llama_layer_dump — Dump intermediate activations from llama.cpp for comparison
 *
 * Uses the cb_eval callback to intercept every tensor during forward pass.
 * Dumps embedding, layer norms, block outputs, and FFN outputs.
 *
 * Build:
 *   g++ -std=c++17 -O2 \
 *       -I ../../research/llama.cpp/include \
 *       -I ../../research/llama.cpp/ggml/include \
 *       -o build/llama_layer_dump tests/llama_layer_dump.cpp \
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

/* Store captured tensors */
struct tensor_capture {
    std::string name;
    std::vector<float> data;
    int64_t ne[4];
    int n_dims;
};

static std::map<std::string, tensor_capture> g_captures;

/* List of EXACT tensor names we want to capture.
 * Use strcmp matching. Only capture small f32 contiguous tensors. */
static const char *g_wanted[] = {
    "model.embed_tokens",              /* embedding output */
    "model.layers.{}.operator_norm-0", /* layer 0: RMSNorm WITH gamma */
    "model.layers.{}.conv.in_proj-0",  /* layer 0: in_proj output */
    "model.layers.{}.conv.conv-0",     /* layer 0: conv output */
    "node_33",                         /* layer 0: c * conv_out (y) */
    "model.layers.{}.conv.out_proj-0", /* layer 0: out_proj output */
    "node_36",                         /* layer 0: residual add (before FFN) */
    "node_43",                         /* layer 0: after residual + FFN */
    "result_norm",                     /* final norm */
    NULL
};

static bool is_wanted(const char *name) {
    if (!name || !name[0]) return false;
    for (int i = 0; g_wanted[i]; i++) {
        if (strcmp(name, g_wanted[i]) == 0) return true;
    }
    return false;
}

/* Also capture all tensor names for debugging */
static std::vector<std::string> g_all_names;

static bool eval_callback(struct ggml_tensor *t, bool ask, void *user_data) {
    (void)user_data;

    if (ask) {
        /* Log ALL tensor names on the ask pass */
        const char *n = ggml_get_name(t);
        if (n && n[0]) {
            g_all_names.push_back(n);
        }
        /* Return true for tensors we want to capture */
        return is_wanted(n);
    }

    /* Not asking — we have the data, capture it */
    const char *name = ggml_get_name(t);
    if (!is_wanted(name)) return true;

    /* Only capture float tensors that are contiguous */
    if (t->type != GGML_TYPE_F32) return true;
    if (!ggml_is_contiguous(t)) return true;

    tensor_capture cap;
    cap.name = name;
    cap.n_dims = ggml_n_dims(t);
    int64_t total = 1;
    for (int i = 0; i < 4; i++) {
        cap.ne[i] = t->ne[i];
        total *= t->ne[i];
    }

    /* Safety: skip huge tensors */
    if (total > 1024 * 1024) return true;

    /* Read data from backend */
    cap.data.resize(total);
    ggml_backend_tensor_get(t, cap.data.data(), 0, total * sizeof(float));

    g_captures[name] = std::move(cap);

    return true;
}

static void print_capture(const tensor_capture &cap) {
    printf("  %-45s shape=[", cap.name.c_str());
    for (int i = 0; i < cap.n_dims; i++) {
        printf("%lld%s", (long long)cap.ne[i], i < cap.n_dims - 1 ? "," : "");
    }
    printf("]  n_elem=%zu\n", cap.data.size());

    /* Compute stats */
    float l2 = 0, mn = cap.data[0], mx = cap.data[0];
    for (size_t i = 0; i < cap.data.size(); i++) {
        l2 += cap.data[i] * cap.data[i];
        if (cap.data[i] < mn) mn = cap.data[i];
        if (cap.data[i] > mx) mx = cap.data[i];
    }
    l2 = sqrtf(l2);
    printf("    L2=%.6f  min=%.6f  max=%.6f\n", l2, mn, mx);

    /* Print first 8 values */
    int show = cap.data.size() < 8 ? (int)cap.data.size() : 8;
    printf("    first %d: ", show);
    for (int i = 0; i < show; i++) printf("%.6f ", cap.data[i]);
    printf("\n");

    /* Also print last 4 for verification */
    if (cap.data.size() > 8) {
        printf("    last 4: ");
        for (int i = (int)cap.data.size() - 4; i < (int)cap.data.size(); i++) {
            printf("%.6f ", cap.data[i]);
        }
        printf("\n");
    }
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

    /* Print all tensor names observed */
    printf("=== All %zu tensor names observed ===\n", g_all_names.size());
    for (auto &n : g_all_names) {
        printf("  %s\n", n.c_str());
    }
    printf("\n");

    /* Print all captured tensors */
    printf("=== Captured %zu tensors ===\n\n", g_captures.size());

    /* Print in a reasonable order */
    for (int i = 0; g_wanted[i]; i++) {
        for (auto &[name, cap] : g_captures) {
            if (name.find(g_wanted[i]) != std::string::npos) {
                print_capture(cap);
                printf("\n");
            }
        }
    }

    /* Compare operator_norm-0 (norm with gamma) against what we'd expect */
    if (g_captures.count("model.layers.{}.operator_norm-0")) {
        auto &op_norm = g_captures["model.layers.{}.operator_norm-0"];
        printf("--- operator_norm-0 (norm * gamma) ---\n");
        printf("  shape: ");
        for (int i = 0; i < op_norm.n_dims; i++) printf("%lld ", (long long)op_norm.ne[i]);
        printf("  n_elem=%zu\n", op_norm.data.size());
        printf("  first 8: ");
        for (int i = 0; i < 8 && i < (int)op_norm.data.size(); i++) printf("%.6f ", op_norm.data[i]);
        printf("\n");
        float ol2 = 0; for (auto v : op_norm.data) ol2 += v*v; ol2 = sqrtf(ol2);
        printf("  L2=%.6f\n", ol2);
    }

    /* Check if embedding and norm-0 shapes are different */
    if (g_captures.count("model.embed_tokens") && g_captures.count("norm-0")) {
        auto &emb = g_captures["model.embed_tokens"];
        auto &norm0 = g_captures["norm-0"];
        printf("--- Embedding vs norm-0 shapes ---\n");
        printf("  emb  n_elem=%zu  shape=", emb.data.size());
        for (int i = 0; i < emb.n_dims; i++) printf("%lld ", (long long)emb.ne[i]);
        printf("\n  norm0 n_elem=%zu  shape=", norm0.data.size());
        for (int i = 0; i < norm0.n_dims; i++) printf("%lld ", (long long)norm0.ne[i]);
        printf("\n");

        /* Verify: manually compute rms_norm of embedding */
        double sum_sq = 0;
        for (size_t i = 0; i < emb.data.size(); i++) sum_sq += (double)emb.data[i] * emb.data[i];
        double mean_sq = sum_sq / emb.data.size();
        float inv_rms = 1.0f / sqrtf((float)mean_sq + 1e-5f);
        printf("--- MANUAL RMS_NORM of embedding (inv_rms=%.6f, emb_size=%zu) ---\n",
               inv_rms, emb.data.size());
        for (int i = 0; i < 8; i++)
            printf("  [%d] manual=%.10f  norm0=%.10f  diff=%.2e\n",
                   i, emb.data[i] * inv_rms, norm0.data[i], emb.data[i] * inv_rms - norm0.data[i]);
    }

    /* Dump full in_proj-0 to binary file for cross-correlation */
    if (g_captures.count("model.layers.{}.conv.in_proj-0")) {
        auto &inproj = g_captures["model.layers.{}.conv.in_proj-0"];
        FILE *fp = fopen("build/llama_inproj0.bin", "wb");
        if (fp) {
            fwrite(inproj.data.data(), sizeof(float), inproj.data.size(), fp);
            fclose(fp);
            printf("--- Wrote %zu floats to build/llama_inproj0.bin ---\n\n", inproj.data.size());
        }
    }

    /* Also get and print the final logits */
    const float *logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(vocab);
    printf("--- Final logits ---\n");
    printf("  first 8: ");
    for (int i = 0; i < 8; i++) printf("%.6f ", logits[i]);
    printf("\n");

    /* Top 5 */
    int top[5] = {0}; float topv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
    for (int i = 0; i < n_vocab; i++) {
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

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
