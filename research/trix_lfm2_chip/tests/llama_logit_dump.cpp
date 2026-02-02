/*
 * llama_logit_dump — Use llama.cpp to dump logits for a single BOS token
 * Build: g++ -std=c++17 -I ../llama.cpp/include -I ../llama.cpp/ggml/include \
 *        -o build/llama_logit_dump tests/llama_logit_dump.cpp \
 *        -L ../llama.cpp/build-mac/bin -lllama -lggml -lggml-base -lggml-cpu -lm \
 *        -Wl,-rpath,../llama.cpp/build-mac/bin
 */

#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.no_perf = true;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    /* Feed BOS token */
    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    printf("BOS token: %d\n", bos);

    llama_batch batch = llama_batch_get_one(&bos, 1);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "llama_decode failed\n"); return 1;
    }

    /* Get logits */
    const float *logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(vocab);
    printf("n_vocab: %d\n", n_vocab);

    /* Print first 8 logits */
    printf("Logits first 8: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", logits[i]);
    printf("\n");

    /* Find top-5 */
    int top[5] = {0}; float topv[5] = {-1e30,-1e30,-1e30,-1e30,-1e30};
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
    printf("Top 5:\n");
    for (int k = 0; k < 5; k++) printf("  token=%d  logit=%.4f\n", top[k], topv[k]);

    /* Also feed "The capital of France is" = [464, 3139, 286, 4881, 318] */
    printf("\n--- After full prompt \"The capital of France is\" ---\n");
    llama_token prompt[] = {464, 3139, 286, 4881, 318};
    llama_batch batch2 = llama_batch_get_one(prompt, 5);
    if (llama_decode(ctx, batch2)) {
        fprintf(stderr, "llama_decode failed for prompt\n"); return 1;
    }

    logits = llama_get_logits(ctx);
    printf("Logits first 8: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", logits[i]);
    printf("\n");

    /* top-5 */
    for (int i = 0; i < 5; i++) { top[i]=0; topv[i]=-1e30; }
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
    printf("Top 5:\n");
    for (int k = 0; k < 5; k++) printf("  token=%d  logit=%.4f\n", top[k], topv[k]);
    printf("Logit for ' Paris' (6342): %.4f\n", logits[6342]);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
