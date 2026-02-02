/*
 * test_prompt — Feed a tokenized prompt and generate, mimicking llama.cpp behavior
 */
#include "../include/lfm2_trix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t argmax(const float *logits, int n) {
    float mx = logits[0]; uint32_t mi = 0;
    for (int i = 1; i < n; i++) if (logits[i] > mx) { mx = logits[i]; mi = i; }
    return mi;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }

    trix_gguf_model_t *model = trix_load_gguf(argv[1]);
    if (!model) return 1;
    const trix_lfm2_weights_t *w = trix_get_weights(model);

    trix_lfm2_state_t *state = trix_lfm2_init(2048);
    float *x = malloc(LFM2_D_MODEL * sizeof(float));
    float *logits = malloc(LFM2_VOCAB_SIZE * sizeof(float));

    /* Prompt: BOS + "The capital of France is" 
     * GPT2 tokenization: [1, 464, 3139, 286, 4881, 318] */
    uint32_t prompt[] = {1, 464, 3139, 286, 4881, 318};
    int prompt_len = 6;
    int n_gen = 10;

    printf("Prompt tokens: ");
    for (int i = 0; i < prompt_len; i++) printf("%u ", prompt[i]);
    printf("\n\n");

    /* Prompt phase: feed each token and advance */
    printf("--- Prompt phase ---\n");
    for (int i = 0; i < prompt_len; i++) {
        trix_embed_token(w, prompt[i], x);
        trix_lfm2_forward(state, w, x, logits);
        uint32_t top = argmax(logits, LFM2_VOCAB_SIZE);
        printf("  pos=%d  token_in=%u  top_pred=%u  logit=%.4f\n",
               i, prompt[i], top, logits[top]);
    }

    /* Generation phase */
    printf("\n--- Generation phase ---\n");
    uint32_t next = argmax(logits, LFM2_VOCAB_SIZE);
    for (int i = 0; i < n_gen; i++) {
        trix_embed_token(w, next, x);
        trix_lfm2_forward(state, w, x, logits);
        uint32_t top = argmax(logits, LFM2_VOCAB_SIZE);
        printf("  pos=%d  token_in=%u  top_pred=%u  logit=%.4f\n",
               prompt_len + i, next, top, logits[top]);
        next = top;
    }

    /* Check: is token 6342 (Paris) in the top predictions anywhere? */
    printf("\n--- Logit for ' Paris' (token 6342) at last prompt position: %.4f ---\n",
           logits[6342]);

    free(logits); free(x);
    trix_lfm2_free(state);
    trix_unload_gguf(model);
    return 0;
}
