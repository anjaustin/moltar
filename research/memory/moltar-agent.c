/* ==========================================================================
 * Moltar Agent — Multi-turn LLM with Working Memory
 * ==========================================================================
 * Links against libllama.so (LFM2 inference) + LCVDB (semantic working memory).
 *
 * Each conversation turn:
 *   1. Read user input
 *   2. Search LCVDB for top-3 related prior turns → build context prefix
 *   3. Tokenize full prompt (context + user input)
 *   4. Generate response via llama_decode + sampling
 *   5. Extract hidden state via llama_get_embeddings_ith(-1) → 2048D float
 *   6. Project to 48D int8 via moltar_proj_apply()
 *   7. Insert into LCVDB with turn metadata
 *   8. Print response, loop
 *
 * Build with NDK clang, link against libllama.so + LCVDB (static).
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "llama.h"
#include "ggml-backend.h"

#include "project.h"
#include "../lcvdb/lcvdb.h"

/* ---------- Configuration ---------- */

#define MAX_TURNS       256      /* max conversation turns in LCVDB         */
#define MAX_CONTEXT     3        /* top-K related turns to prepend          */
#define MAX_INPUT       1024     /* max user input chars                    */
#define MAX_RESPONSE    512      /* max generated tokens per turn           */
#define MAX_PROMPT_TOKENS 2048   /* max tokens in full prompt               */
#define MAX_TURN_TEXT   2048     /* max stored text per turn                */

/* Projection seed "MOLT" */
#define PROJ_SEED       0x4D4F4C54

/* ---------- Turn storage ---------- */

typedef struct {
    char text[MAX_TURN_TEXT];   /* "User: ...\nAssistant: ..."             */
    int  valid;                 /* 1 if this slot is used                  */
} turn_t;

static turn_t     turns[MAX_TURNS];
static int        n_turns = 0;

/* ---------- LCVDB working memory ---------- */

static lcvdb_t    vdb __attribute__((aligned(64)));
static void      *topo_buf;
static void      *vec_buf;

/* ---------- Projection ---------- */

static moltar_proj_t proj;

/* ---------- Helpers ---------- */

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s <model.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  -t N    threads for generation (default: 2)\n"
        "  -c N    context size (default: 4096)\n"
        "  -n N    max tokens per response (default: %d)\n"
        "  --temp F  temperature (default: 0.7)\n"
        "  --top-k N top-k sampling (default: 40)\n"
        "  --top-p F top-p sampling (default: 0.9)\n"
        "  --no-memory  disable LCVDB working memory\n"
        "\n"
        "Runs a multi-turn conversation with semantic working memory.\n"
        "Type your message and press Enter. Ctrl-D or 'quit' to exit.\n",
        argv0, MAX_RESPONSE);
}

/* Build prompt with context from working memory using ChatML format.
 * LFM2 expects:
 *   <|im_start|>system\n...<|im_end|>\n
 *   <|im_start|>user\n...<|im_end|>\n
 *   <|im_start|>assistant\n
 *
 * BOS token (id=1) is added automatically by llama_tokenize(add_special=true).
 * The model generates until <|im_end|> (token 7, EOG).
 */
static int build_prompt(char *prompt, int prompt_max,
                        const char *user_input, int use_memory) {
    int pos = 0;

    /* System message with optional working memory context */
    pos += snprintf(prompt + pos, prompt_max - pos,
        "<|im_start|>system\n"
        "You are Moltar, a helpful AI assistant running on a Motorola phone. "
        "Be concise and direct.");

    /* Retrieve context from LCVDB working memory */
    if (use_memory && n_turns > 0) {
        /* Search using the most recent turn's vector as a proxy for the
         * current topic. This finds turns contextually related to the
         * recent conversation flow. */
        if (n_turns >= 1) {
            uint16_t last_id = (uint16_t)(n_turns - 1);

            uint16_t result_ids[MAX_CONTEXT + 1];
            int32_t  result_scores[MAX_CONTEXT + 1];
            int nresults = lcvdb_search(&vdb,
                (const int8_t *)((char *)vec_buf + last_id * sizeof(lcvdb_vec_t)),
                MAX_CONTEXT + 1, result_ids, result_scores);

            /* Collect retrieved turns (skip most-recent, it's added separately) */
            int context_count = 0;
            char context_buf[MAX_TURN_TEXT * MAX_CONTEXT];
            int ctx_pos = 0;
            for (int i = 0; i < nresults && context_count < MAX_CONTEXT; i++) {
                uint16_t tid = result_ids[i];
                if (tid == last_id) continue;
                if (tid < MAX_TURNS && turns[tid].valid) {
                    ctx_pos += snprintf(context_buf + ctx_pos,
                        sizeof(context_buf) - ctx_pos,
                        "- %s\n", turns[tid].text);
                    context_count++;
                }
            }

            if (context_count > 0) {
                pos += snprintf(prompt + pos, prompt_max - pos,
                    "\n\nRelevant prior conversation:\n%s", context_buf);
            }
        }
    }

    pos += snprintf(prompt + pos, prompt_max - pos, "<|im_end|>\n");

    /* Include most recent turn as a user/assistant exchange for continuity */
    if (use_memory && n_turns >= 1 && turns[n_turns - 1].valid) {
        /* The stored turn text is "User: ...\nAssistant: ..." — split it */
        const char *turn_text = turns[n_turns - 1].text;
        const char *asst_part = strstr(turn_text, "\nAssistant: ");
        if (asst_part) {
            /* User part: skip "User: " prefix */
            const char *user_part = turn_text;
            if (strncmp(user_part, "User: ", 6) == 0) user_part += 6;
            int user_len = (int)(asst_part - user_part);

            pos += snprintf(prompt + pos, prompt_max - pos,
                "<|im_start|>user\n%.*s<|im_end|>\n"
                "<|im_start|>assistant\n%s<|im_end|>\n",
                user_len, user_part, asst_part + 12);
        }
    }

    /* Current user message */
    pos += snprintf(prompt + pos, prompt_max - pos,
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n", user_input);

    return pos;
}

/* ---------- Main ---------- */

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    int n_threads   = 2;
    int n_ctx       = 4096;
    int n_predict   = MAX_RESPONSE;
    float temp      = 0.7f;
    int top_k       = 40;
    float top_p     = 0.9f;
    int use_memory  = 1;

    /* Parse options */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc)
            n_ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n_predict = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc)
            temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc)
            top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
            top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--no-memory") == 0)
            use_memory = 0;
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* ---------- Initialize backends ---------- */
    fprintf(stderr, "[moltar] Loading backends...\n");
    ggml_backend_load_all();

    /* ---------- Load model ---------- */
    fprintf(stderr, "[moltar] Loading model: %s\n", model_path);

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  /* CPU only */

    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "[moltar] ERROR: failed to load model\n");
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    fprintf(stderr, "[moltar] Model loaded: n_embd=%d\n", n_embd);

    /* ---------- Create context with embeddings ---------- */
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx        = n_ctx;
    ctx_params.n_batch      = n_ctx;
    ctx_params.n_threads     = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.embeddings   = 1;  /* KEY: enable hidden state extraction */
    ctx_params.no_perf      = 1;

    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "[moltar] ERROR: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    /* ---------- Create sampler ---------- */
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = 1;
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);

    if (temp < 0.01f) {
        /* Greedy */
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    /* ---------- Initialize projection + LCVDB ---------- */
    if (use_memory) {
        fprintf(stderr, "[moltar] Initializing working memory (LCVDB + projection)\n");
        moltar_proj_init(&proj, n_embd, PROJ_SEED);

        posix_memalign(&topo_buf, 64, MAX_TURNS * sizeof(lcvdb_topo_t));
        posix_memalign(&vec_buf, 64, MAX_TURNS * sizeof(lcvdb_vec_t));
        lcvdb_init(&vdb, topo_buf, vec_buf, MAX_TURNS);

        fprintf(stderr, "[moltar] Projection: %dD -> 48D int8 (%.1f KB matrix)\n",
                n_embd, (float)(48 * n_embd) / 1024.0f);
    }

    /* ---------- Token buffers ---------- */
    llama_token *prompt_tokens = (llama_token *)malloc(MAX_PROMPT_TOKENS * sizeof(llama_token));
    char prompt_buf[MAX_INPUT * 4];  /* room for context + input */
    char input_buf[MAX_INPUT];
    char response_buf[MAX_TURN_TEXT];

    fprintf(stderr, "[moltar] Ready. Type your message (Ctrl-D or 'quit' to exit).\n");
    fprintf(stderr, "[moltar] Memory: %s | Threads: %d | Context: %d | Temp: %.1f\n\n",
            use_memory ? "ON" : "OFF", n_threads, n_ctx, temp);

    /* ---------- Conversation loop ---------- */
    while (1) {
        /* Read user input */
        fprintf(stdout, "\nYou: ");
        fflush(stdout);

        if (!fgets(input_buf, sizeof(input_buf), stdin)) {
            fprintf(stdout, "\n");
            break;  /* EOF */
        }

        /* Strip trailing newline */
        int len = (int)strlen(input_buf);
        while (len > 0 && (input_buf[len - 1] == '\n' || input_buf[len - 1] == '\r'))
            input_buf[--len] = '\0';

        if (len == 0) continue;
        if (strcmp(input_buf, "quit") == 0 || strcmp(input_buf, "exit") == 0) break;

        /* Special commands */
        if (strcmp(input_buf, "/memory") == 0) {
            fprintf(stdout, "Working memory: %d turns stored", n_turns);
            if (use_memory)
                fprintf(stdout, ", %u nodes in LCVDB graph\n", vdb.node_count);
            else
                fprintf(stdout, " (memory disabled)\n");
            continue;
        }

        /* Build prompt with working memory context */
        int prompt_len = build_prompt(prompt_buf, sizeof(prompt_buf),
                                      input_buf, use_memory);

        /* Tokenize */
        int n_tokens = -llama_tokenize(vocab, prompt_buf, prompt_len,
                                       NULL, 0, 1, 1);
        if (n_tokens > MAX_PROMPT_TOKENS) {
            fprintf(stderr, "[moltar] WARN: prompt too long (%d tokens), truncating\n",
                    n_tokens);
            n_tokens = MAX_PROMPT_TOKENS;
        }
        llama_tokenize(vocab, prompt_buf, prompt_len,
                       prompt_tokens, n_tokens, 1, 1);

        /* Clear KV cache for fresh generation */
        llama_memory_clear(llama_get_memory(ctx), 1);

        /* Process prompt */
        struct llama_batch batch = llama_batch_get_one(prompt_tokens, n_tokens);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[moltar] ERROR: failed to decode prompt\n");
            continue;
        }

        /* Generate response */
        fprintf(stdout, "Moltar: ");
        fflush(stdout);

        int response_len = 0;
        int n_generated = 0;

        for (int i = 0; i < n_predict; i++) {
            /* Sample next token */
            llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

            /* Check for end of generation */
            if (llama_vocab_is_eog(vocab, new_token)) break;

            /* Detokenize and print */
            char piece[128];
            int piece_len = llama_token_to_piece(vocab, new_token,
                                                  piece, sizeof(piece) - 1, 0, 1);
            if (piece_len > 0) {
                piece[piece_len] = '\0';
                fprintf(stdout, "%s", piece);
                fflush(stdout);

                /* Accumulate response text */
                if (response_len + piece_len < MAX_TURN_TEXT - 1) {
                    memcpy(response_buf + response_len, piece, piece_len);
                    response_len += piece_len;
                }
            }

            n_generated++;

            /* Decode the new token */
            batch = llama_batch_get_one(&new_token, 1);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "\n[moltar] ERROR: decode failed at token %d\n", i);
                break;
            }
        }

        response_buf[response_len] = '\0';
        fprintf(stdout, "\n");

        /* ---------- Store in working memory ---------- */
        if (use_memory && n_turns < MAX_TURNS) {
            /* Extract hidden state from the last generated token */
            float *embd = llama_get_embeddings_ith(ctx, -1);

            if (embd) {
                /* Project 2048D float -> 48D int8 */
                int8_t vec48[48];
                moltar_proj_apply(&proj, embd, vec48);

                /* Insert into LCVDB */
                lcvdb_insert(&vdb, vec48, (uint32_t)n_turns);

                /* Store turn text */
                snprintf(turns[n_turns].text, MAX_TURN_TEXT,
                         "User: %s\nAssistant: %s", input_buf, response_buf);
                turns[n_turns].valid = 1;
                n_turns++;

                /* Debug info */
                fprintf(stderr, "[moltar] Turn %d stored (%d tokens, %d generated)\n",
                        n_turns, n_tokens, n_generated);
            } else {
                fprintf(stderr, "[moltar] WARN: could not extract embeddings\n");
            }
        }
    }

    /* ---------- Cleanup ---------- */
    fprintf(stderr, "\n[moltar] Session: %d turns\n", n_turns);

    free(prompt_tokens);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (use_memory) {
        free(topo_buf);
        free(vec_buf);
    }

    return 0;
}
