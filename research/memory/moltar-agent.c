/* ==========================================================================
 * Moltar Agent — Multi-turn LLM with Three-Layer Memory
 * ==========================================================================
 * Links against libllama.so (LFM2 inference) + LCVDB (semantic working memory).
 * Optionally calls ColBERT RAG pipeline via subprocess for knowledge retrieval.
 *
 * Three memory layers:
 *   1. LCVDB working memory  (23 us)  — conversation turn recall
 *   2. ColBERT knowledge     (80 ms)  — document retrieval via subprocess
 *   3. LFM2-1.2B generation  (21 tok/s) — with merged context
 *
 * Each conversation turn:
 *   1. Read user input
 *   2. Search LCVDB for top-3 related prior turns
 *   3. (Optional) Run ColBERT retrieval for knowledge context
 *   4. Build ChatML prompt merging working memory + knowledge
 *   5. Generate response via llama_decode + sampling
 *   6. Extract hidden state → project → insert into LCVDB
 *   7. Print response, loop
 *
 * Build with NDK clang, link against libllama.so + LCVDB (static).
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include "llama.h"
#include "ggml-backend.h"

#include "project.h"
#include "../lcvdb/lcvdb.h"

/* ---------- Configuration ---------- */

#define MAX_TURNS       256      /* max conversation turns in LCVDB         */
#define MAX_CONTEXT     3        /* top-K related turns to prepend          */
#undef  MAX_INPUT
#define MAX_INPUT       1024     /* max user input chars                    */
#define MAX_RESPONSE    512      /* max generated tokens per turn           */
#define MAX_PROMPT_TOKENS 2048   /* max tokens in full prompt               */
#define MAX_TURN_TEXT   2048     /* max stored text per turn                */
#define MAX_KNOWLEDGE   4096     /* max chars of RAG knowledge context      */
#define MAX_PROMPT_BUF  16384    /* prompt buffer (room for all context)    */

/* Projection seed "MOLT" */
#define PROJ_SEED       0x4D4F4C54

/* RAG defaults (all paths on device) */
#define RAG_BASE        "/data/local/tmp"
#define RAG_COLBERT     RAG_BASE "/LFM2-ColBERT-350M-Q4_0.gguf"
#define RAG_EMBEDDING   RAG_BASE "/llama-embedding"
#define RAG_SEARCH      RAG_BASE "/moltar_rag"
#define RAG_INDEX       RAG_BASE "/rag_index"
#define RAG_TOP_K       3        /* top-K knowledge chunks to retrieve      */

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

/* ---------- Session Persistence ---------- */

#define SESSION_MAGIC   0x4D4F4C54  /* "MOLT" */
#define SESSION_VERSION 1

/* Save session state (LCVDB + turns) to a binary file.
 * Format: [magic][version][n_turns][vdb scalars][topo][vec][turns] */
static int session_save(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t magic = SESSION_MAGIC;
    uint32_t version = SESSION_VERSION;
    uint32_t nt = (uint32_t)n_turns;

    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&nt, 4, 1, f);

    /* Write LCVDB scalar state (first 32 bytes: node_count..prng_state) */
    fwrite(&vdb, 32, 1, f);

    /* Write topology and vector arrays (only used slots) */
    fwrite(topo_buf, sizeof(lcvdb_topo_t), nt, f);
    fwrite(vec_buf, sizeof(lcvdb_vec_t), nt, f);

    /* Write turn text */
    fwrite(turns, sizeof(turn_t), nt, f);

    fclose(f);
    return 0;
}

/* Load session state from a binary file. Returns 0 on success, -1 on error. */
static int session_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic, version, nt;
    if (fread(&magic, 4, 1, f) != 1 || magic != SESSION_MAGIC) {
        fclose(f); return -1;
    }
    if (fread(&version, 4, 1, f) != 1 || version != SESSION_VERSION) {
        fclose(f); return -1;
    }
    if (fread(&nt, 4, 1, f) != 1 || nt > MAX_TURNS) {
        fclose(f); return -1;
    }

    /* Read LCVDB scalar state (32 bytes), then restore pointers */
    fread(&vdb, 32, 1, f);
    vdb.topo_array = (lcvdb_topo_t *)topo_buf;
    vdb.vec_array  = (lcvdb_vec_t *)vec_buf;
    vdb.max_nodes  = MAX_TURNS;

    /* Read topology and vector arrays */
    fread(topo_buf, sizeof(lcvdb_topo_t), nt, f);
    fread(vec_buf, sizeof(lcvdb_vec_t), nt, f);

    /* Read turn text */
    fread(turns, sizeof(turn_t), nt, f);
    n_turns = (int)nt;

    fclose(f);
    return 0;
}

/* ---------- RAG Configuration ---------- */

typedef struct {
    int   enabled;
    char  colbert_model[256];
    char  embedding_bin[256];
    char  search_bin[256];
    char  index_dir[256];
    int   top_k;
    int   n_chunks;        /* from manifest, 0 = not loaded */
} rag_config_t;

static rag_config_t rag_cfg;

/* ---------- RAG: Knowledge Retrieval via Subprocess ---------- */

/* Load the chunk count from the index manifest */
static int rag_load_manifest(rag_config_t *cfg) {
    char path[512];
    snprintf(path, sizeof(path), "%s/manifest.txt", cfg->index_dir);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int n = 0;
    if (fscanf(f, "%d", &n) != 1) n = 0;
    fclose(f);
    return n;
}

/* Run ColBERT retrieval for a query. Returns knowledge text in out_buf.
 *
 * Steps:
 *   1. llama-embedding embeds the query → tmp file
 *   2. moltar_rag search finds top-K chunks
 *   3. Read chunk text files and concatenate
 *
 * This shells out to external binaries because the ColBERT model
 * (209 MB) can't coexist in RAM with LFM2-1.2B (661 MB).
 * The OS handles memory pressure via mmap page eviction/fault-in.
 */
static int rag_retrieve(const rag_config_t *cfg, const char *query,
                        char *out_buf, int out_max,
                        const char *llm_model_path) {
    if (!cfg->enabled || cfg->n_chunks <= 0) return 0;

    char cmd[2048];
    char emb_path[256];
    char results_path[256];
    char query_path[256];

    snprintf(emb_path, sizeof(emb_path), "%s/rag_query_agent.emb", RAG_BASE);
    snprintf(results_path, sizeof(results_path), "%s/rag_results_agent.txt", RAG_BASE);
    snprintf(query_path, sizeof(query_path), "%s/rag_query_agent.txt", RAG_BASE);

    /* Write query to temp file to avoid shell injection.
     * No user input ever touches the shell command string. */
    FILE *qf = fopen(query_path, "w");
    if (!qf) {
        fprintf(stderr, "[rag] WARN: could not write query file %s\n", query_path);
        return 0;
    }
    fputs(query, qf);
    fclose(qf);

    /* Step 1: Embed query with ColBERT model (read from file, not -p) */
    fprintf(stderr, "[rag] Embedding query...\n");
    snprintf(cmd, sizeof(cmd),
        "export LD_LIBRARY_PATH=%s && "
        "taskset c0 %s "
        "-m %s -c 1024 -f %s "
        "--pooling none --embd-normalize -1 --embd-output-format raw "
        "-t 2 --no-warmup 2>/dev/null > %s",
        RAG_BASE, cfg->embedding_bin, cfg->colbert_model, query_path, emb_path);

    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "[rag] WARN: embedding failed (ret=%d)\n", ret);
        return 0;
    }

    /* Prefetch LLM model pages back into page cache.
     * The ColBERT subprocess likely evicted LFM2 mmap pages. Start
     * a background cat to /dev/null to fault them back in while we
     * do the MaxSim search. This overlaps I/O with computation. */
    if (llm_model_path) {
        snprintf(cmd, sizeof(cmd),
            "cat %s > /dev/null 2>/dev/null &", llm_model_path);
        system(cmd);
        fprintf(stderr, "[rag] Prefetching LLM model pages...\n");
    }

    /* Step 2: MaxSim search */
    fprintf(stderr, "[rag] Searching %d chunks...\n", cfg->n_chunks);
    snprintf(cmd, sizeof(cmd),
        "taskset c0 %s search %s %s %d %d > %s 2>/dev/null",
        cfg->search_bin, emb_path, cfg->index_dir,
        cfg->n_chunks, cfg->top_k, results_path);

    ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "[rag] WARN: search failed (ret=%d)\n", ret);
        return 0;
    }

    /* Step 3: Read results and collect chunk text */
    FILE *f = fopen(results_path, "r");
    if (!f) return 0;

    int pos = 0;
    char line[256];
    int n_results = 0;

    while (fgets(line, sizeof(line), f) && n_results < cfg->top_k) {
        int rank, score, chunk_id;
        if (sscanf(line, "%d|%d|%d", &rank, &score, &chunk_id) != 3)
            continue;

        /* Read chunk text file */
        char chunk_path[512];
        snprintf(chunk_path, sizeof(chunk_path),
                 "%s/chunk_%d.txt", cfg->index_dir, chunk_id);

        FILE *cf = fopen(chunk_path, "r");
        if (!cf) continue;

        /* Read chunk text */
        char chunk_text[2048];
        int chunk_len = (int)fread(chunk_text, 1, sizeof(chunk_text) - 1, cf);
        fclose(cf);
        chunk_text[chunk_len] = '\0';

        /* Strip trailing whitespace */
        while (chunk_len > 0 &&
               (chunk_text[chunk_len-1] == '\n' || chunk_text[chunk_len-1] == ' '))
            chunk_text[--chunk_len] = '\0';

        /* Append to output */
        if (pos + chunk_len + 10 < out_max) {
            pos += snprintf(out_buf + pos, out_max - pos,
                            "- %s\n", chunk_text);
            n_results++;
        }

        fprintf(stderr, "[rag]   Chunk %d (score=%d): %.60s...\n",
                chunk_id, score, chunk_text);
    }

    fclose(f);
    fprintf(stderr, "[rag] Retrieved %d knowledge chunks\n", n_results);
    return pos;
}

/* ---------- Helpers ---------- */

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s <model.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  -t N         threads for generation (default: 2)\n"
        "  -c N         context size (default: 4096)\n"
        "  -n N         max tokens per response (default: %d)\n"
        "  --temp F     temperature (default: 0.7)\n"
        "  --top-k N    top-k sampling (default: 40)\n"
        "  --top-p F    top-p sampling (default: 0.9)\n"
        "  --no-memory  disable LCVDB working memory\n"
        "  --rag        enable ColBERT knowledge retrieval\n"
        "  --no-rag     disable ColBERT knowledge retrieval (default)\n"
        "  --rag-index DIR  RAG index directory (default: %s)\n"
        "  --session FILE   session file for persistent memory\n"
        "\n"
        "Commands (during conversation):\n"
        "  /memory      show working memory status\n"
        "  /rag         toggle RAG on/off\n"
        "  /rag status  show RAG configuration\n"
        "  /save        save session to file (requires --session)\n"
        "  /load        reload session from file (requires --session)\n"
        "  /ingest FILE ingest a text file into the knowledge base\n"
        "  /learn TEXT  learn a piece of text (added as a knowledge chunk)\n"
        "\n"
        "Runs a multi-turn conversation with three-layer memory.\n"
        "Type your message and press Enter. Ctrl-D or 'quit' to exit.\n",
        argv0, MAX_RESPONSE, RAG_INDEX);
}

/* Build prompt with working memory + knowledge context in ChatML format.
 *
 * LFM2 expects:
 *   <|im_start|>system\n...<|im_end|>\n
 *   <|im_start|>user\n...<|im_end|>\n
 *   <|im_start|>assistant\n
 *
 * BOS token (id=1) is added automatically by llama_tokenize(add_special=true).
 * The model generates until <|im_end|> (token 7, EOG).
 */
static int build_prompt(char *prompt, int prompt_max,
                        const char *user_input,
                        int use_memory,
                        const char *knowledge_ctx) {
    int pos = 0;

    /* System message */
    pos += snprintf(prompt + pos, prompt_max - pos,
        "<|im_start|>system\n"
        "You are Moltar, a helpful AI assistant running on a Motorola phone. "
        "Be concise and direct.");

    /* Knowledge context from ColBERT RAG */
    if (knowledge_ctx && knowledge_ctx[0]) {
        pos += snprintf(prompt + pos, prompt_max - pos,
            "\n\nKnowledge base:\n%s", knowledge_ctx);
    }

    /* Working memory context from LCVDB */
    if (use_memory && n_turns > 0) {
        if (n_turns >= 1) {
            uint16_t last_id = (uint16_t)(n_turns - 1);

            uint16_t result_ids[MAX_CONTEXT + 1];
            int32_t  result_scores[MAX_CONTEXT + 1];
            int nresults = lcvdb_search(&vdb,
                (const int8_t *)((char *)vec_buf + last_id * sizeof(lcvdb_vec_t)),
                MAX_CONTEXT + 1, result_ids, result_scores);

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
        const char *turn_text = turns[n_turns - 1].text;
        const char *asst_part = strstr(turn_text, "\nAssistant: ");
        if (asst_part) {
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
    const char *session_path = NULL;  /* --session FILE for persistence */

    /* RAG defaults */
    rag_cfg.enabled = 0;
    snprintf(rag_cfg.colbert_model, sizeof(rag_cfg.colbert_model), "%s", RAG_COLBERT);
    snprintf(rag_cfg.embedding_bin, sizeof(rag_cfg.embedding_bin), "%s", RAG_EMBEDDING);
    snprintf(rag_cfg.search_bin, sizeof(rag_cfg.search_bin), "%s", RAG_SEARCH);
    snprintf(rag_cfg.index_dir, sizeof(rag_cfg.index_dir), "%s", RAG_INDEX);
    rag_cfg.top_k = RAG_TOP_K;
    rag_cfg.n_chunks = 0;

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
        else if (strcmp(argv[i], "--rag") == 0)
            rag_cfg.enabled = 1;
        else if (strcmp(argv[i], "--no-rag") == 0)
            rag_cfg.enabled = 0;
        else if (strcmp(argv[i], "--rag-index") == 0 && i + 1 < argc)
            snprintf(rag_cfg.index_dir, sizeof(rag_cfg.index_dir), "%s", argv[++i]);
        else if (strcmp(argv[i], "--session") == 0 && i + 1 < argc)
            session_path = argv[++i];
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* ---------- Initialize RAG ---------- */
    if (rag_cfg.enabled) {
        rag_cfg.n_chunks = rag_load_manifest(&rag_cfg);
        if (rag_cfg.n_chunks > 0) {
            fprintf(stderr, "[moltar] RAG enabled: %d indexed chunks in %s\n",
                    rag_cfg.n_chunks, rag_cfg.index_dir);
        } else {
            fprintf(stderr, "[moltar] WARN: RAG enabled but no index found at %s\n",
                    rag_cfg.index_dir);
            rag_cfg.enabled = 0;
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

        if (posix_memalign(&topo_buf, 64, MAX_TURNS * sizeof(lcvdb_topo_t)) != 0 ||
            posix_memalign(&vec_buf, 64, MAX_TURNS * sizeof(lcvdb_vec_t)) != 0) {
            fprintf(stderr, "[moltar] ERROR: failed to allocate LCVDB buffers\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        lcvdb_init(&vdb, topo_buf, vec_buf, MAX_TURNS);

        fprintf(stderr, "[moltar] Projection: %dD -> 48D int8 (%.1f KB matrix)\n",
                n_embd, (float)(48 * n_embd) / 1024.0f);

        /* Restore session if file exists */
        if (session_path) {
            if (session_load(session_path) == 0) {
                fprintf(stderr, "[moltar] Session restored: %d turns from %s\n",
                        n_turns, session_path);
            } else {
                fprintf(stderr, "[moltar] No existing session at %s (starting fresh)\n",
                        session_path);
            }
        }
    }

    /* ---------- Token buffers ---------- */
    llama_token *prompt_tokens = (llama_token *)malloc(MAX_PROMPT_TOKENS * sizeof(llama_token));
    char *prompt_buf = (char *)malloc(MAX_PROMPT_BUF);
    char input_buf[MAX_INPUT];
    char response_buf[MAX_TURN_TEXT];
    char knowledge_buf[MAX_KNOWLEDGE];

    fprintf(stderr, "[moltar] Ready. Type your message (Ctrl-D or 'quit' to exit).\n");
    fprintf(stderr, "[moltar] Memory: %s | RAG: %s | Session: %s | Threads: %d | Temp: %.1f\n\n",
            use_memory ? "ON" : "OFF",
            rag_cfg.enabled ? "ON" : "OFF",
            session_path ? session_path : "none",
            n_threads, temp);

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
        if (strcmp(input_buf, "/rag") == 0) {
            rag_cfg.enabled = !rag_cfg.enabled;
            if (rag_cfg.enabled && rag_cfg.n_chunks == 0) {
                rag_cfg.n_chunks = rag_load_manifest(&rag_cfg);
            }
            fprintf(stdout, "RAG: %s (%d chunks indexed)\n",
                    rag_cfg.enabled ? "ON" : "OFF", rag_cfg.n_chunks);
            continue;
        }
        if (strcmp(input_buf, "/rag status") == 0) {
            fprintf(stdout, "RAG: %s\n", rag_cfg.enabled ? "ON" : "OFF");
            fprintf(stdout, "  Index: %s (%d chunks)\n",
                    rag_cfg.index_dir, rag_cfg.n_chunks);
            fprintf(stdout, "  ColBERT: %s\n", rag_cfg.colbert_model);
            fprintf(stdout, "  Top-K: %d\n", rag_cfg.top_k);
            continue;
        }
        if (strcmp(input_buf, "/save") == 0) {
            if (!session_path) {
                fprintf(stdout, "No session file configured. Use --session FILE\n");
            } else if (session_save(session_path) == 0) {
                fprintf(stdout, "Session saved: %d turns -> %s\n", n_turns, session_path);
            } else {
                fprintf(stdout, "ERROR: failed to save session to %s\n", session_path);
            }
            continue;
        }
        if (strcmp(input_buf, "/load") == 0) {
            if (!session_path) {
                fprintf(stdout, "No session file configured. Use --session FILE\n");
            } else if (session_load(session_path) == 0) {
                fprintf(stdout, "Session loaded: %d turns from %s\n", n_turns, session_path);
            } else {
                fprintf(stdout, "ERROR: failed to load session from %s\n", session_path);
            }
            continue;
        }
        if (strncmp(input_buf, "/ingest ", 8) == 0) {
            const char *file_path = input_buf + 8;
            while (*file_path == ' ') file_path++;
            if (*file_path == '\0') {
                fprintf(stdout, "Usage: /ingest <file_path>\n");
                continue;
            }
            fprintf(stdout, "Ingesting %s...\n", file_path);
            fflush(stdout);

            char cmd[2048];
            snprintf(cmd, sizeof(cmd),
                "taskset c0 %s ingest %s %s %s %s 2>&1",
                rag_cfg.search_bin, file_path, rag_cfg.index_dir,
                rag_cfg.embedding_bin, rag_cfg.colbert_model);
            int ret = system(cmd);

            if (ret == 0) {
                /* Reload manifest to pick up new chunks */
                rag_cfg.n_chunks = rag_load_manifest(&rag_cfg);
                fprintf(stdout, "Ingestion complete. Index now has %d chunks.\n",
                        rag_cfg.n_chunks);
                if (!rag_cfg.enabled) {
                    rag_cfg.enabled = 1;
                    fprintf(stdout, "RAG auto-enabled.\n");
                }
            } else {
                fprintf(stdout, "ERROR: ingestion failed (ret=%d)\n", ret);
            }

            /* Prefetch LLM model back since ColBERT evicted it */
            snprintf(cmd, sizeof(cmd),
                "cat %s > /dev/null 2>/dev/null &", model_path);
            system(cmd);
            continue;
        }
        if (strncmp(input_buf, "/learn ", 7) == 0) {
            const char *text = input_buf + 7;
            while (*text == ' ') text++;
            if (*text == '\0') {
                fprintf(stdout, "Usage: /learn <text to remember>\n");
                continue;
            }

            /* Write text to a temp file, then ingest it */
            const char *learn_path = RAG_BASE "/learn_tmp.txt";
            FILE *lf = fopen(learn_path, "w");
            if (!lf) {
                fprintf(stdout, "ERROR: cannot write temp file\n");
                continue;
            }
            fputs(text, lf);
            fclose(lf);

            fprintf(stdout, "Learning: %.60s%s\n", text,
                    (int)strlen(text) > 60 ? "..." : "");
            fflush(stdout);

            char cmd[2048];
            snprintf(cmd, sizeof(cmd),
                "taskset c0 %s ingest %s %s %s %s 2>&1",
                rag_cfg.search_bin, learn_path, rag_cfg.index_dir,
                rag_cfg.embedding_bin, rag_cfg.colbert_model);
            int ret = system(cmd);

            remove(learn_path);

            if (ret == 0) {
                rag_cfg.n_chunks = rag_load_manifest(&rag_cfg);
                fprintf(stdout, "Learned. Index now has %d chunks.\n",
                        rag_cfg.n_chunks);
                if (!rag_cfg.enabled) {
                    rag_cfg.enabled = 1;
                    fprintf(stdout, "RAG auto-enabled.\n");
                }
            } else {
                fprintf(stdout, "ERROR: learning failed (ret=%d)\n", ret);
            }

            /* Prefetch LLM model back */
            snprintf(cmd, sizeof(cmd),
                "cat %s > /dev/null 2>/dev/null &", model_path);
            system(cmd);
            continue;
        }

        /* ---------- ColBERT knowledge retrieval ---------- */
        knowledge_buf[0] = '\0';
        if (rag_cfg.enabled) {
            rag_retrieve(&rag_cfg, input_buf, knowledge_buf, MAX_KNOWLEDGE,
                        model_path);
        }

        /* Build prompt with working memory + knowledge context */
        int prompt_len = build_prompt(prompt_buf, MAX_PROMPT_BUF,
                                      input_buf, use_memory, knowledge_buf);

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

    /* Auto-save session on exit */
    if (session_path && use_memory && n_turns > 0) {
        if (session_save(session_path) == 0) {
            fprintf(stderr, "[moltar] Session saved: %d turns -> %s\n",
                    n_turns, session_path);
        } else {
            fprintf(stderr, "[moltar] WARN: failed to save session to %s\n",
                    session_path);
        }
    }

    free(prompt_tokens);
    free(prompt_buf);
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
