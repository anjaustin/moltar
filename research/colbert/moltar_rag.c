/*
 * moltar_rag.c — ColBERT MaxSim search tool for RAG pipeline
 *
 * Usage:
 *   moltar_rag search <query.emb> <index_dir> <n_chunks> <top_k>
 *
 * Reads llama-embedding text output (.emb files), parses 128D float embeddings,
 * quantizes to int8, runs MaxSim search, outputs ranked results.
 *
 * .emb file format (llama-embedding --pooling none --embd-output-format raw):
 *   f1 f2 f3 ... f128  (128 space-separated floats per line, one per token)
 */

#include "colbert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CHUNKS     512
#define MAX_LINE_LEN   8192
#define MAX_TOKENS_PER_CHUNK 512

/*
 * Parse a raw embedding line (space-separated floats).
 * Format: "f1 f2 f3 ... f128\n"
 * Returns number of floats parsed.
 */
static int parse_raw_line(const char *line, float *out, int max_dim)
{
    const char *p = line;
    int n = 0;

    while (n < max_dim) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n') break;

        char *end;
        float v = strtof(p, &end);
        if (end == p) break;
        out[n++] = v;
        p = end;
    }
    return n;
}

/*
 * Load embeddings from a .emb file (llama-embedding output).
 * Returns number of token embeddings loaded.
 * Writes quantized int8 embeddings into out_i8.
 */
static int load_embeddings(const char *path, int8_t *out_i8, int max_tokens)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char line[MAX_LINE_LEN];
    float fbuf[COLBERT_EMB_DIM];
    int n_tokens = 0;

    while (fgets(line, sizeof(line), f) && n_tokens < max_tokens) {
        /* Skip empty lines and lines that don't start with a number or minus */
        char c = line[0];
        if (c != '-' && (c < '0' || c > '9')) continue;

        int n = parse_raw_line(line, fbuf, COLBERT_EMB_DIM);
        if (n != COLBERT_EMB_DIM) continue; /* skip incomplete lines */

        colbert_quantize_f32_to_i8(fbuf, &out_i8[n_tokens * COLBERT_EMB_DIM],
                                    COLBERT_EMB_DIM);
        n_tokens++;
    }

    fclose(f);
    return n_tokens;
}

/*
 * Command: search
 *
 * moltar_rag search <query.emb> <index_dir> <n_chunks> <top_k>
 *
 * Outputs: rank|score|chunk_id  (one per line)
 */
static int cmd_search(const char *query_emb_path, const char *index_dir,
                      int n_chunks, int top_k)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Load query embeddings */
    int8_t query_i8[COLBERT_MAX_QTOKENS * COLBERT_EMB_DIM];
    int n_qtokens = load_embeddings(query_emb_path, query_i8,
                                     COLBERT_MAX_QTOKENS);
    if (n_qtokens == 0) {
        fprintf(stderr, "moltar_rag: failed to load query embeddings from %s\n",
                query_emb_path);
        return 1;
    }

    /* Load document chunk embeddings and compute MaxSim scores */
    typedef struct {
        int32_t score;
        int     chunk_id;
    } scored_t;

    scored_t scores[MAX_CHUNKS];
    int n_scored = 0;

    for (int c = 0; c < n_chunks && c < MAX_CHUNKS; c++) {
        char emb_path[512];
        snprintf(emb_path, sizeof(emb_path), "%s/chunk_%d.emb", index_dir, c);

        int8_t doc_i8[MAX_TOKENS_PER_CHUNK * COLBERT_EMB_DIM];
        int n_dtokens = load_embeddings(emb_path, doc_i8, MAX_TOKENS_PER_CHUNK);
        if (n_dtokens == 0) continue;

        int32_t s = colbert_maxsim_i8(query_i8, doc_i8, n_qtokens, n_dtokens);
        scores[n_scored].score = s;
        scores[n_scored].chunk_id = c;
        n_scored++;
    }

    /* Sort by score descending (insertion sort, small N) */
    for (int i = 1; i < n_scored; i++) {
        scored_t key = scores[i];
        int j = i - 1;
        while (j >= 0 && scores[j].score < key.score) {
            scores[j + 1] = scores[j];
            j--;
        }
        scores[j + 1] = key;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    int64_t ns = (t1.tv_sec - t0.tv_sec) * 1000000000LL +
                 (t1.tv_nsec - t0.tv_nsec);

    /* Output top-k results */
    int k = top_k < n_scored ? top_k : n_scored;
    for (int i = 0; i < k; i++) {
        printf("%d|%d|%d\n", i + 1, scores[i].score, scores[i].chunk_id);
    }

    fprintf(stderr, "moltar_rag: searched %d chunks in %lld us "
            "(%d query tokens)\n",
            n_scored, (long long)(ns / 1000), n_qtokens);

    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: moltar_rag search <query.emb> <index_dir> "
                "<n_chunks> <top_k>\n");
        return 1;
    }

    if (strcmp(argv[1], "search") == 0) {
        if (argc < 6) {
            fprintf(stderr, "Usage: moltar_rag search <query.emb> <index_dir> "
                    "<n_chunks> <top_k>\n");
            return 1;
        }
        return cmd_search(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]));
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
}
