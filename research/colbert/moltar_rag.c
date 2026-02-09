/*
 * moltar_rag.c — ColBERT MaxSim search tool for RAG pipeline
 *
 * Usage:
 *   moltar_rag search <query.emb> <index_dir> <n_chunks> <top_k>
 *
 * Reads llama-embedding text output (.emb files), parses 128D float embeddings,
 * quantizes to int8, runs IDF-weighted MaxSim search, outputs ranked results.
 *
 * IDF-weighted MaxSim:
 *   Standard MaxSim sums the best dot product for each query token across all
 *   document tokens. This treats all query tokens equally, so common tokens
 *   (matching well in every chunk) dominate the score. IDF weighting computes
 *   how "distinctive" each query token is: tokens that match uniformly across
 *   chunks get downweighted, tokens that match strongly in few chunks get
 *   upweighted. This dramatically improves retrieval quality on small corpora.
 *
 * .emb file format (llama-embedding --pooling none --embd-output-format raw):
 *   f1 f2 f3 ... f128  (128 space-separated floats per line, one per token)
 */

#include "colbert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
 * Compute the max dot product of a single query token against all tokens
 * in a document chunk. This is the per-query-token MaxSim contribution.
 *
 * Returns max_j dot(query_token, doc_token_j)
 */
static int32_t max_dot_single(const int8_t *query_token,
                               const int8_t *doc_tokens, int n_dtokens)
{
    int32_t best = -2147483647; /* INT32_MIN+1 to avoid overflow */
    for (int j = 0; j < n_dtokens; j++) {
        int32_t d = colbert_dot_i8(query_token,
                                    &doc_tokens[j * COLBERT_EMB_DIM]);
        if (d > best) best = d;
    }
    return best;
}

/*
 * Command: search (IDF-weighted MaxSim)
 *
 * moltar_rag search <query.emb> <index_dir> <n_chunks> <top_k>
 *
 * Algorithm:
 *   1. Load all chunk embeddings
 *   2. For each query token q_i, for each chunk c:
 *      per_token_score[i][c] = max_j dot(q_i, doc_c_j)
 *   3. For each query token, compute weight = stdev / mean across chunks
 *      (high = distinctive, low = common)
 *   4. Clamp weights to [0.1, 3.0] to avoid extreme values
 *   5. Final score per chunk = Σ_i weight_i * per_token_score[i][c]
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

    /* Load all chunk embeddings into memory */
    typedef struct {
        int8_t *emb;        /* token embeddings (heap-allocated) */
        int     n_tokens;
        int     chunk_id;
    } chunk_data_t;

    chunk_data_t chunks[MAX_CHUNKS];
    int n_loaded = 0;

    for (int c = 0; c < n_chunks && c < MAX_CHUNKS; c++) {
        char emb_path[512];
        snprintf(emb_path, sizeof(emb_path), "%s/chunk_%d.emb", index_dir, c);

        int8_t *buf = (int8_t *)malloc(MAX_TOKENS_PER_CHUNK * COLBERT_EMB_DIM);
        if (!buf) continue;

        int n_dtokens = load_embeddings(emb_path, buf, MAX_TOKENS_PER_CHUNK);
        if (n_dtokens == 0) { free(buf); continue; }

        chunks[n_loaded].emb = buf;
        chunks[n_loaded].n_tokens = n_dtokens;
        chunks[n_loaded].chunk_id = c;
        n_loaded++;
    }

    if (n_loaded == 0) {
        fprintf(stderr, "moltar_rag: no chunks loaded from %s\n", index_dir);
        return 1;
    }

    /*
     * Phase 1: Compute per-query-token, per-chunk max dot products.
     *
     * per_token[q][c] = max_j dot(query_token_q, chunk_c_token_j)
     *
     * This decomposes MaxSim into individual query token contributions.
     */
    /* Heap-allocated: up to 32 query tokens × 512 chunks × 4 bytes = 64 KB */
    int32_t (*per_token)[MAX_CHUNKS] = (int32_t (*)[MAX_CHUNKS])
        malloc(COLBERT_MAX_QTOKENS * MAX_CHUNKS * sizeof(int32_t));
    if (!per_token) {
        fprintf(stderr, "moltar_rag: failed to allocate per_token array\n");
        for (int c = 0; c < n_loaded; c++) free(chunks[c].emb);
        return 1;
    }

    for (int q = 0; q < n_qtokens; q++) {
        const int8_t *qt = &query_i8[q * COLBERT_EMB_DIM];
        for (int c = 0; c < n_loaded; c++) {
            per_token[q][c] = max_dot_single(qt, chunks[c].emb,
                                              chunks[c].n_tokens);
        }
    }

    /*
     * Phase 2: Mean-centered MaxSim scoring.
     *
     * For each query token, subtract the mean of its per-chunk scores.
     * This removes the "baseline match" that common tokens contribute
     * equally to all chunks, leaving only the discriminative signal.
     *
     * centered[q][c] = per_token[q][c] - mean(per_token[q][*])
     *
     * Then: score(c) = Σ_q centered[q][c]
     *
     * Intuition:
     * - "Moltar" matches well in all chunks → after centering, contributes ~0
     * - "GPU" matches strongly in chunk 6 → after centering, chunk 6 gets
     *   a large positive contribution, others get negative
     *
     * This is equivalent to IDF weighting where the weight is proportional
     * to how much a token's match varies across chunks.
     */
    float per_token_mean[COLBERT_MAX_QTOKENS];
    float per_token_std[COLBERT_MAX_QTOKENS];

    for (int q = 0; q < n_qtokens; q++) {
        double sum = 0.0;
        for (int c = 0; c < n_loaded; c++)
            sum += (double)per_token[q][c];
        per_token_mean[q] = (float)(sum / n_loaded);

        double var_sum = 0.0;
        for (int c = 0; c < n_loaded; c++) {
            double diff = (double)per_token[q][c] - per_token_mean[q];
            var_sum += diff * diff;
        }
        per_token_std[q] = (float)sqrt(var_sum / n_loaded);
    }

    /* Debug: print per-token stats */
    fprintf(stderr, "moltar_rag: per-token stdev [");
    for (int q = 0; q < n_qtokens; q++)
        fprintf(stderr, "%s%.0f", q ? ", " : "", per_token_std[q]);
    fprintf(stderr, "]\n");

    /*
     * Phase 3: Compute mean-centered scores per chunk.
     *
     * score(c) = Σ_q (per_token[q][c] - mean_q)
     *
     * Tokens with high variance naturally contribute more because their
     * centered values span a wider range. Tokens with near-zero variance
     * (common matches) contribute near-zero regardless of their absolute
     * dot product value.
     */
    typedef struct {
        int32_t score;
        int     chunk_id;
    } scored_t;

    scored_t scores[MAX_CHUNKS];

    /*
     * Compute max stdev across all query tokens (for normalization).
     * Tokens with stdev near max_stdev are "most distinctive".
     */
    float max_stdev = 0.0f;
    for (int q = 0; q < n_qtokens; q++)
        if (per_token_std[q] > max_stdev) max_stdev = per_token_std[q];

    for (int c = 0; c < n_loaded; c++) {
        float fscore = 0.0f;
        for (int q = 0; q < n_qtokens; q++) {
            float centered = (float)per_token[q][c] - per_token_mean[q];

            /*
             * Variance weighting: multiply centered score by (stdev / max_stdev).
             * Tokens with high variance (distinctive) get full weight.
             * Tokens with low variance (common/BOS) get near-zero weight.
             * This is the key innovation over plain centering.
             */
            float w = (max_stdev > 0.0f) ? (per_token_std[q] / max_stdev) : 1.0f;
            fscore += w * centered;
        }

        /*
         * Length normalization: divide by sqrt(n_doc_tokens).
         *
         * Longer documents get higher MaxSim scores because more tokens
         * means more chances for high dot products. Dividing by sqrt(N)
         * compensates for this bias without over-penalizing long docs.
         */
        float len_norm = sqrtf((float)chunks[c].n_tokens);
        if (len_norm > 0.0f)
            fscore /= len_norm;

        /* Scale to preserve resolution in int32 output */
        scores[c].score = (int32_t)(fscore * 1000.0f);
        scores[c].chunk_id = chunks[c].chunk_id;
    }

    /* Sort by score descending (insertion sort, small N) */
    for (int i = 1; i < n_loaded; i++) {
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
    int k = top_k < n_loaded ? top_k : n_loaded;
    for (int i = 0; i < k; i++) {
        printf("%d|%d|%d\n", i + 1, scores[i].score, scores[i].chunk_id);
    }

    fprintf(stderr, "moltar_rag: searched %d chunks in %lld us "
            "(%d query tokens, idf-weighted)\n",
            n_loaded, (long long)(ns / 1000), n_qtokens);

    /* Cleanup */
    free(per_token);
    for (int c = 0; c < n_loaded; c++)
        free(chunks[c].emb);

    return 0;
}

/*
 * Command: ingest
 *
 * moltar_rag ingest <text_file> <index_dir> <embedding_bin> <colbert_model>
 *
 * Reads a text file, splits into paragraph-delimited chunks (min 100 chars),
 * embeds each chunk with llama-embedding, and writes to the index directory.
 * Appends to existing index if manifest already exists.
 *
 * Output: one line per chunk:  chunk_id|n_tokens|n_chars
 */
#define INGEST_MAX_CHUNK_CHARS 1600  /* ~400 tokens at 4 chars/tok */
#define INGEST_MIN_CHUNK_CHARS 80    /* don't embed tiny fragments */
#define INGEST_MAX_CHUNKS      512
#define INGEST_MAX_FILE_SIZE   (256 * 1024)  /* 256 KB max input file */

static int cmd_ingest(const char *text_file, const char *index_dir,
                      const char *embedding_bin, const char *colbert_model)
{
    /* Read the entire text file */
    FILE *f = fopen(text_file, "r");
    if (!f) {
        fprintf(stderr, "moltar_rag: cannot open %s\n", text_file);
        return 1;
    }

    char *file_buf = (char *)malloc(INGEST_MAX_FILE_SIZE);
    if (!file_buf) {
        fprintf(stderr, "moltar_rag: malloc failed\n");
        fclose(f);
        return 1;
    }

    int file_len = (int)fread(file_buf, 1, INGEST_MAX_FILE_SIZE - 1, f);
    fclose(f);
    file_buf[file_len] = '\0';

    /* Read existing manifest to get starting chunk ID (append mode) */
    int start_chunk_id = 0;
    {
        char manifest_path[512];
        snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", index_dir);
        FILE *mf = fopen(manifest_path, "r");
        if (mf) {
            if (fscanf(mf, "%d", &start_chunk_id) != 1) start_chunk_id = 0;
            fclose(mf);
        }
    }

    /*
     * Split into chunks by paragraph (double newline).
     *
     * Strategy: accumulate lines until we hit a blank line and have
     * enough text (>= INGEST_MIN_CHUNK_CHARS), then flush the chunk.
     * If a single paragraph exceeds INGEST_MAX_CHUNK_CHARS, split it
     * at the limit.
     */
    typedef struct {
        char *text;
        int   len;
    } chunk_t;

    chunk_t chunks[INGEST_MAX_CHUNKS];
    int n_chunks = 0;

    char *chunk_start = file_buf;
    char *p = file_buf;

    while (*p && n_chunks < INGEST_MAX_CHUNKS) {
        /* Find next double-newline or end of file */
        char *break_pos = NULL;
        char *scan = p;
        while (*scan) {
            if (scan[0] == '\n' && scan[1] == '\n') {
                break_pos = scan;
                break;
            }
            scan++;
        }

        if (!break_pos) {
            /* End of file — flush remaining text */
            int remaining = file_len - (int)(p - file_buf);
            if (remaining >= INGEST_MIN_CHUNK_CHARS) {
                chunks[n_chunks].text = p;
                chunks[n_chunks].len = remaining;
                n_chunks++;
            }
            break;
        }

        /* We found a paragraph break */
        int chunk_len = (int)(break_pos - chunk_start);

        if (chunk_len >= INGEST_MIN_CHUNK_CHARS) {
            /* Check if chunk is too long — split at max */
            if (chunk_len > INGEST_MAX_CHUNK_CHARS) {
                /* Split at last space before the limit */
                int split_at = INGEST_MAX_CHUNK_CHARS;
                while (split_at > INGEST_MIN_CHUNK_CHARS &&
                       chunk_start[split_at] != ' ')
                    split_at--;
                if (split_at <= INGEST_MIN_CHUNK_CHARS)
                    split_at = INGEST_MAX_CHUNK_CHARS;

                chunks[n_chunks].text = chunk_start;
                chunks[n_chunks].len = split_at;
                n_chunks++;

                /* Remainder becomes next chunk's start */
                chunk_start = chunk_start + split_at;
                while (*chunk_start == ' ' || *chunk_start == '\n')
                    chunk_start++;
                p = break_pos + 2;
                continue;
            }

            chunks[n_chunks].text = chunk_start;
            chunks[n_chunks].len = chunk_len;
            n_chunks++;
        }

        /* Skip past the double newline */
        p = break_pos + 2;
        while (*p == '\n') p++;  /* skip extra blank lines */
        chunk_start = p;
    }

    if (n_chunks == 0) {
        fprintf(stderr, "moltar_rag: no chunks found in %s (file too short?)\n",
                text_file);
        free(file_buf);
        return 1;
    }

    fprintf(stderr, "moltar_rag: split %s into %d chunks (starting at id %d)\n",
            text_file, n_chunks, start_chunk_id);

    /* Process each chunk: write text, embed, write embeddings */
    int chunk_id = start_chunk_id;
    int n_ingested = 0;

    for (int i = 0; i < n_chunks; i++) {
        char txt_path[512], emb_path[512], tmp_path[512];
        snprintf(txt_path, sizeof(txt_path), "%s/chunk_%d.txt", index_dir, chunk_id);
        snprintf(emb_path, sizeof(emb_path), "%s/chunk_%d.emb", index_dir, chunk_id);
        snprintf(tmp_path, sizeof(tmp_path), "%s/_ingest_tmp.txt", index_dir);

        /* Write chunk text */
        FILE *tf = fopen(txt_path, "w");
        if (!tf) {
            fprintf(stderr, "moltar_rag: cannot write %s\n", txt_path);
            chunk_id++;
            continue;
        }
        fwrite(chunks[i].text, 1, chunks[i].len, tf);
        fclose(tf);

        /* Write chunk text to temp file for embedding (avoid shell injection) */
        FILE *tmpf = fopen(tmp_path, "w");
        if (!tmpf) {
            fprintf(stderr, "moltar_rag: cannot write temp file\n");
            chunk_id++;
            continue;
        }
        fwrite(chunks[i].text, 1, chunks[i].len, tmpf);
        fclose(tmpf);

        /* Call llama-embedding to produce per-token 128D embeddings */
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "export LD_LIBRARY_PATH=/data/local/tmp && "
            "taskset c0 %s "
            "-m %s -c 1024 -f %s "
            "--pooling none --embd-normalize -1 --embd-output-format raw "
            "-t 2 --no-warmup 2>/dev/null > %s",
            embedding_bin, colbert_model, tmp_path, emb_path);

        int ret = system(cmd);
        if (ret != 0) {
            fprintf(stderr, "moltar_rag: embedding failed for chunk %d (ret=%d)\n",
                    chunk_id, ret);
            chunk_id++;
            continue;
        }

        /* Count token embeddings (one per line in .emb file) */
        FILE *ef = fopen(emb_path, "r");
        int n_tok = 0;
        if (ef) {
            char line[MAX_LINE_LEN];
            while (fgets(line, sizeof(line), ef)) {
                char c = line[0];
                if (c == '-' || (c >= '0' && c <= '9')) n_tok++;
            }
            fclose(ef);
        }

        printf("%d|%d|%d\n", chunk_id, n_tok, chunks[i].len);
        fprintf(stderr, "moltar_rag: chunk %d: %d chars, %d tokens\n",
                chunk_id, chunks[i].len, n_tok);

        n_ingested++;
        chunk_id++;
    }

    /* Remove temp file */
    {
        char tmp_path[512];
        snprintf(tmp_path, sizeof(tmp_path), "%s/_ingest_tmp.txt", index_dir);
        remove(tmp_path);
    }

    /* Update manifest with new total count */
    {
        char manifest_path[512];
        snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", index_dir);
        FILE *mf = fopen(manifest_path, "w");
        if (mf) {
            fprintf(mf, "%d\n", chunk_id);
            fclose(mf);
        }
    }

    fprintf(stderr, "moltar_rag: ingested %d chunks, index now has %d total\n",
            n_ingested, chunk_id);

    free(file_buf);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage:\n"
            "  moltar_rag search  <query.emb> <index_dir> <n_chunks> <top_k>\n"
            "  moltar_rag ingest  <text_file> <index_dir> <embedding_bin> <colbert_model>\n");
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

    if (strcmp(argv[1], "ingest") == 0) {
        if (argc < 6) {
            fprintf(stderr, "Usage: moltar_rag ingest <text_file> <index_dir> "
                    "<embedding_bin> <colbert_model>\n");
            return 1;
        }
        return cmd_ingest(argv[2], argv[3], argv[4], argv[5]);
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
}
