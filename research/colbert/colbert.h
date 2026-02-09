/*
 * colbert.h — On-device ColBERT late-interaction retrieval engine
 *
 * Designed for LFM2-ColBERT-350M on Moto G Power 5G (Dimensity 930).
 * Stores per-token 128D int8 embeddings with MaxSim scoring.
 *
 * Storage layout:
 *   - Token embeddings: flat array of int8[128] vectors
 *   - Document index: maps doc_id -> (token_offset, token_count, text_offset, text_len)
 *   - Text store: flat char array of document text chunks
 *
 * MaxSim scoring:
 *   score(q, d) = Σ_i max_j dot(q_i, d_j)
 *   where q has n_q token vectors and d has n_d token vectors.
 */

#ifndef COLBERT_H
#define COLBERT_H

#include <stdint.h>

/* ── Configuration ── */

#define COLBERT_EMB_DIM     128     /* embedding dimension (after dense_2 projection) */
#define COLBERT_MAX_DOCS    4096    /* max documents in the index */
#define COLBERT_MAX_TOKENS  (COLBERT_MAX_DOCS * 256)  /* ~1M token embeddings */
#define COLBERT_MAX_TEXT    (4 * 1024 * 1024)          /* 4 MB text store */
#define COLBERT_MAX_QTOKENS 32      /* max query tokens (ColBERT spec) */
#define COLBERT_MAX_DTOKENS 512     /* max doc tokens (ColBERT spec) */
#define COLBERT_TOP_K       10      /* default top-k results */

/* ── Data Structures ── */

/*
 * Document index entry — 16 bytes
 * Maps a document ID to its token embeddings and text.
 */
typedef struct {
    uint32_t token_offset;  /* index into token embedding array */
    uint16_t token_count;   /* number of token embeddings for this doc */
    uint16_t text_len;      /* length of text chunk in bytes */
    uint32_t text_offset;   /* offset into text store */
    uint32_t reserved;      /* padding / future use */
} colbert_doc_t;

/*
 * Search result — 8 bytes
 */
typedef struct {
    uint16_t doc_id;        /* document index */
    uint16_t reserved;
    int32_t  score;         /* MaxSim score (sum of max dot products) */
} colbert_result_t;

/*
 * ColBERT index — main database struct
 */
typedef struct {
    /* Document storage */
    uint32_t      n_docs;           /* number of indexed documents */
    uint32_t      n_tokens;         /* total token embeddings stored */
    uint32_t      text_used;        /* bytes used in text store */
    uint32_t      reserved;

    /* Flat arrays (caller-provided buffers) */
    int8_t       *token_embs;       /* [n_tokens][COLBERT_EMB_DIM] — token embeddings */
    colbert_doc_t *docs;            /* [n_docs] — document index */
    char         *text_store;       /* [text_used] — document text chunks */
} colbert_t;

/* ── API ── */

/*
 * Initialize the ColBERT index with caller-provided buffers.
 *
 * token_buf:  buffer for token embeddings, size >= max_tokens * COLBERT_EMB_DIM
 * doc_buf:    buffer for document index, size >= max_docs * sizeof(colbert_doc_t)
 * text_buf:   buffer for text storage, size >= max_text
 */
void colbert_init(colbert_t *db,
                  void *token_buf, uint32_t max_tokens,
                  void *doc_buf,   uint32_t max_docs,
                  void *text_buf,  uint32_t max_text);

/*
 * Add a document to the index.
 *
 * embeddings:  per-token embeddings, float32 [n_tokens][COLBERT_EMB_DIM]
 *              Will be quantized to int8 internally.
 * n_tokens:    number of token embeddings (1..COLBERT_MAX_DTOKENS)
 * text:        document text chunk (stored verbatim for retrieval)
 * text_len:    length of text in bytes
 *
 * Returns: document ID (0-based), or -1 on error (capacity exceeded).
 */
int colbert_add_doc(colbert_t *db,
                    const float *embeddings, uint16_t n_tokens,
                    const char *text, uint16_t text_len);

/*
 * Add a document with pre-quantized int8 embeddings.
 *
 * embeddings:  per-token embeddings, int8 [n_tokens][COLBERT_EMB_DIM]
 * n_tokens:    number of token embeddings
 * text:        document text chunk
 * text_len:    length of text in bytes
 *
 * Returns: document ID (0-based), or -1 on error.
 */
int colbert_add_doc_i8(colbert_t *db,
                       const int8_t *embeddings, uint16_t n_tokens,
                       const char *text, uint16_t text_len);

/*
 * Search the index using MaxSim scoring.
 *
 * query_embs:  query token embeddings, float32 [n_qtokens][COLBERT_EMB_DIM]
 *              Will be quantized to int8 internally.
 * n_qtokens:   number of query token embeddings (1..COLBERT_MAX_QTOKENS)
 * results:     output array of top-k results (sorted by score, descending)
 * k:           number of results to return
 *
 * Returns: number of results written (may be < k if fewer docs exist).
 */
int colbert_search(const colbert_t *db,
                   const float *query_embs, uint16_t n_qtokens,
                   colbert_result_t *results, int k);

/*
 * Search with pre-quantized int8 query embeddings.
 */
int colbert_search_i8(const colbert_t *db,
                      const int8_t *query_embs, uint16_t n_qtokens,
                      colbert_result_t *results, int k);

/*
 * Get the text of a document by ID.
 *
 * Returns: pointer to text (NOT null-terminated), length in *out_len.
 *          Returns NULL if doc_id is out of range.
 */
const char *colbert_get_text(const colbert_t *db, uint16_t doc_id, uint16_t *out_len);

/*
 * Quantize a float32 embedding vector to int8.
 * Uses per-vector absmax quantization: scale = 127 / max(|x|)
 *
 * in:   float32 vector [dim]
 * out:  int8 vector [dim]
 * dim:  vector dimension
 */
void colbert_quantize_f32_to_i8(const float *in, int8_t *out, int dim);

/* ── NEON-optimized primitives (defined in maxsim_neon.S) ── */

/*
 * Compute dot product of two int8 128D vectors.
 * Returns int32 result. Overflow-safe via per-chunk widening.
 */
int32_t colbert_dot_i8(const int8_t *a, const int8_t *b);

/*
 * Compute MaxSim score for one query against one document.
 *
 * query:      [n_q][128] int8 query token embeddings
 * doc:        [n_d][128] int8 document token embeddings
 * n_q:        number of query tokens
 * n_d:        number of document tokens
 *
 * Returns: MaxSim score = Σ_i max_j dot(q_i, d_j)
 */
int32_t colbert_maxsim_i8(const int8_t *query, const int8_t *doc,
                          int n_q, int n_d);

#endif /* COLBERT_H */
