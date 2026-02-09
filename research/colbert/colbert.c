/*
 * colbert.c — ColBERT index implementation (C reference)
 *
 * Provides document storage, quantization, and search.
 * MaxSim dot products are provided by maxsim_neon.S on device.
 */

#include "colbert.h"
#include <string.h>
#include <math.h>

/* ── Initialization ── */

void colbert_init(colbert_t *db,
                  void *token_buf, uint32_t max_tokens,
                  void *doc_buf,   uint32_t max_docs,
                  void *text_buf,  uint32_t max_text)
{
    memset(db, 0, sizeof(*db));
    db->token_embs = (int8_t *)token_buf;
    db->docs       = (colbert_doc_t *)doc_buf;
    db->text_store = (char *)text_buf;
    db->n_docs     = 0;
    db->n_tokens   = 0;
    db->text_used  = 0;
    (void)max_tokens;
    (void)max_docs;
    (void)max_text;
}

/* ── Quantization ── */

void colbert_quantize_f32_to_i8(const float *in, int8_t *out, int dim)
{
    /* Per-vector absmax quantization */
    float absmax = 0.0f;
    for (int i = 0; i < dim; i++) {
        float a = fabsf(in[i]);
        if (a > absmax) absmax = a;
    }

    if (absmax < 1e-10f) {
        memset(out, 0, dim);
        return;
    }

    float scale = 127.0f / absmax;
    for (int i = 0; i < dim; i++) {
        float v = in[i] * scale;
        if (v > 127.0f)  v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        out[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }
}

/* ── Document Insertion ── */

int colbert_add_doc(colbert_t *db,
                    const float *embeddings, uint16_t n_tokens,
                    const char *text, uint16_t text_len)
{
    if (db->n_docs >= COLBERT_MAX_DOCS) return -1;
    if (db->n_tokens + n_tokens > COLBERT_MAX_TOKENS) return -1;
    if (db->text_used + text_len > COLBERT_MAX_TEXT) return -1;
    if (n_tokens == 0 || n_tokens > COLBERT_MAX_DTOKENS) return -1;

    uint32_t doc_id = db->n_docs;
    uint32_t tok_off = db->n_tokens;
    uint32_t txt_off = db->text_used;

    /* Quantize each token embedding */
    for (uint16_t t = 0; t < n_tokens; t++) {
        colbert_quantize_f32_to_i8(
            &embeddings[t * COLBERT_EMB_DIM],
            &db->token_embs[(tok_off + t) * COLBERT_EMB_DIM],
            COLBERT_EMB_DIM
        );
    }

    /* Copy text */
    memcpy(&db->text_store[txt_off], text, text_len);

    /* Write doc index entry */
    db->docs[doc_id].token_offset = tok_off;
    db->docs[doc_id].token_count  = n_tokens;
    db->docs[doc_id].text_offset  = txt_off;
    db->docs[doc_id].text_len     = text_len;
    db->docs[doc_id].reserved     = 0;

    db->n_docs++;
    db->n_tokens += n_tokens;
    db->text_used += text_len;

    return (int)doc_id;
}

int colbert_add_doc_i8(colbert_t *db,
                       const int8_t *embeddings, uint16_t n_tokens,
                       const char *text, uint16_t text_len)
{
    if (db->n_docs >= COLBERT_MAX_DOCS) return -1;
    if (db->n_tokens + n_tokens > COLBERT_MAX_TOKENS) return -1;
    if (db->text_used + text_len > COLBERT_MAX_TEXT) return -1;
    if (n_tokens == 0 || n_tokens > COLBERT_MAX_DTOKENS) return -1;

    uint32_t doc_id = db->n_docs;
    uint32_t tok_off = db->n_tokens;
    uint32_t txt_off = db->text_used;

    /* Copy token embeddings directly */
    memcpy(&db->token_embs[tok_off * COLBERT_EMB_DIM],
           embeddings,
           n_tokens * COLBERT_EMB_DIM);

    /* Copy text */
    memcpy(&db->text_store[txt_off], text, text_len);

    /* Write doc index entry */
    db->docs[doc_id].token_offset = tok_off;
    db->docs[doc_id].token_count  = n_tokens;
    db->docs[doc_id].text_offset  = txt_off;
    db->docs[doc_id].text_len     = text_len;
    db->docs[doc_id].reserved     = 0;

    db->n_docs++;
    db->n_tokens += n_tokens;
    db->text_used += text_len;

    return (int)doc_id;
}

/* ── Text Retrieval ── */

const char *colbert_get_text(const colbert_t *db, uint16_t doc_id, uint16_t *out_len)
{
    if (doc_id >= db->n_docs) return 0;
    *out_len = db->docs[doc_id].text_len;
    return &db->text_store[db->docs[doc_id].text_offset];
}

/* ── Search ── */

/*
 * Insert into a top-k min-heap (sorted descending by score).
 * We maintain a min-heap so we can efficiently replace the minimum element.
 */
static void topk_insert(colbert_result_t *heap, int *heap_size, int k,
                         uint16_t doc_id, int32_t score)
{
    if (*heap_size < k) {
        /* Heap not full — add to end and sift up */
        int i = *heap_size;
        heap[i].doc_id = doc_id;
        heap[i].score  = score;
        heap[i].reserved = 0;
        (*heap_size)++;
        /* Sift up (min-heap: parent <= children) */
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (heap[parent].score <= heap[i].score) break;
            colbert_result_t tmp = heap[parent];
            heap[parent] = heap[i];
            heap[i] = tmp;
            i = parent;
        }
    } else if (score > heap[0].score) {
        /* Replace minimum and sift down */
        heap[0].doc_id = doc_id;
        heap[0].score  = score;
        heap[0].reserved = 0;
        int i = 0;
        for (;;) {
            int left  = 2 * i + 1;
            int right = 2 * i + 2;
            int smallest = i;
            if (left < k && heap[left].score < heap[smallest].score)
                smallest = left;
            if (right < k && heap[right].score < heap[smallest].score)
                smallest = right;
            if (smallest == i) break;
            colbert_result_t tmp = heap[smallest];
            heap[smallest] = heap[i];
            heap[i] = tmp;
            i = smallest;
        }
    }
}

/* Sort results descending by score (simple insertion sort for small k) */
static void sort_results_desc(colbert_result_t *results, int n)
{
    for (int i = 1; i < n; i++) {
        colbert_result_t key = results[i];
        int j = i - 1;
        while (j >= 0 && results[j].score < key.score) {
            results[j + 1] = results[j];
            j--;
        }
        results[j + 1] = key;
    }
}

int colbert_search(const colbert_t *db,
                   const float *query_embs, uint16_t n_qtokens,
                   colbert_result_t *results, int k)
{
    if (n_qtokens == 0 || n_qtokens > COLBERT_MAX_QTOKENS) return 0;
    if (db->n_docs == 0) return 0;
    if (k <= 0) return 0;
    if (k > COLBERT_TOP_K) k = COLBERT_TOP_K;

    /* Quantize query */
    int8_t query_i8[COLBERT_MAX_QTOKENS * COLBERT_EMB_DIM];
    for (uint16_t t = 0; t < n_qtokens; t++) {
        colbert_quantize_f32_to_i8(
            &query_embs[t * COLBERT_EMB_DIM],
            &query_i8[t * COLBERT_EMB_DIM],
            COLBERT_EMB_DIM
        );
    }

    return colbert_search_i8(db, query_i8, n_qtokens, results, k);
}

int colbert_search_i8(const colbert_t *db,
                      const int8_t *query_embs, uint16_t n_qtokens,
                      colbert_result_t *results, int k)
{
    if (n_qtokens == 0 || n_qtokens > COLBERT_MAX_QTOKENS) return 0;
    if (db->n_docs == 0) return 0;
    if (k <= 0) return 0;
    if (k > COLBERT_TOP_K) k = COLBERT_TOP_K;

    int heap_size = 0;

    for (uint32_t d = 0; d < db->n_docs; d++) {
        const colbert_doc_t *doc = &db->docs[d];
        const int8_t *doc_embs = &db->token_embs[doc->token_offset * COLBERT_EMB_DIM];

        /* Compute MaxSim score using NEON assembly */
        int32_t score = colbert_maxsim_i8(query_embs, doc_embs,
                                           n_qtokens, doc->token_count);

        topk_insert(results, &heap_size, k, (uint16_t)d, score);
    }

    /* Sort results descending by score */
    sort_results_desc(results, heap_size);

    return heap_size;
}
