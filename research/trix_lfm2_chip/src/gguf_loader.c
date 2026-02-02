/*
 * GGUF Loader for TriX LFM2 Chip
 *
 * Standalone GGUF v3 parser. No dependencies on llama.cpp.
 * Maps Q4_0 tensor data directly into trix_lfm2_weights_t by pointer
 * (zero-copy: weights point into the mmap'd file).
 *
 * GGUF v3 file layout:
 *   [magic: 4 bytes "GGUF"]
 *   [version: uint32]
 *   [n_tensors: uint64]
 *   [n_kv: uint64]
 *   [KV pairs...]
 *   [tensor infos...]
 *   [alignment padding]
 *   [tensor data...]
 *
 * Created by: Tripp + Claude
 * Date: February 1, 2026
 */

#include "../include/lfm2_trix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * GGUF v3 Constants and Types
 * ═══════════════════════════════════════════════════════════════════════════ */

#define GGUF_MAGIC_VALUE 0x46554747  /* "GGUF" in little-endian */
#define GGUF_VERSION_3   3
#define GGUF_DEFAULT_ALIGNMENT 32

/* GGUF value types */
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* GGML tensor types */
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
};

/* Size of each GGUF value type in bytes */
static size_t gguf_type_size(enum gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_STRING:  return 0; /* variable */
        case GGUF_TYPE_ARRAY:   return 0; /* variable */
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GGUF Reader — Cursor-based parser
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const uint8_t *data;
    size_t         size;
    size_t         pos;
} gguf_cursor_t;

static bool cursor_check(const gguf_cursor_t *c, size_t need) {
    return c->pos + need <= c->size;
}

static uint8_t read_u8(gguf_cursor_t *c) {
    uint8_t v = c->data[c->pos];
    c->pos += 1;
    return v;
}

static uint32_t read_u32(gguf_cursor_t *c) {
    uint32_t v;
    memcpy(&v, c->data + c->pos, 4);
    c->pos += 4;
    return v;
}

static uint64_t read_u64(gguf_cursor_t *c) {
    uint64_t v;
    memcpy(&v, c->data + c->pos, 8);
    c->pos += 8;
    return v;
}

static float read_f32(gguf_cursor_t *c) {
    float v;
    memcpy(&v, c->data + c->pos, 4);
    c->pos += 4;
    return v;
}

/* Read a GGUF string: uint64 length + chars (NOT null-terminated in file) */
typedef struct {
    const char *str;
    uint64_t    len;
} gguf_string_t;

static gguf_string_t read_string(gguf_cursor_t *c) {
    gguf_string_t s;
    s.len = read_u64(c);
    s.str = (const char *)(c->data + c->pos);
    c->pos += s.len;
    return s;
}

/* Skip over a KV value (we don't need most metadata) */
static void skip_value(gguf_cursor_t *c, enum gguf_type type) {
    if (type == GGUF_TYPE_STRING) {
        gguf_string_t s = read_string(c);
        (void)s;
    } else if (type == GGUF_TYPE_ARRAY) {
        enum gguf_type arr_type = (enum gguf_type)read_u32(c);
        uint64_t arr_len = read_u64(c);
        if (arr_type == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < arr_len; i++) {
                gguf_string_t s = read_string(c);
                (void)s;
            }
        } else {
            size_t elem_size = gguf_type_size(arr_type);
            c->pos += arr_len * elem_size;
        }
    } else {
        c->pos += gguf_type_size(type);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tensor Info from GGUF header
 * ═══════════════════════════════════════════════════════════════════════════ */

#define MAX_GGUF_TENSORS 256
#define MAX_TENSOR_NAME  128
#define MAX_DIMS         4

typedef struct {
    char     name[MAX_TENSOR_NAME];
    uint32_t n_dims;
    uint64_t ne[MAX_DIMS];        /* shape */
    uint32_t type;                /* ggml_type */
    uint64_t offset;              /* offset within tensor data section */
} gguf_tensor_info_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * GGUF File Handle (mmap'd)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* mmap state */
    void    *mmap_addr;
    size_t   mmap_size;
    int      fd;

    /* Parsed header */
    uint64_t n_tensors;
    uint64_t n_kv;
    uint64_t data_offset;  /* offset to start of tensor data */

    /* Tensor directory */
    gguf_tensor_info_t tensors[MAX_GGUF_TENSORS];
} gguf_file_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * Open and parse a GGUF file
 * ═══════════════════════════════════════════════════════════════════════════ */

static gguf_file_t *gguf_open(const char *path) {
    gguf_file_t *gf = (gguf_file_t *)calloc(1, sizeof(gguf_file_t));
    if (!gf) return NULL;

#ifdef _WIN32
    /* TODO: Windows mmap via CreateFileMapping */
    fprintf(stderr, "GGUF: Windows not yet supported\n");
    free(gf);
    return NULL;
#else
    gf->fd = open(path, O_RDONLY);
    if (gf->fd < 0) {
        fprintf(stderr, "GGUF: failed to open '%s'\n", path);
        free(gf);
        return NULL;
    }

    struct stat st;
    if (fstat(gf->fd, &st) != 0) {
        close(gf->fd);
        free(gf);
        return NULL;
    }
    gf->mmap_size = (size_t)st.st_size;

    gf->mmap_addr = mmap(NULL, gf->mmap_size, PROT_READ, MAP_PRIVATE, gf->fd, 0);
    if (gf->mmap_addr == MAP_FAILED) {
        fprintf(stderr, "GGUF: mmap failed for '%s' (%zu bytes)\n", path, gf->mmap_size);
        close(gf->fd);
        free(gf);
        return NULL;
    }
#endif

    /* Parse header */
    gguf_cursor_t c = { .data = (const uint8_t *)gf->mmap_addr, .size = gf->mmap_size, .pos = 0 };

    /* Magic */
    uint32_t magic = read_u32(&c);
    if (magic != GGUF_MAGIC_VALUE) {
        fprintf(stderr, "GGUF: bad magic 0x%08x (expected 0x%08x)\n", magic, GGUF_MAGIC_VALUE);
        goto fail;
    }

    /* Version */
    uint32_t version = read_u32(&c);
    if (version != GGUF_VERSION_3) {
        fprintf(stderr, "GGUF: unsupported version %u (expected %u)\n", version, GGUF_VERSION_3);
        goto fail;
    }

    /* Counts */
    gf->n_tensors = read_u64(&c);
    gf->n_kv      = read_u64(&c);

    if (gf->n_tensors > MAX_GGUF_TENSORS) {
        fprintf(stderr, "GGUF: too many tensors (%llu > %d)\n",
                (unsigned long long)gf->n_tensors, MAX_GGUF_TENSORS);
        goto fail;
    }

    fprintf(stderr, "GGUF: version=%u, n_kv=%llu, n_tensors=%llu\n",
            version, (unsigned long long)gf->n_kv, (unsigned long long)gf->n_tensors);

    /* Skip KV pairs — we don't need metadata, only tensor data */
    for (uint64_t i = 0; i < gf->n_kv; i++) {
        if (!cursor_check(&c, 8)) goto fail;
        gguf_string_t key = read_string(&c);
        (void)key;
        if (!cursor_check(&c, 4)) goto fail;
        enum gguf_type vtype = (enum gguf_type)read_u32(&c);
        skip_value(&c, vtype);
    }

    /* Read tensor infos */
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        gguf_tensor_info_t *ti = &gf->tensors[i];

        gguf_string_t name = read_string(&c);
        size_t name_len = name.len < MAX_TENSOR_NAME - 1 ? name.len : MAX_TENSOR_NAME - 1;
        memcpy(ti->name, name.str, name_len);
        ti->name[name_len] = '\0';

        ti->n_dims = read_u32(&c);
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            ti->ne[d] = read_u64(&c);
        }
        for (uint32_t d = ti->n_dims; d < MAX_DIMS; d++) {
            ti->ne[d] = 1;
        }

        ti->type   = read_u32(&c);
        ti->offset = read_u64(&c);
    }

    /* Data section starts after alignment padding */
    size_t header_end = c.pos;
    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    gf->data_offset = (header_end + alignment - 1) & ~(alignment - 1);

    fprintf(stderr, "GGUF: header_end=%zu, data_offset=%llu\n",
            header_end, (unsigned long long)gf->data_offset);

    return gf;

fail:
    munmap(gf->mmap_addr, gf->mmap_size);
    close(gf->fd);
    free(gf);
    return NULL;
}

static void gguf_close(gguf_file_t *gf) {
    if (!gf) return;
#ifndef _WIN32
    munmap(gf->mmap_addr, gf->mmap_size);
    close(gf->fd);
#endif
    free(gf);
}

/* Get a pointer to tensor data given a tensor info */
static const void *gguf_tensor_data(const gguf_file_t *gf, const gguf_tensor_info_t *ti) {
    return (const uint8_t *)gf->mmap_addr + gf->data_offset + ti->offset;
}

/* Find a tensor by name, returns NULL if not found */
static const gguf_tensor_info_t *gguf_find_tensor(const gguf_file_t *gf, const char *name) {
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        if (strcmp(gf->tensors[i].name, name) == 0) {
            return &gf->tensors[i];
        }
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight Mapping: GGUF tensor names → trix_lfm2_weights_t
 *
 * GGUF tensor names for LFM2-350M (from llama-arch.cpp):
 *
 *   Global:
 *     "token_embd.weight"              → tok_embd
 *     "output.weight"                  → output
 *     "token_embd_norm.weight"         → output_norm (LFM2 naming quirk)
 *
 *   Per-layer (blk.%d.):
 *     "blk.%d.attn_norm.weight"        → operator norm (shared: attn or shortconv)
 *     "blk.%d.ffn_norm.weight"         → FFN norm
 *     "blk.%d.ffn_gate.weight"         → FFN gate
 *     "blk.%d.ffn_up.weight"           → FFN up
 *     "blk.%d.ffn_down.weight"         → FFN down
 *
 *   Attention layers only:
 *     "blk.%d.attn_q.weight"           → wq
 *     "blk.%d.attn_k.weight"           → wk
 *     "blk.%d.attn_v.weight"           → wv
 *     "blk.%d.attn_output.weight"      → wo
 *     "blk.%d.attn_q_norm.weight"      → q_norm
 *     "blk.%d.attn_k_norm.weight"      → k_norm
 *
 *   ShortConv layers only:
 *     "blk.%d.shortconv.in_proj.weight"   → in_proj
 *     "blk.%d.shortconv.conv.weight"      → conv_kernel
 *     "blk.%d.shortconv.out_proj.weight"  → out_proj
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Helper: find tensor or die */
static const gguf_tensor_info_t *require_tensor(const gguf_file_t *gf, const char *name) {
    const gguf_tensor_info_t *ti = gguf_find_tensor(gf, name);
    if (!ti) {
        fprintf(stderr, "GGUF LOADER: FATAL — required tensor '%s' not found!\n", name);
        fprintf(stderr, "  Available tensors:\n");
        for (uint64_t i = 0; i < gf->n_tensors; i++) {
            fprintf(stderr, "    [%llu] %s\n", (unsigned long long)i, gf->tensors[i].name);
        }
        exit(1);
    }
    return ti;
}

/* Helper: get Q4_0 block pointer for a tensor */
static const q4_0_block_t *tensor_q4_0(const gguf_file_t *gf, const char *name) {
    const gguf_tensor_info_t *ti = require_tensor(gf, name);
    if (ti->type != GGML_TYPE_Q4_0) {
        fprintf(stderr, "GGUF LOADER: tensor '%s' has type %u, expected Q4_0 (%u)\n",
                name, ti->type, GGML_TYPE_Q4_0);
        exit(1);
    }
    return (const q4_0_block_t *)gguf_tensor_data(gf, ti);
}

/* Helper: get F32 pointer for a tensor */
static const float *tensor_f32(const gguf_file_t *gf, const char *name) {
    const gguf_tensor_info_t *ti = require_tensor(gf, name);
    if (ti->type != GGML_TYPE_F32) {
        fprintf(stderr, "GGUF LOADER: tensor '%s' has type %u, expected F32 (%u)\n",
                name, ti->type, GGML_TYPE_F32);
        exit(1);
    }
    return (const float *)gguf_tensor_data(gf, ti);
}

/* Helper: get pointer for tensor that could be Q4_0 or F32 (conv kernel may be either) */
static const void *tensor_any(const gguf_file_t *gf, const char *name) {
    const gguf_tensor_info_t *ti = require_tensor(gf, name);
    return gguf_tensor_data(gf, ti);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API: Load GGUF into trix_lfm2_weights_t
 * ═══════════════════════════════════════════════════════════════════════════ */

struct trix_gguf_model {
    gguf_file_t         *gf;      /* keep alive for mmap lifetime */
    trix_lfm2_weights_t  weights; /* pointers into mmap'd data */
};

const trix_lfm2_weights_t *trix_get_weights(const trix_gguf_model_t *model) {
    return &model->weights;
}

trix_gguf_model_t *trix_load_gguf(const char *path) {
    trix_gguf_model_t *model = (trix_gguf_model_t *)calloc(1, sizeof(trix_gguf_model_t));
    if (!model) return NULL;

    model->gf = gguf_open(path);
    if (!model->gf) {
        free(model);
        return NULL;
    }

    const gguf_file_t *gf = model->gf;
    trix_lfm2_weights_t *w = &model->weights;

    /* ─── Global tensors ─── */

    /* token_embd: may be Q4_0 or Q6_K */
    const gguf_tensor_info_t *embd_ti = require_tensor(gf, "token_embd.weight");
    w->tok_embd = gguf_tensor_data(gf, embd_ti);
    if (embd_ti->type == GGML_TYPE_Q4_0) {
        w->tok_embd_type = TRIX_QTYPE_Q4_0;
        fprintf(stderr, "GGUF LOADER: token_embd is Q4_0\n");
    } else if (embd_ti->type == GGML_TYPE_Q6_K) {
        w->tok_embd_type = TRIX_QTYPE_Q6_K;
        fprintf(stderr, "GGUF LOADER: token_embd is Q6_K\n");
    } else {
        fprintf(stderr, "GGUF LOADER: WARNING — token_embd has unsupported type %u, treating as raw\n",
                embd_ti->type);
        w->tok_embd_type = TRIX_QTYPE_Q6_K; /* best guess */
    }

    /* output: may be absent (weight tying with token_embd) */
    const gguf_tensor_info_t *output_ti = gguf_find_tensor(gf, "output.weight");
    if (output_ti) {
        w->output = gguf_tensor_data(gf, output_ti);
        w->output_tied = 0;
        if (output_ti->type == GGML_TYPE_Q4_0) {
            w->output_type = TRIX_QTYPE_Q4_0;
        } else if (output_ti->type == GGML_TYPE_Q6_K) {
            w->output_type = TRIX_QTYPE_Q6_K;
        } else {
            w->output_type = TRIX_QTYPE_Q6_K;
        }
        fprintf(stderr, "GGUF LOADER: output head is separate (type %u)\n", output_ti->type);
    } else {
        /* Weight tying: output shares token_embd */
        w->output = w->tok_embd;
        w->output_type = w->tok_embd_type;
        w->output_tied = 1;
        fprintf(stderr, "GGUF LOADER: output head is weight-tied to token_embd\n");
    }

    w->output_norm = tensor_f32(gf, "token_embd_norm.weight");

    fprintf(stderr, "GGUF LOADER: loaded global tensors\n");

    /* ─── Per-layer tensors ─── */
    int conv_idx = 0;
    int attn_idx = 0;
    char name_buf[128];

    for (int il = 0; il < LFM2_N_LAYERS; il++) {
        /* FFN (all 16 layers) */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", il);
        w->ffn[il].norm_weight = tensor_f32(gf, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate.weight", il);
        w->ffn[il].gate = tensor_q4_0(gf, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up.weight", il);
        w->ffn[il].up = tensor_q4_0(gf, name_buf);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down.weight", il);
        w->ffn[il].down = tensor_q4_0(gf, name_buf);

        if (LFM2_IS_ATTN(il)) {
            /* Attention layer */
            snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", il);
            w->attention[attn_idx].norm_weight = tensor_f32(gf, name_buf);

            snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q.weight", il);
            w->attention[attn_idx].wq = tensor_q4_0(gf, name_buf);

            snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k.weight", il);
            w->attention[attn_idx].wk = tensor_q4_0(gf, name_buf);

            snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_v.weight", il);
            w->attention[attn_idx].wv = tensor_q4_0(gf, name_buf);

            snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_output.weight", il);
            w->attention[attn_idx].wo = tensor_q4_0(gf, name_buf);

            snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q_norm.weight", il);
            w->attention[attn_idx].q_norm = tensor_f32(gf, name_buf);

            snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k_norm.weight", il);
            w->attention[attn_idx].k_norm = tensor_f32(gf, name_buf);

            attn_idx++;
            fprintf(stderr, "GGUF LOADER: layer %d — attention [%d]\n", il, attn_idx - 1);
        } else {
            /* ShortConv layer */
            snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", il);
            w->shortconv[conv_idx].norm_weight = tensor_f32(gf, name_buf);

            snprintf(name_buf, sizeof(name_buf), "blk.%d.shortconv.in_proj.weight", il);
            w->shortconv[conv_idx].in_proj = tensor_q4_0(gf, name_buf);
            {
                const gguf_tensor_info_t *ti = gguf_find_tensor(gf, name_buf);
                if (ti && il == 0) {
                    fprintf(stderr, "GGUF LOADER: %s: offset=%llu, data_offset=%llu, abs=%llu, ne=[%llu,%llu], type=%u\n",
                            name_buf, (unsigned long long)ti->offset,
                            (unsigned long long)gf->data_offset,
                            (unsigned long long)(gf->data_offset + ti->offset),
                            (unsigned long long)ti->ne[0], (unsigned long long)ti->ne[1],
                            ti->type);
                    /* Print first few bytes of the tensor data */
                    const uint8_t *raw = (const uint8_t *)gguf_tensor_data(gf, ti);
                    fprintf(stderr, "GGUF LOADER: first 20 bytes: ");
                    for (int zz = 0; zz < 20; zz++) fprintf(stderr, "%02x ", raw[zz]);
                    fprintf(stderr, "\n");
                }
            }

            snprintf(name_buf, sizeof(name_buf), "blk.%d.shortconv.conv.weight", il);
            /* conv kernel is always F32 (small tensor: [D_CONV, D_MODEL]) */
            w->shortconv[conv_idx].conv_kernel = tensor_f32(gf, name_buf);

            snprintf(name_buf, sizeof(name_buf), "blk.%d.shortconv.out_proj.weight", il);
            w->shortconv[conv_idx].out_proj = tensor_q4_0(gf, name_buf);

            conv_idx++;
            fprintf(stderr, "GGUF LOADER: layer %d — shortconv [%d]\n", il, conv_idx - 1);
        }
    }

    fprintf(stderr, "GGUF LOADER: loaded %d attention layers, %d shortconv layers\n",
            attn_idx, conv_idx);
    fprintf(stderr, "GGUF LOADER: all %llu tensors mapped successfully\n",
            (unsigned long long)gf->n_tensors);

    return model;
}

void trix_unload_gguf(trix_gguf_model_t *model) {
    if (!model) return;
    gguf_close(model->gf);
    free(model);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * FP16 -> FP32 helper
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Use trix_fp16_to_f32 from header */
#define fp16_to_f32 trix_fp16_to_f32

/* ═══════════════════════════════════════════════════════════════════════════
 * Q6_K Dequantization
 *
 * Q6_K block layout (256 values, 210 bytes):
 *   ql[128]   — lower 4 bits of each quantized value
 *   qh[64]    — upper 2 bits of each quantized value (packed)
 *   scales[16] — int8 scales for 16 groups of 16 values
 *   d (fp16)  — super-block scale
 *
 * Dequant: val = d * scale[group] * (q6_val - 32)
 *   where q6_val = (ql_bits | (qh_bits << 4)) for the 6-bit value
 * ═══════════════════════════════════════════════════════════════════════════ */

static void dequant_q6_k_row(const q6_k_block_t *blocks, int n_values, float *out) {
    const int n_blocks = n_values / Q6_K_BLOCK_SIZE;

    for (int i = 0; i < n_blocks; i++) {
        const q6_k_block_t *blk = &blocks[i];
        float d = fp16_to_f32(blk->d_f16);

        const uint8_t *ql = blk->ql;
        const uint8_t *qh = blk->qh;
        const int8_t  *sc = blk->scales;
        float *y = out + i * Q6_K_BLOCK_SIZE;

        /* Process 256 values in two 128-value chunks */
        for (int n = 0; n < Q6_K_BLOCK_SIZE; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                /* Extract 4 values per position from different parts of ql/qh:
                 *   q1: low nibble of ql[l+0],   bits 0-1 of qh[l]
                 *   q2: low nibble of ql[l+32],  bits 2-3 of qh[l]
                 *   q3: high nibble of ql[l+0],  bits 4-5 of qh[l]
                 *   q4: high nibble of ql[l+32], bits 6-7 of qh[l] */
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Embedding Lookup Helper
 *
 * Supports Q4_0 and Q6_K token embeddings.
 * tok_embd is [VOCAB_SIZE, D_MODEL].
 * ═══════════════════════════════════════════════════════════════════════════ */

void trix_embed_token(
    const trix_lfm2_weights_t *w,
    uint32_t token_id,
    float *out  /* [D_MODEL] */
) {
    if (w->tok_embd_type == TRIX_QTYPE_Q6_K) {
        /* Q6_K: 256 values per block, D_MODEL=1024 → 4 blocks per row */
        const int n_blocks_per_row = LFM2_D_MODEL / Q6_K_BLOCK_SIZE;
        const q6_k_block_t *row = (const q6_k_block_t *)w->tok_embd + token_id * n_blocks_per_row;
        dequant_q6_k_row(row, LFM2_D_MODEL, out);
    } else {
        /* Q4_0: 32 values per block, D_MODEL=1024 → 32 blocks per row */
        const int n_blocks = LFM2_D_MODEL / Q4_0_BLOCK_SIZE;
        const q4_0_block_t *row = (const q4_0_block_t *)w->tok_embd + token_id * n_blocks;

        for (int b = 0; b < n_blocks; b++) {
            float scale = trix_fp16_to_f32(row[b].scale_f16);
            float *dst = out + b * Q4_0_BLOCK_SIZE;
            for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
                uint8_t packed = row[b].qs[j];
                dst[j]      = ((float)(packed & 0x0F) - 8.0f) * scale;
                dst[j + 16] = ((float)(packed >> 4) - 8.0f) * scale;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Q6_K Matrix-Vector Multiply
 *
 * For the output head when it's Q6_K (or weight-tied to Q6_K embedding).
 * y[i] = SUM_k(dequant(W[i,k]) * x[k])
 * ═══════════════════════════════════════════════════════════════════════════ */

void trix_matvec_q6_k(
    const q6_k_block_t *W,
    const float        *x,
    float              *y,
    int M, int K
) {
    const int n_blocks_per_row = K / Q6_K_BLOCK_SIZE;

    /* Temporary buffer for one dequantized row */
    float row_buf[LFM2_D_MODEL]; /* max K we'll see is D_MODEL=1024 */

    for (int i = 0; i < M; i++) {
        const q6_k_block_t *row = W + i * n_blocks_per_row;
        dequant_q6_k_row(row, K, row_buf);

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += row_buf[k] * x[k];
        }
        y[i] = sum;
    }
}
