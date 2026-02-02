/*
 * q4_sparsity_probe — Analyze weight distribution in Q4_0 GGUF files
 *
 * Reads raw Q4_0 blocks and histograms the 4-bit quantized values (0-15).
 * Q4_0 format: blocks of 32 weights, each block = 2 bytes scale (f16) + 16 bytes data.
 * Each byte holds 2 weights as nibbles (low nibble first).
 * Values 0-15 map to signed: value - 8, so range is [-8, +7].
 * Value 8 = zero weight.
 *
 * Usage: q4_sparsity_probe model.gguf
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

#define GGUF_MAGIC 0x46554747

/* Q4_0 block: 32 weights per block */
#define Q4_0_BLOCK_SIZE 18  /* 2 bytes f16 scale + 16 bytes data */
#define Q4_0_WEIGHTS_PER_BLOCK 32

/* GGUF tensor types */
#define GGUF_TYPE_Q4_0  2
#define GGUF_TYPE_Q6_K  14
#define GGUF_TYPE_F32   0

/* Read GGUF string: uint64 length + chars (no null terminator in file) */
static const uint8_t *read_gguf_string(const uint8_t *p, char *out, int maxlen) {
    uint64_t len = *(const uint64_t *)p;
    p += 8;
    int copy = (len < (uint64_t)maxlen - 1) ? (int)len : maxlen - 1;
    memcpy(out, p, copy);
    out[copy] = '\0';
    return p + len;
}

/* Skip a GGUF value (recursive for arrays) */
static const uint8_t *skip_gguf_value(const uint8_t *p, uint32_t type) {
    switch (type) {
        case 0: return p + 1;   /* uint8 */
        case 1: return p + 1;   /* int8 */
        case 2: return p + 2;   /* uint16 */
        case 3: return p + 2;   /* int16 */
        case 4: return p + 4;   /* uint32 */
        case 5: return p + 4;   /* int32 */
        case 6: return p + 4;   /* float32 */
        case 7: return p + 1;   /* bool */
        case 8: {               /* string */
            uint64_t len = *(const uint64_t *)p;
            return p + 8 + len;
        }
        case 9: {               /* array */
            uint32_t elem_type = *(const uint32_t *)p; p += 4;
            uint64_t count = *(const uint64_t *)p; p += 8;
            for (uint64_t i = 0; i < count; i++)
                p = skip_gguf_value(p, elem_type);
            return p;
        }
        case 10: return p + 8;  /* uint64 */
        case 11: return p + 8;  /* int64 */
        case 12: return p + 8;  /* float64 */
        default: return p;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 1;
    }

    /* mmap the file */
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }

    /* Parse GGUF header */
    const uint8_t *p = data;
    uint32_t magic = *(const uint32_t *)p; p += 4;
    if (magic != GGUF_MAGIC) { fprintf(stderr, "Not a GGUF file\n"); return 1; }
    uint32_t version = *(const uint32_t *)p; p += 4;
    uint64_t n_tensors = *(const uint64_t *)p; p += 8;
    uint64_t n_kv = *(const uint64_t *)p; p += 8;

    printf("GGUF v%u: %llu tensors, %llu KV pairs\n", version, n_tensors, n_kv);

    /* Skip KV pairs */
    for (uint64_t i = 0; i < n_kv; i++) {
        char key[256];
        p = read_gguf_string(p, key, sizeof(key));
        uint32_t vtype = *(const uint32_t *)p; p += 4;
        p = skip_gguf_value(p, vtype);
    }

    /* Read tensor infos */
    typedef struct {
        char name[256];
        uint32_t n_dims;
        uint64_t dims[4];
        uint32_t type;
        uint64_t offset;
    } TensorInfo;

    TensorInfo *tensors = calloc(n_tensors, sizeof(TensorInfo));

    for (uint64_t i = 0; i < n_tensors; i++) {
        p = read_gguf_string(p, tensors[i].name, sizeof(tensors[i].name));
        tensors[i].n_dims = *(const uint32_t *)p; p += 4;
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            tensors[i].dims[d] = *(const uint64_t *)p; p += 8;
        }
        tensors[i].type = *(const uint32_t *)p; p += 4;
        tensors[i].offset = *(const uint64_t *)p; p += 8;
    }

    /* Find data start (aligned) */
    uint64_t header_size = p - data;
    uint64_t alignment = 32;
    uint64_t data_offset = (header_size + alignment - 1) & ~(alignment - 1);

    /* Global histogram of 4-bit values (0-15) */
    uint64_t global_hist[16] = {0};
    uint64_t total_q4_weights = 0;
    uint64_t total_q4_tensors = 0;

    /* Per-tensor analysis */
    printf("\n%-45s %10s %8s %8s %8s %8s\n",
           "Tensor", "Weights", "Zeros%", "|val|<=1", "|val|<=2", "Ternary%");
    printf("%-45s %10s %8s %8s %8s %8s\n",
           "─────────────────────────────────────────────", "──────────", "────────",
           "────────", "────────", "────────");

    for (uint64_t t = 0; t < n_tensors; t++) {
        if (tensors[t].type != GGUF_TYPE_Q4_0) continue;
        total_q4_tensors++;

        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < tensors[t].n_dims; d++)
            n_elements *= tensors[t].dims[d];

        uint64_t n_blocks = n_elements / Q4_0_WEIGHTS_PER_BLOCK;
        const uint8_t *block_ptr = data + data_offset + tensors[t].offset;

        uint64_t hist[16] = {0};

        for (uint64_t b = 0; b < n_blocks; b++) {
            /* Skip 2-byte f16 scale */
            const uint8_t *nibbles = block_ptr + 2;

            for (int j = 0; j < 16; j++) {
                uint8_t byte = nibbles[j];
                uint8_t lo = byte & 0x0F;       /* low nibble: first weight */
                uint8_t hi = (byte >> 4) & 0x0F; /* high nibble: second weight */
                hist[lo]++;
                hist[hi]++;
            }

            block_ptr += Q4_0_BLOCK_SIZE;
        }

        /* Analyze: value 8 = zero, 7/9 = +/-1, 6/10 = +/-2 */
        uint64_t zeros = hist[8];
        uint64_t abs_le1 = hist[7] + hist[8] + hist[9];
        uint64_t abs_le2 = hist[6] + hist[7] + hist[8] + hist[9] + hist[10];
        /* "Ternary-compatible" = could be mapped to {-1, 0, +1} without too much loss */
        /* That's everything with |signed_val| <= 1, i.e., quant values 7, 8, 9 */
        uint64_t ternary = abs_le1;

        printf("%-45s %10llu %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n",
               tensors[t].name, n_elements,
               100.0 * zeros / n_elements,
               100.0 * abs_le1 / n_elements,
               100.0 * abs_le2 / n_elements,
               100.0 * ternary / n_elements);

        for (int i = 0; i < 16; i++) {
            global_hist[i] += hist[i];
        }
        total_q4_weights += n_elements;
    }

    /* Global summary */
    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("GLOBAL Q4_0 WEIGHT DISTRIBUTION (%llu tensors, %llu weights)\n",
           total_q4_tensors, total_q4_weights);
    printf("══════════════════════════════════════════════════════════════════\n\n");

    printf("  4-bit value → signed → count → percentage\n");
    printf("  ─────────────────────────────────────────────\n");
    for (int i = 0; i < 16; i++) {
        int signed_val = i - 8;
        float pct = 100.0 * global_hist[i] / total_q4_weights;
        printf("  %2d → %+3d    %12llu    %6.2f%%  ", i, signed_val, global_hist[i], pct);
        /* Histogram bar */
        int bar = (int)(pct * 2);
        for (int b = 0; b < bar; b++) printf("█");
        printf("\n");
    }

    /* Key metrics */
    uint64_t exact_zeros = global_hist[8];
    uint64_t abs_le1 = global_hist[7] + global_hist[8] + global_hist[9];
    uint64_t abs_le2 = global_hist[6] + global_hist[7] + global_hist[8] + global_hist[9] + global_hist[10];
    uint64_t abs_le3 = 0;
    for (int i = 5; i <= 11; i++) abs_le3 += global_hist[i];

    printf("\n  KEY METRICS:\n");
    printf("  ─────────────────────────────────────────────\n");
    printf("  Exact zeros (val=8):         %6.2f%%  (%llu / %llu)\n",
           100.0 * exact_zeros / total_q4_weights, exact_zeros, total_q4_weights);
    printf("  |signed| <= 1 (ternary):     %6.2f%%  (%llu / %llu)\n",
           100.0 * abs_le1 / total_q4_weights, abs_le1, total_q4_weights);
    printf("  |signed| <= 2 (5-level):     %6.2f%%  (%llu / %llu)\n",
           100.0 * abs_le2 / total_q4_weights, abs_le2, total_q4_weights);
    printf("  |signed| <= 3 (7-level):     %6.2f%%  (%llu / %llu)\n",
           100.0 * abs_le3 / total_q4_weights, abs_le3, total_q4_weights);
    printf("  Weights that could be SKIPPED (zero): %.1f%% → %.1fx fewer MACs\n",
           100.0 * exact_zeros / total_q4_weights,
           (double)total_q4_weights / (total_q4_weights - exact_zeros));

    free(tensors);
    munmap((void *)data, st.st_size);
    close(fd);
    return 0;
}
