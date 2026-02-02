/*
 * q4_to_ternary.c — Convert Q4_0 GGUF weights to N-level quantization
 *
 * Copies the GGUF file byte-for-byte, but for every Q4_0 tensor, clamps
 * each 4-bit nibble to a reduced set of levels centered on zero.
 *
 * Levels (controlled by -l parameter):
 *   -l 1 = ternary {-1, 0, +1}     (nibbles 7,8,9)
 *   -l 2 = 5-level  {-2,-1,0,+1,+2} (nibbles 6..10)
 *   -l 3 = 7-level  {-3..+3}         (nibbles 5..11)
 *   -l 7 = full Q4_0 (no-op)         (nibbles 1..15, same as original)
 *
 * Block scales (fp16, 2 bytes per 32-weight block) are PRESERVED.
 * Non-Q4_0 tensors (Q6_K embeddings, F32 conv kernels, etc.) are untouched.
 *
 * Build (host macOS):
 *   cc -O2 -o q4_to_ternary q4_to_ternary.c -lm
 *
 * Usage:
 *   ./q4_to_ternary [-l levels] input.gguf output.gguf
 *   levels: 1=ternary (default), 2=5-level, 3=7-level, 7=full
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>

#define GGUF_MAGIC 0x46554747
#define GGUF_TYPE_Q4_0 2

/* Q4_0 block: 2 bytes fp16 scale + 16 bytes data (32 nibbles) = 18 bytes */
#define Q4_BLOCK_SIZE 18

typedef struct {
    char name[256];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;    /* relative to data section start */
    uint64_t n_elements;
    uint64_t n_blocks;  /* total Q4_0 blocks in this tensor */
} TensorInfo;

static const uint8_t *skip_gguf_string(const uint8_t *p) {
    uint64_t len = *(const uint64_t *)p;
    return p + 8 + len;
}

static void read_gguf_string(const uint8_t *p, char *out, int maxlen) {
    uint64_t len = *(const uint64_t *)p;
    int copy = (len < (uint64_t)maxlen - 1) ? (int)len : maxlen - 1;
    memcpy(out, p + 8, copy);
    out[copy] = '\0';
}

static const uint8_t *skip_gguf_value(const uint8_t *p, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return p + 1;
        case 2: case 3: return p + 2;
        case 4: case 5: case 6: return p + 4;
        case 10: case 11: case 12: return p + 8;
        case 8: return skip_gguf_string(p);
        case 9: {
            uint32_t et = *(const uint32_t *)p; p += 4;
            uint64_t cnt = *(const uint64_t *)p; p += 8;
            for (uint64_t i = 0; i < cnt; i++) p = skip_gguf_value(p, et);
            return p;
        }
        default: return p;
    }
}

/* Clamp a signed Q4_0 value (range -8..+7) to [-clamp_level, +clamp_level].
 * Returns the nibble value (unsigned, with +8 bias applied). */
static int g_clamp_level = 1;  /* default: ternary */

static inline uint8_t clamp_nibble(int8_t signed_val) {
    if (signed_val > g_clamp_level) signed_val = g_clamp_level;
    if (signed_val < -g_clamp_level) signed_val = -g_clamp_level;
    return (uint8_t)(signed_val + 8);
}

int main(int argc, char *argv[]) {
    /* Parse arguments */
    int argi = 1;
    if (argc >= 3 && strcmp(argv[1], "-l") == 0) {
        g_clamp_level = atoi(argv[2]);
        if (g_clamp_level < 1 || g_clamp_level > 7) {
            fprintf(stderr, "Clamp level must be 1-7\n");
            return 1;
        }
        argi = 3;
    }
    if (argc - argi != 2) {
        fprintf(stderr, "Usage: %s [-l levels] input.gguf output.gguf\n", argv[0]);
        fprintf(stderr, "  levels: 1=ternary, 2=5-level, 3=7-level, 7=full (default: 1)\n");
        return 1;
    }

    const char *in_path = argv[argi];
    const char *out_path = argv[argi + 1];

    /* Read entire file */
    FILE *f = fopen(in_path, "rb");
    if (!f) { perror("open input"); return 1; }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *data = malloc(file_size);
    if (!data) { fprintf(stderr, "malloc failed for %ld bytes\n", file_size); return 1; }
    if (fread(data, 1, file_size, f) != (size_t)file_size) {
        fprintf(stderr, "short read\n"); return 1;
    }
    fclose(f);

    printf("Loaded %s (%ld bytes = %.1f MiB)\n", in_path, file_size, file_size / (1024.0 * 1024.0));
    printf("Clamp level: %d (values clamped to [-%d, +%d] = %d levels)\n",
           g_clamp_level, g_clamp_level, g_clamp_level, 2 * g_clamp_level + 1);

    /* Parse GGUF header */
    const uint8_t *p = data;
    uint32_t magic = *(const uint32_t *)p; p += 4;
    if (magic != GGUF_MAGIC) { fprintf(stderr, "Not a GGUF file\n"); return 1; }

    uint32_t version = *(const uint32_t *)p; p += 4;
    uint64_t n_tensors = *(const uint64_t *)p; p += 8;
    uint64_t n_kv = *(const uint64_t *)p; p += 8;

    printf("GGUF v%u: %llu tensors, %llu KV pairs\n", version, n_tensors, n_kv);

    /* Skip KV metadata */
    for (uint64_t i = 0; i < n_kv; i++) {
        p = skip_gguf_string(p);        /* key */
        uint32_t vtype = *(const uint32_t *)p; p += 4;
        p = skip_gguf_value(p, vtype);  /* value */
    }

    /* Read tensor infos */
    TensorInfo *tensors = calloc(n_tensors, sizeof(TensorInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        read_gguf_string(p, tensors[i].name, sizeof(tensors[i].name));
        p = skip_gguf_string(p);
        tensors[i].n_dims = *(const uint32_t *)p; p += 4;
        uint64_t n_el = 1;
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            tensors[i].dims[d] = *(const uint64_t *)p; p += 8;
            n_el *= tensors[i].dims[d];
        }
        tensors[i].type = *(const uint32_t *)p; p += 4;
        tensors[i].offset = *(const uint64_t *)p; p += 8;
        tensors[i].n_elements = n_el;
        tensors[i].n_blocks = n_el / 32;  /* Q4_0: 32 weights per block */
    }

    /* Data section starts at aligned offset */
    uint64_t data_offset = ((p - data) + 31) & ~31ULL;

    /* Process each Q4_0 tensor in-place */
    uint64_t total_weights = 0;
    uint64_t total_clamped = 0;
    uint64_t total_zeros = 0;
    int q4_tensor_count = 0;

    printf("\n%-40s %10s %10s %10s %10s\n", "Tensor", "Weights", "Clamped", "Zeros", "Clamp%");
    printf("%-40s %10s %10s %10s %10s\n",
           "────────────────────────────────────────",
           "──────────", "──────────", "──────────", "──────────");

    for (uint64_t i = 0; i < n_tensors; i++) {
        if (tensors[i].type != GGUF_TYPE_Q4_0) continue;

        q4_tensor_count++;
        uint8_t *tensor_data = data + data_offset + tensors[i].offset;
        uint64_t n_blocks = tensors[i].n_blocks;
        uint64_t t_weights = 0, t_clamped = 0, t_zeros = 0;

        for (uint64_t b = 0; b < n_blocks; b++) {
            uint8_t *block = tensor_data + b * Q4_BLOCK_SIZE;
            /* block[0..1] = fp16 scale — leave untouched */
            uint8_t *nibbles = block + 2;

            for (int j = 0; j < 16; j++) {
                uint8_t byte = nibbles[j];

                /* Low nibble: weight at position 2*j */
                int8_t lo_signed = (byte & 0x0F) - 8;
                uint8_t lo_tern = clamp_nibble(lo_signed);

                /* High nibble: weight at position 2*j+1 */
                int8_t hi_signed = ((byte >> 4) & 0x0F) - 8;
                uint8_t hi_tern = clamp_nibble(hi_signed);

                /* Count statistics */
                t_weights += 2;
                if (lo_tern != (uint8_t)((byte & 0x0F))) t_clamped++;
                if (hi_tern != (uint8_t)((byte >> 4) & 0x0F)) t_clamped++;
                if (lo_tern == 8) t_zeros++;
                if (hi_tern == 8) t_zeros++;

                /* Write back */
                nibbles[j] = (hi_tern << 4) | lo_tern;
            }
        }

        printf("%-40s %10llu %10llu %10llu %9.1f%%\n",
               tensors[i].name, t_weights, t_clamped, t_zeros,
               100.0 * t_clamped / t_weights);

        total_weights += t_weights;
        total_clamped += t_clamped;
        total_zeros += t_zeros;
    }

    printf("%-40s %10s %10s %10s %10s\n",
           "────────────────────────────────────────",
           "──────────", "──────────", "──────────", "──────────");
    printf("%-40s %10llu %10llu %10llu %9.1f%%\n",
           "TOTAL", total_weights, total_clamped, total_zeros,
           100.0 * total_clamped / total_weights);

    printf("\nConverted %d Q4_0 tensors.\n", q4_tensor_count);
    printf("Ternary distribution: zero=%.1f%%, non-zero=%.1f%%\n",
           100.0 * total_zeros / total_weights,
           100.0 * (total_weights - total_zeros) / total_weights);
    printf("Weights unchanged (already ternary): %.1f%%\n",
           100.0 * (total_weights - total_clamped) / total_weights);

    /* Write output */
    FILE *out = fopen(out_path, "wb");
    if (!out) { perror("open output"); return 1; }
    if (fwrite(data, 1, file_size, out) != (size_t)file_size) {
        fprintf(stderr, "short write\n"); return 1;
    }
    fclose(out);

    printf("\nWrote %s (%.1f MiB) — same size, ternary Q4_0 weights.\n",
           out_path, file_size / (1024.0 * 1024.0));

    free(data);
    free(tensors);
    return 0;
}
