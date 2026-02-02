/*
 * q4_to_ternary_v2.c — Dequant-Requant Ternarization with Optimal Scales
 *
 * The v1 converter clamped nibbles but kept original scales — which were
 * calibrated for 16-level Q4_0. That's a 7x magnitude error on tail weights.
 * The model collapsed.
 *
 * v2 fixes this:
 *   1. Dequantize each Q4_0 block to float32 (using the original scale)
 *   2. Compute ternary sign: +1, 0, or -1
 *   3. Compute OPTIMAL scale for ternary reconstruction:
 *      new_scale = mean(|float_i|) for all non-zero weights in the block
 *   4. Encode: new fp16 scale + ternary nibbles {7,8,9}
 *
 * The result is a valid Q4_0 GGUF that llama.cpp reads natively.
 * llama.cpp reconstructs: (nibble - 8) * new_scale = sign * new_scale
 *
 * Build (host macOS):
 *   cc -O2 -o q4_to_ternary_v2 q4_to_ternary_v2.c -lm
 *
 * Usage:
 *   ./q4_to_ternary_v2 input.gguf output.gguf
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define GGUF_MAGIC 0x46554747
#define GGUF_TYPE_Q4_0 2
#define Q4_BLOCK_SIZE 18  /* 2 bytes fp16 scale + 16 bytes data */

typedef struct {
    char name[256];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
    uint64_t n_elements;
    uint64_t n_blocks;
} TensorInfo;

/* ── FP16 conversion ─────────────────────────────────────────── */

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

static uint16_t f32_to_f16(float val) {
    uint32_t f;
    memcpy(&f, &val, 4);
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;

    if (exp <= 0) {
        if (exp < -10) return sign;  /* too small, flush to zero */
        mant = (mant | 0x400) >> (1 - exp);
        return sign | mant;
    } else if (exp >= 31) {
        return sign | 0x7C00;  /* infinity */
    }
    return sign | (exp << 10) | mant;
}

/* ── GGUF parsing ────────────────────────────────────────────── */

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

/* ── Main ────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input.gguf output.gguf\n", argv[0]);
        return 1;
    }

    const char *in_path = argv[1];
    const char *out_path = argv[2];

    /* Read entire file */
    FILE *f = fopen(in_path, "rb");
    if (!f) { perror("open input"); return 1; }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *data = malloc(file_size);
    if (!data) { fprintf(stderr, "malloc failed\n"); return 1; }
    if (fread(data, 1, file_size, f) != (size_t)file_size) {
        fprintf(stderr, "short read\n"); return 1;
    }
    fclose(f);

    printf("Loaded %s (%.1f MiB)\n", in_path, file_size / (1024.0 * 1024.0));

    /* Parse header */
    const uint8_t *p = data;
    uint32_t magic = *(const uint32_t *)p; p += 4;
    if (magic != GGUF_MAGIC) { fprintf(stderr, "Not GGUF\n"); return 1; }
    p += 4; /* version */
    uint64_t n_tensors = *(const uint64_t *)p; p += 8;
    uint64_t n_kv = *(const uint64_t *)p; p += 8;

    /* Skip KV */
    for (uint64_t i = 0; i < n_kv; i++) {
        p = skip_gguf_string(p);
        uint32_t vtype = *(const uint32_t *)p; p += 4;
        p = skip_gguf_value(p, vtype);
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
        tensors[i].n_blocks = n_el / 32;
    }

    uint64_t data_offset = ((p - data) + 31) & ~31ULL;

    /* Process each Q4_0 tensor */
    double total_mse_v1 = 0, total_mse_v2 = 0;
    uint64_t total_weights = 0;
    uint64_t total_zero = 0;
    int q4_count = 0;

    printf("\n%-40s %10s %10s %10s %10s\n",
           "Tensor", "MSE(v1)", "MSE(v2)", "Improve", "Zeros%");
    printf("%-40s %10s %10s %10s %10s\n",
           "----------------------------------------",
           "----------", "----------", "----------", "----------");

    for (uint64_t ti = 0; ti < n_tensors; ti++) {
        if (tensors[ti].type != GGUF_TYPE_Q4_0) continue;
        q4_count++;

        uint8_t *tensor_data = data + data_offset + tensors[ti].offset;
        uint64_t nb = tensors[ti].n_blocks;
        double tensor_mse_v1 = 0, tensor_mse_v2 = 0;
        uint64_t t_weights = 0, t_zero = 0;

        for (uint64_t b = 0; b < nb; b++) {
            uint8_t *block = tensor_data + b * Q4_BLOCK_SIZE;

            /* Read original scale */
            uint16_t scale_h;
            memcpy(&scale_h, block, 2);
            float old_scale = f16_to_f32(scale_h);
            uint8_t *nibbles = block + 2;

            /* Step 1: Dequantize all 32 weights to float */
            float orig[32];
            int8_t signs[32];
            int n_nonzero = 0;
            float sum_abs = 0.0f;

            for (int j = 0; j < 16; j++) {
                uint8_t byte = nibbles[j];
                int8_t lo = (byte & 0x0F) - 8;
                int8_t hi = ((byte >> 4) & 0x0F) - 8;

                orig[j * 2]     = lo * old_scale;
                orig[j * 2 + 1] = hi * old_scale;

                /* Step 2: Compute ternary signs */
                signs[j * 2]     = (lo > 0) ? +1 : (lo < 0) ? -1 : 0;
                signs[j * 2 + 1] = (hi > 0) ? +1 : (hi < 0) ? -1 : 0;

                if (signs[j * 2] != 0) {
                    sum_abs += fabsf(orig[j * 2]);
                    n_nonzero++;
                }
                if (signs[j * 2 + 1] != 0) {
                    sum_abs += fabsf(orig[j * 2 + 1]);
                    n_nonzero++;
                }
            }

            /* Step 3: Compute optimal ternary scale */
            float new_scale = (n_nonzero > 0) ? (sum_abs / n_nonzero) : 0.0f;

            /* Compute MSE for v1 (old scale, clamped nibbles) and v2 (new scale) */
            double block_mse_v1 = 0, block_mse_v2 = 0;
            for (int k = 0; k < 32; k++) {
                float recon_v1 = signs[k] * old_scale;  /* v1: sign * old_scale */
                float recon_v2 = signs[k] * new_scale;  /* v2: sign * new_scale */
                float err_v1 = orig[k] - recon_v1;
                float err_v2 = orig[k] - recon_v2;
                block_mse_v1 += err_v1 * err_v1;
                block_mse_v2 += err_v2 * err_v2;
                if (signs[k] == 0) t_zero++;
            }
            tensor_mse_v1 += block_mse_v1;
            tensor_mse_v2 += block_mse_v2;
            t_weights += 32;

            /* Step 4: Write back — new scale + ternary nibbles */
            uint16_t new_scale_h = f32_to_f16(new_scale);
            memcpy(block, &new_scale_h, 2);

            for (int j = 0; j < 16; j++) {
                uint8_t lo_nib = (signs[j * 2] == 1) ? 9 :
                                 (signs[j * 2] == -1) ? 7 : 8;
                uint8_t hi_nib = (signs[j * 2 + 1] == 1) ? 9 :
                                 (signs[j * 2 + 1] == -1) ? 7 : 8;
                nibbles[j] = (hi_nib << 4) | lo_nib;
            }
        }

        tensor_mse_v1 /= t_weights;
        tensor_mse_v2 /= t_weights;
        float improve = (tensor_mse_v1 > 0) ?
            (1.0 - tensor_mse_v2 / tensor_mse_v1) * 100.0 : 0;

        printf("%-40s %10.6f %10.6f %9.1f%% %9.1f%%\n",
               tensors[ti].name, tensor_mse_v1, tensor_mse_v2, improve,
               100.0 * t_zero / t_weights);

        total_mse_v1 += tensor_mse_v1 * t_weights;
        total_mse_v2 += tensor_mse_v2 * t_weights;
        total_weights += t_weights;
        total_zero += t_zero;
    }

    double avg_mse_v1 = total_mse_v1 / total_weights;
    double avg_mse_v2 = total_mse_v2 / total_weights;
    float total_improve = (avg_mse_v1 > 0) ?
        (1.0 - avg_mse_v2 / avg_mse_v1) * 100.0 : 0;

    printf("%-40s %10s %10s %10s %10s\n",
           "----------------------------------------",
           "----------", "----------", "----------", "----------");
    printf("%-40s %10.6f %10.6f %9.1f%% %9.1f%%\n",
           "TOTAL (weighted avg)", avg_mse_v1, avg_mse_v2, total_improve,
           100.0 * total_zero / total_weights);

    printf("\nConverted %d Q4_0 tensors (%llu weights).\n", q4_count, total_weights);
    printf("Zeros: %.1f%%\n", 100.0 * total_zero / total_weights);
    printf("MSE reduction: v1=%.6f -> v2=%.6f (%.1f%% better)\n",
           avg_mse_v1, avg_mse_v2, total_improve);
    printf("RMSE: v1=%.4f -> v2=%.4f\n", sqrt(avg_mse_v1), sqrt(avg_mse_v2));

    /* Write output */
    FILE *out = fopen(out_path, "wb");
    if (!out) { perror("open output"); return 1; }
    if (fwrite(data, 1, file_size, out) != (size_t)file_size) {
        fprintf(stderr, "short write\n"); return 1;
    }
    fclose(out);

    printf("\nWrote %s (%.1f MiB)\n", out_path, file_size / (1024.0 * 1024.0));

    free(data);
    free(tensors);
    return 0;
}
