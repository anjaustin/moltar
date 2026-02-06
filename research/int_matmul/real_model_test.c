/*
 * Real Model Validation - Test hybrid matvec with actual LFM2 weights
 * 
 * Loads Q4_0 tensors from a GGUF file and compares baseline vs hybrid.
 *
 * Build for Android:
 *   $NDK/.../aarch64-linux-android24-clang -O3 -march=armv8.2-a+dotprod \
 *     -o real_model_test real_model_test.c -static -lm
 *
 * Usage:
 *   ./real_model_test /path/to/model.gguf [tensor_name]
 */

#define _GNU_SOURCE
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

extern void* memalign(size_t alignment, size_t size);
#define aligned_alloc(align, size) memalign(align, size)

#define GGUF_MAGIC 0x46554747
#define GGUF_TYPE_Q4_0 2
#define BLOCK_SIZE 32

// Q4_0 block (matches GGUF)
typedef struct {
    uint16_t d;        // FP16 scale
    uint8_t qs[16];    // 32 x 4-bit weights
} block_q4_0;

// Tensor info
typedef struct {
    char name[256];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
} tensor_info_t;

// FP16 -> FP32
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp += 127 - 15;
    }
    
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// GGUF string skip
static const uint8_t* skip_string(const uint8_t* p) {
    uint64_t len = *(const uint64_t*)p;
    return p + 8 + len;
}

static void read_string(const uint8_t* p, char* out, int maxlen) {
    uint64_t len = *(const uint64_t*)p;
    int copy = (len < (uint64_t)maxlen - 1) ? (int)len : maxlen - 1;
    memcpy(out, p + 8, copy);
    out[copy] = '\0';
}

// Skip GGUF value based on type
static const uint8_t* skip_value(const uint8_t* p, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return p + 1;
        case 2: case 3: return p + 2;
        case 4: case 5: case 6: return p + 4;
        case 10: case 11: case 12: return p + 8;
        case 8: return skip_string(p);
        case 9: {
            uint32_t et = *(const uint32_t*)p; p += 4;
            uint64_t cnt = *(const uint64_t*)p; p += 8;
            for (uint64_t i = 0; i < cnt; i++) p = skip_value(p, et);
            return p;
        }
        default: return p;
    }
}

// ============================================================================
// BASELINE: Float per-block scaling
// ============================================================================
void matvec_baseline(
    int n_out, int n_in,
    const block_q4_0* weights,
    const int8_t* activations,
    const float* act_scales,
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        float sum = 0.0f;
        
        for (int b = 0; b < nb; b++) {
            uint8x16_t q4_packed = vld1q_u8(row_weights[b].qs);
            int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
            int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
            q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
            q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
            
            int8x16_t act_lo = vld1q_s8(activations + b * 32);
            int8x16_t act_hi = vld1q_s8(activations + b * 32 + 16);
            
            int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
            int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
            int32_t dot = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
            
            float w_scale = fp16_to_fp32(row_weights[b].d);
            sum += (float)dot * w_scale * act_scales[b];
        }
        
        output[row] = sum;
    }
}

// ============================================================================
// HYBRID: Precomputed combined scales
// ============================================================================
void matvec_hybrid(
    int n_out, int n_in,
    const block_q4_0* weights,
    const int8_t* activations,
    const float* combined_scales,
    float* output
) {
    int nb = n_in / BLOCK_SIZE;
    
    for (int row = 0; row < n_out; row++) {
        const block_q4_0* row_weights = weights + row * nb;
        const float* row_scales = combined_scales + row * nb;
        
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        int b = 0;
        for (; b + 3 < nb; b += 4) {
            int32_t dots[4];
            
            for (int i = 0; i < 4; i++) {
                uint8x16_t q4_packed = vld1q_u8(row_weights[b + i].qs);
                int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
                int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
                q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
                q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
                
                int8x16_t act_lo = vld1q_s8(activations + (b + i) * 32);
                int8x16_t act_hi = vld1q_s8(activations + (b + i) * 32 + 16);
                
                int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
                int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
                dots[i] = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
            }
            
            int32x4_t dot_vec = vld1q_s32(dots);
            float32x4_t scale_vec = vld1q_f32(row_scales + b);
            acc = vmlaq_f32(acc, vcvtq_f32_s32(dot_vec), scale_vec);
        }
        
        float sum = vaddvq_f32(acc);
        
        for (; b < nb; b++) {
            uint8x16_t q4_packed = vld1q_u8(row_weights[b].qs);
            int8x16_t q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_packed, vdupq_n_u8(0x0F)));
            int8x16_t q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_packed, 4));
            q4_lo = vsubq_s8(q4_lo, vdupq_n_s8(8));
            q4_hi = vsubq_s8(q4_hi, vdupq_n_s8(8));
            
            int8x16_t act_lo = vld1q_s8(activations + b * 32);
            int8x16_t act_hi = vld1q_s8(activations + b * 32 + 16);
            
            int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, act_lo);
            int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, act_hi);
            int32_t dot = vaddvq_s32(vaddq_s32(dot_lo, dot_hi));
            
            sum += (float)dot * row_scales[b];
        }
        
        output[row] = sum;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.gguf> [tensor_name_pattern]\n", argv[0]);
        printf("Example: %s LFM2-350M-Q4_0.gguf ffn_gate\n", argv[0]);
        return 1;
    }
    
    const char* path = argv[1];
    const char* pattern = argc > 2 ? argv[2] : "ffn_gate";
    
    // Memory-map the file
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    
    uint8_t* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }
    
    printf("Loaded %s (%.1f MiB)\n", path, file_size / (1024.0 * 1024.0));
    
    // Parse GGUF header
    const uint8_t* p = data;
    uint32_t magic = *(const uint32_t*)p; p += 4;
    if (magic != GGUF_MAGIC) {
        printf("Not a GGUF file\n");
        return 1;
    }
    
    p += 4; // version
    uint64_t n_tensors = *(const uint64_t*)p; p += 8;
    uint64_t n_kv = *(const uint64_t*)p; p += 8;
    
    // Skip KV pairs
    for (uint64_t i = 0; i < n_kv; i++) {
        p = skip_string(p);
        uint32_t vtype = *(const uint32_t*)p; p += 4;
        p = skip_value(p, vtype);
    }
    
    // Find Q4_0 tensor matching pattern
    tensor_info_t* tensors = calloc(n_tensors, sizeof(tensor_info_t));
    int found_count = 0;
    
    for (uint64_t i = 0; i < n_tensors; i++) {
        read_string(p, tensors[i].name, sizeof(tensors[i].name));
        p = skip_string(p);
        tensors[i].n_dims = *(const uint32_t*)p; p += 4;
        
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            tensors[i].dims[d] = *(const uint64_t*)p; p += 8;
        }
        
        tensors[i].type = *(const uint32_t*)p; p += 4;
        tensors[i].offset = *(const uint64_t*)p; p += 8;
        
        if (tensors[i].type == GGUF_TYPE_Q4_0 && strstr(tensors[i].name, pattern)) {
            printf("Found: %s [%llu x %llu] Q4_0\n", 
                   tensors[i].name, tensors[i].dims[1], tensors[i].dims[0]);
            found_count++;
        }
    }
    
    if (found_count == 0) {
        printf("No Q4_0 tensors matching '%s'\n", pattern);
        free(tensors);
        munmap(data, file_size);
        close(fd);
        return 1;
    }
    
    uint64_t data_offset = ((p - data) + 31) & ~31ULL;
    
    // Test the first matching tensor
    tensor_info_t* test_tensor = NULL;
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (tensors[i].type == GGUF_TYPE_Q4_0 && strstr(tensors[i].name, pattern)) {
            test_tensor = &tensors[i];
            break;
        }
    }
    
    int n_out = test_tensor->dims[1];  // Output dimension
    int n_in = test_tensor->dims[0];   // Input dimension
    int nb = n_in / BLOCK_SIZE;
    
    printf("\nTesting: %s (%d x %d)\n", test_tensor->name, n_out, n_in);
    
    // Get pointer to tensor data
    const block_q4_0* weights = (const block_q4_0*)(data + data_offset + test_tensor->offset);
    
    // Analyze weight scales
    float min_scale = 1e10, max_scale = 0, sum_scale = 0;
    for (int i = 0; i < n_out * nb; i++) {
        float s = fp16_to_fp32(weights[i].d);
        if (s < min_scale) min_scale = s;
        if (s > max_scale) max_scale = s;
        sum_scale += s;
    }
    printf("Weight scale range: [%.6f, %.6f], mean: %.6f\n", 
           min_scale, max_scale, sum_scale / (n_out * nb));
    
    // Create random activations
    int8_t* activations = aligned_alloc(64, n_in);
    float* act_scales = aligned_alloc(64, nb * sizeof(float));
    
    srand(42);
    for (int i = 0; i < n_in; i++) {
        activations[i] = (rand() % 256) - 128;
    }
    for (int b = 0; b < nb; b++) {
        act_scales[b] = 0.01f + (rand() % 50) * 0.001f;
    }
    
    // Precompute combined scales for hybrid
    float* combined_scales = aligned_alloc(64, n_out * nb * sizeof(float));
    for (int row = 0; row < n_out; row++) {
        for (int b = 0; b < nb; b++) {
            float ws = fp16_to_fp32(weights[row * nb + b].d);
            combined_scales[row * nb + b] = ws * act_scales[b];
        }
    }
    
    // Allocate outputs
    float* output_base = aligned_alloc(64, n_out * sizeof(float));
    float* output_hybrid = aligned_alloc(64, n_out * sizeof(float));
    
    // Warmup
    matvec_baseline(n_out, n_in, weights, activations, act_scales, output_base);
    matvec_hybrid(n_out, n_in, weights, activations, combined_scales, output_hybrid);
    
    // Benchmark
    int iters = 100;
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_baseline(n_out, n_in, weights, activations, act_scales, output_base);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double baseline_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iters; i++) {
        matvec_hybrid(n_out, n_in, weights, activations, combined_scales, output_hybrid);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double hybrid_us = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3) / iters;
    
    // Compute error
    float max_abs_err = 0, max_rel_err = 0, sum_sq_err = 0;
    for (int i = 0; i < n_out; i++) {
        float abs_err = fabsf(output_base[i] - output_hybrid[i]);
        float rel_err = (fabsf(output_base[i]) > 1e-6f) ? abs_err / fabsf(output_base[i]) : 0;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        sum_sq_err += abs_err * abs_err;
    }
    float rmse = sqrtf(sum_sq_err / n_out);
    
    printf("\nResults:\n");
    printf("  Baseline:  %.1f us\n", baseline_us);
    printf("  Hybrid:    %.1f us (%.2fx speedup)\n", hybrid_us, baseline_us / hybrid_us);
    printf("  Max abs error: %.2e\n", max_abs_err);
    printf("  Max rel error: %.2e (%.4f%%)\n", max_rel_err, max_rel_err * 100);
    printf("  RMSE: %.2e\n", rmse);
    
    printf("\nSample outputs (first 5):\n");
    for (int i = 0; i < 5 && i < n_out; i++) {
        printf("  [%d] base=%.4f, hybrid=%.4f, diff=%.2e\n",
               i, output_base[i], output_hybrid[i], 
               fabsf(output_base[i] - output_hybrid[i]));
    }
    
    // Memory overhead
    size_t weight_bytes = n_out * nb * sizeof(block_q4_0);
    size_t scale_bytes = n_out * nb * sizeof(float);
    printf("\nMemory:\n");
    printf("  Weights: %.2f MB\n", weight_bytes / (1024.0 * 1024.0));
    printf("  Combined scales: %.2f MB (+%.1f%%)\n", 
           scale_bytes / (1024.0 * 1024.0),
           100.0 * scale_bytes / weight_bytes);
    
    // Cleanup
    free(tensors);
    free(activations);
    free(act_scales);
    free(combined_scales);
    free(output_base);
    free(output_hybrid);
    munmap(data, file_size);
    close(fd);
    
    return 0;
}
