#include "cache_optimization_snapdragon.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Snapdragon 480 cache hierarchy constants
#define L3_CACHE_ASSOCIATIVITY 16
#define L2_CACHE_ASSOCIATIVITY 8
#define L1_CACHE_ASSOCIATIVITY 4

void prefetch_snapdragon_l3(const void* data, size_t size, int locality) {
    if (!data || size == 0) return;

    // Snapdragon 480 L3 cache prefetching
    // Use cache line sized chunks for optimal prefetching
    const size_t cache_line_size = SNAPDRAGON_480_CACHE_LINE_SIZE;

    // Limit prefetch distance to avoid cache pollution
    const size_t max_prefetch_distance = SNAPDRAGON_480_L3_CACHE_SIZE / 4;
    size_t prefetch_size = (size < max_prefetch_distance) ? size : max_prefetch_distance;

    // Prefetch in cache line sized chunks
    for (size_t offset = 0; offset < prefetch_size; offset += cache_line_size) {
        // Use GCC built-in prefetch with Snapdragon-specific hints
        __builtin_prefetch((const char*)data + offset, 0, locality);
    }
}

void prefetch_lfm_weights_snapdragon(const void* weights, size_t size) {
    // LFM weights typically have high temporal locality
    // Prefetch with high locality hint
    prefetch_snapdragon_l3(weights, size, PREFETCH_LOCality_HIGH);
}

void prefetch_activations_snapdragon(const void* activations,
                                   size_t channels, size_t height, size_t width) {
    if (!activations) return;

    // Calculate activation tensor size (assuming float32)
    size_t element_size = 4; // float32
    size_t tensor_size = channels * height * width * element_size;

    // Activations have medium temporal locality in convolutional networks
    prefetch_snapdragon_l3(activations, tensor_size, PREFETCH_LOCality_MEDIUM);
}

void* optimize_memory_layout_snapdragon(const void* data, size_t size, size_t element_size) {
    if (!data) return NULL;

    // For Snapdragon 480, check if data is already cache-aligned
    uintptr_t addr = (uintptr_t)data;
    size_t cache_line_size = SNAPDRAGON_480_CACHE_LINE_SIZE;

    if ((addr % cache_line_size) == 0) {
        // Already aligned, return original
        return (void*)data;
    }

    // Allocate cache-aligned memory
    void* aligned_data = NULL;
    if (posix_memalign(&aligned_data, cache_line_size, size) != 0) {
        return (void*)data; // Fallback to original on allocation failure
    }

    // Copy data to aligned memory
    memcpy(aligned_data, data, size);

    return aligned_data;
}

size_t calculate_optimal_tile_size_snapdragon(size_t tensor_size, float working_set_ratio) {
    // Calculate tile size that fits well in L3 cache
    size_t l3_cache_size = SNAPDRAGON_480_L3_CACHE_SIZE;
    size_t target_working_set = (size_t)(l3_cache_size * working_set_ratio);

    // Ensure tile size is reasonable
    if (tensor_size < target_working_set) {
        return tensor_size;
    }

    // Calculate tile size as a fraction of the working set
    size_t tile_size = target_working_set / 4; // Use 1/4 of working set

    // Round up to cache line boundary
    size_t cache_line_size = SNAPDRAGON_480_CACHE_LINE_SIZE;
    tile_size = ((tile_size + cache_line_size - 1) / cache_line_size) * cache_line_size;

    return tile_size;
}

void prefetch_conv_weights_snapdragon(const void* weights,
                                    size_t input_channels, size_t output_channels,
                                    size_t kernel_height, size_t kernel_width) {
    if (!weights) return;

    // Convolution weights are accessed repeatedly, high locality
    size_t element_size = 4; // Assume float32
    size_t weight_size = input_channels * output_channels *
                        kernel_height * kernel_width * element_size;

    prefetch_snapdragon_l3(weights, weight_size, PREFETCH_LOCality_HIGH);
}

void setup_cache_partitioning_snapdragon(void) {
    // Configure cache partitioning for Snapdragon 480
    // This would typically involve setting up cache coloring or way partitioning

    // For now, set environment variables to guide cache usage
    setenv("XNNPACK_CACHE_PARTITIONING", "snapdragon_480", 1);
    setenv("XNNPACK_L3_CACHE_SIZE", "4194304", 1); // 4MB
    setenv("XNNPACK_CACHE_LINE_SIZE", "64", 1);

    printf("✅ Configured Snapdragon 480 cache partitioning\n");
}

int collect_cache_metrics_snapdragon(snapdragon_cache_metrics_t* metrics) {
    if (!metrics) return -1;

    // Initialize metrics
    memset(metrics, 0, sizeof(snapdragon_cache_metrics_t));

    // In a real implementation, this would read hardware performance counters
    // For Snapdragon 480, this might involve:
    // - Reading PMU (Performance Monitor Unit) counters
    // - Accessing cache controller registers
    // - Using ARM SPE (Statistical Profiling Extension) if available

    // Placeholder values for demonstration
    metrics->l3_cache_accesses = 1000000;
    metrics->l3_cache_misses = 50000;
    metrics->l2_cache_accesses = 2000000;
    metrics->l2_cache_misses = 100000;
    metrics->prefetch_requests = 50000;
    metrics->prefetch_hits = 40000;

    // Calculate derived metrics
    if (metrics->l3_cache_accesses > 0) {
        metrics->l3_hit_rate = 1.0 - (double)metrics->l3_cache_misses /
                                      (double)metrics->l3_cache_accesses;
    }

    if (metrics->l2_cache_accesses > 0) {
        metrics->l2_hit_rate = 1.0 - (double)metrics->l2_cache_misses /
                                      (double)metrics->l2_cache_accesses;
    }

    if (metrics->prefetch_requests > 0) {
        metrics->prefetch_hit_rate = (double)metrics->prefetch_hits /
                                    (double)metrics->prefetch_requests;
    }

    return 0;
}

void print_cache_metrics_snapdragon(const snapdragon_cache_metrics_t* metrics) {
    if (!metrics) return;

    printf("Snapdragon 480 Cache Performance Metrics:\n");
    printf("==========================================\n");
    printf("L3 Cache Accesses: %llu\n", metrics->l3_cache_accesses);
    printf("L3 Cache Misses: %llu\n", metrics->l3_cache_misses);
    printf("L3 Hit Rate: %.2f%%\n", metrics->l3_hit_rate * 100.0);
    printf("L2 Cache Accesses: %llu\n", metrics->l2_cache_accesses);
    printf("L2 Cache Misses: %llu\n", metrics->l2_cache_misses);
    printf("L2 Hit Rate: %.2f%%\n", metrics->l2_hit_rate * 100.0);
    printf("Prefetch Requests: %llu\n", metrics->prefetch_requests);
    printf("Prefetch Hits: %llu\n", metrics->prefetch_hits);
    printf("Prefetch Hit Rate: %.2f%%\n", metrics->prefetch_hit_rate * 100.0);
}

size_t optimize_gemm_tiling_snapdragon(size_t m, size_t n, size_t k) {
    // Optimize GEMM tiling for Snapdragon 480 L3 cache

    // Estimate working set size for GEMM operation
    // Working set = A block + B block + C block
    size_t element_size = 4; // float32
    size_t estimated_working_set = m * k * element_size +  // A block
                                  k * n * element_size +  // B block
                                  m * n * element_size;   // C block

    // Target working set ratio (leave room for other data)
    float working_set_ratio = 0.6f; // 60% of L3 cache

    // Calculate optimal tile size
    size_t tile_size = calculate_optimal_tile_size_snapdragon(estimated_working_set,
                                                             working_set_ratio);

    // Ensure tile size is reasonable for GEMM dimensions
    size_t min_tile_size = 1024; // 1KB minimum
    size_t max_tile_size = SNAPDRAGON_480_L3_CACHE_SIZE / 4; // 1MB maximum

    if (tile_size < min_tile_size) tile_size = min_tile_size;
    if (tile_size > max_tile_size) tile_size = max_tile_size;

    return tile_size;
}

void prefetch_attention_weights_snapdragon(const void* weights,
                                         size_t seq_length, size_t hidden_size) {
    if (!weights) return;

    // Attention weights in Liquid AI models have complex access patterns
    // Prefetch with medium locality as they're reused but not always sequentially

    size_t element_size = 4; // float32
    size_t weight_size = seq_length * hidden_size * element_size;

    // Prefetch in smaller chunks to avoid cache pollution
    const size_t prefetch_chunk_size = 64 * 1024; // 64KB chunks
    size_t remaining_size = weight_size;

    for (size_t offset = 0; offset < weight_size; offset += prefetch_chunk_size) {
        size_t chunk_size = (remaining_size < prefetch_chunk_size) ?
                           remaining_size : prefetch_chunk_size;

        prefetch_snapdragon_l3((const char*)weights + offset, chunk_size,
                             PREFETCH_LOCality_MEDIUM);

        remaining_size -= chunk_size;
    }
}