#ifndef XNNPACK_CACHE_OPTIMIZATION_SNAPDRAGON_H
#define XNNPACK_CACHE_OPTIMIZATION_SNAPDRAGON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Snapdragon 480 L3 Cache Optimization
// Optimized for 4MB L3 cache with Cortex-A76 prefetching

#define SNAPDRAGON_480_L3_CACHE_SIZE (4 * 1024 * 1024)  // 4MB
#define SNAPDRAGON_480_L2_CACHE_SIZE (512 * 1024)       // 512KB per core
#define SNAPDRAGON_480_L1_CACHE_SIZE (64 * 1024)        // 64KB per core
#define SNAPDRAGON_480_CACHE_LINE_SIZE 64               // 64-byte cache lines

// Prefetch locality hints
#define PREFETCH_LOCality_LOW 0     // Low temporal locality
#define PREFETCH_LOCality_MEDIUM 1  // Medium temporal locality
#define PREFETCH_LOCality_HIGH 2    // High temporal locality

/**
 * Prefetch data into Snapdragon 480 L3 cache
 *
 * @param data Pointer to data to prefetch
 * @param size Size of data in bytes
 * @param locality Temporal locality hint (0-3)
 */
void prefetch_snapdragon_l3(const void* data, size_t size, int locality);

/**
 * Prefetch LFM model weights for optimal L3 cache usage
 *
 * @param weights Pointer to weight data
 * @param size Size of weight data
 */
void prefetch_lfm_weights_snapdragon(const void* weights, size_t size);

/**
 * Prefetch input activations for Snapdragon 480
 *
 * @param activations Pointer to activation data
 * @param channels Number of channels
 * @param height Height of activation map
 * @param width Width of activation map
 */
void prefetch_activations_snapdragon(const void* activations,
                                   size_t channels, size_t height, size_t width);

/**
 * Optimize memory layout for Snapdragon 480 cache hierarchy
 *
 * @param data Input data pointer
 * @param size Data size
 * @param element_size Size of each element
 * @return Pointer to cache-optimized data (may be the same as input)
 */
void* optimize_memory_layout_snapdragon(const void* data, size_t size, size_t element_size);

/**
 * Calculate optimal tile size for Snapdragon 480 L3 cache
 *
 * @param tensor_size Size of tensor in bytes
 * @param working_set_ratio Ratio of working set to cache size (0.0-1.0)
 * @return Optimal tile size in bytes
 */
size_t calculate_optimal_tile_size_snapdragon(size_t tensor_size, float working_set_ratio);

/**
 * Prefetch convolution weights for Snapdragon 480
 *
 * @param weights Pointer to convolution weights
 * @param input_channels Number of input channels
 * @param output_channels Number of output channels
 * @param kernel_height Kernel height
 * @param kernel_width Kernel width
 */
void prefetch_conv_weights_snapdragon(const void* weights,
                                    size_t input_channels, size_t output_channels,
                                    size_t kernel_height, size_t kernel_width);

/**
 * Setup cache partitioning for Snapdragon 480
 *
 * Optimizes cache usage by partitioning between different data types
 * (weights, activations, outputs)
 */
void setup_cache_partitioning_snapdragon(void);

/**
 * Cache performance monitoring for Snapdragon 480
 */
typedef struct {
    uint64_t l3_cache_accesses;
    uint64_t l3_cache_misses;
    uint64_t l2_cache_accesses;
    uint64_t l2_cache_misses;
    uint64_t prefetch_requests;
    uint64_t prefetch_hits;
    double l3_hit_rate;
    double l2_hit_rate;
    double prefetch_hit_rate;
} snapdragon_cache_metrics_t;

/**
 * Collect cache performance metrics
 *
 * @param metrics Pointer to metrics structure
 * @return 0 on success, errno on failure
 */
int collect_cache_metrics_snapdragon(snapdragon_cache_metrics_t* metrics);

/**
 * Print cache performance metrics
 *
 * @param metrics Metrics to print
 */
void print_cache_metrics_snapdragon(const snapdragon_cache_metrics_t* metrics);

/**
 * Optimize GEMM operation for Snapdragon 480 L3 cache
 *
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Optimal tile size for the operation
 */
size_t optimize_gemm_tiling_snapdragon(size_t m, size_t n, size_t k);

/**
 * Prefetch attention weights for Liquid AI models
 *
 * @param weights Pointer to attention weight matrix
 * @param seq_length Sequence length
 * @param hidden_size Hidden dimension size
 */
void prefetch_attention_weights_snapdragon(const void* weights,
                                         size_t seq_length, size_t hidden_size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // XNNPACK_CACHE_OPTIMIZATION_SNAPDRAGON_H