#ifndef XNNPACK_SNAPDRAGON_480_OPTIMIZATION_H
#define XNNPACK_SNAPDRAGON_480_OPTIMIZATION_H

#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <sched.h>

#ifdef __cplusplus
extern "C" {
#endif

// Snapdragon 480 Hardware Constants
#define SNAPDRAGON_480_BIG_CORES 2    // Cortex-A76 cores (0, 1)
#define SNAPDRAGON_480_LITTLE_CORES 6 // Cortex-A55 cores (2-7)
#define SNAPDRAGON_480_TOTAL_CORES 8
#define SNAPDRAGON_480_L3_CACHE_SIZE (4096 * 1024) // 4MB L3 cache

// CPU Affinity Masks for Snapdragon 480
#define SNAPDRAGON_480_BIG_CORE_MASK 0x03   // Cores 0-1 (A76)
#define SNAPDRAGON_480_LITTLE_CORE_MASK 0xFC // Cores 2-7 (A55)
#define SNAPDRAGON_480_ALL_CORE_MASK 0xFF    // All cores

// Hardware Capability Detection
typedef struct {
    bool has_dotprod;           // ARMv8.2-A +dotprod support
    bool has_fp16;             // Half-precision floating point
    bool has_sve;              // Scalable Vector Extension (if available)
    uint32_t l3_cache_size_kb; // L3 cache size
    uint32_t big_core_count;   // Number of big cores (A76)
    uint32_t little_core_count; // Number of little cores (A55)
    uint32_t max_frequency_mhz; // Maximum CPU frequency
} snapdragon_480_caps_t;

/**
 * Detect Snapdragon 480 hardware capabilities at runtime
 *
 * @return snapdragon_480_caps_t structure with detected capabilities
 */
snapdragon_480_caps_t detect_snapdragon_480_capabilities(void);

/**
 * Check if running on Snapdragon 480 with required features
 *
 * @return true if Snapdragon 480 with dotprod support detected
 */
bool is_snapdragon_480_with_dotprod(void);

/**
 * Get optimal thread count for Snapdragon 480 (big cores only)
 *
 * @return Recommended thread count (2 for big cores)
 */
uint32_t get_optimal_thread_count_snapdragon_480(void);

/**
 * Pin threads to Snapdragon 480 big cores (Cortex-A76)
 *
 * @param thread Thread to pin
 * @param thread_index Index of thread (0-based)
 * @return 0 on success, errno on failure
 */
int pin_thread_to_big_cores(pthread_t thread, uint32_t thread_index);

/**
 * Configure thread pool for Snapdragon 480 optimization
 *
 * @param threadpool Thread pool to configure
 * @param thread_count Number of threads to use
 * @return 0 on success, errno on failure
 */
int configure_threadpool_snapdragon_480(void* threadpool, uint32_t thread_count);

/**
 * Enable Snapdragon 480 specific optimizations in XNNPack
 *
 * This function should be called during XNNPack initialization
 * to enable dot product kernels and other Snapdragon-specific features.
 */
void enable_snapdragon_480_optimizations(void);

/**
 * Get Snapdragon 480 specific GEMM kernel function pointer
 *
 * @return Function pointer to optimized GEMM kernel, or NULL if not available
 */
void* get_snapdragon_480_gemm_kernel(void);

/**
 * Prefetch data for Snapdragon 480 L3 cache optimization
 *
 * @param data Pointer to data to prefetch
 * @param size Size of data in bytes
 * @param locality Locality hint (0-3, higher = more temporal)
 */
void prefetch_snapdragon_480_l3(const void* data, size_t size, int locality);

/**
 * Get cache line size for Snapdragon 480
 *
 * @return Cache line size in bytes (typically 64)
 */
size_t get_snapdragon_480_cache_line_size(void);

/**
 * Performance monitoring for Snapdragon 480 optimizations
 */
typedef struct {
    uint64_t dotprod_instructions;    // Number of dot product instructions executed
    uint64_t cache_misses_l3;         // L3 cache misses
    uint64_t big_core_time_us;        // Time spent on big cores (microseconds)
    uint64_t total_instructions;      // Total instructions executed
    double dotprod_utilization;       // Percentage of operations using dot product
    double cache_hit_rate;           // L3 cache hit rate
    double big_core_utilization;     // Percentage of time on big cores
} snapdragon_480_metrics_t;

/**
 * Collect performance metrics for Snapdragon 480 optimizations
 *
 * @param metrics Pointer to metrics structure to fill
 * @return 0 on success, errno on failure
 */
int collect_snapdragon_480_metrics(snapdragon_480_metrics_t* metrics);

/**
 * Print Snapdragon 480 performance metrics
 *
 * @param metrics Metrics to print
 */
void print_snapdragon_480_metrics(const snapdragon_480_metrics_t* metrics);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // XNNPACK_SNAPDRAGON_480_OPTIMIZATION_H