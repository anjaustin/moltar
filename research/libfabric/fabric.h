/*
 * libfabric - Uncached Memory Fabric for Mobile LLM Inference
 * 
 * This library provides uncached memory allocation via Android's dma_heap,
 * enabling cross-core memory cooperation without cache coherency overhead.
 * 
 * Key insight: For large random-access working sets (like KV cache),
 * uncached memory is faster than cached due to eliminating coherency protocol.
 * 
 * Measured results:
 *   - Uncached random access: 1.3x faster than cached
 *   - Uncached with prefetch: 1.4x faster, 10x lower variance
 *   - Cached cross-core: 0.75x (coherency overhead hurts)
 *   - Uncached cross-core: 1.09x (no coherency, row warming works)
 */

#ifndef FABRIC_H
#define FABRIC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Memory class types */
typedef enum {
    FABRIC_MEM_CACHED,      /* Normal malloc - for sequential/repeated/small */
    FABRIC_MEM_UNCACHED,    /* dma_heap - for large random access */
} fabric_mem_class_t;

/* Access pattern hints */
typedef enum {
    FABRIC_PATTERN_AUTO,        /* Let fabric decide */
    FABRIC_PATTERN_SEQUENTIAL,  /* Sequential reads - use cached */
    FABRIC_PATTERN_RANDOM,      /* Random access - use uncached if large */
    FABRIC_PATTERN_REPEATED,    /* Repeated access - use cached */
} fabric_pattern_t;

/* Allocation handle */
typedef struct fabric_alloc {
    void *ptr;                  /* Memory pointer */
    size_t size;                /* Allocation size */
    fabric_mem_class_t mem_class;   /* Memory class */
    int heap_fd;                /* dma_heap fd (uncached only) */
    int buf_fd;                 /* buffer fd (uncached only) */
} fabric_alloc_t;

/* Prefetch request handle */
typedef struct fabric_prefetch_req {
    fabric_alloc_t *alloc;      /* Allocation to prefetch */
    size_t offset;              /* Start offset */
    size_t length;              /* Bytes to prefetch */
    volatile int done;          /* Set to 1 when complete */
} fabric_prefetch_req_t;

/*
 * Initialize the fabric library.
 * Call once at startup.
 * 
 * num_prefetch_threads: Number of LITTLE cores to use for prefetch (0-6)
 *                       Recommended: 2
 * 
 * Returns 0 on success, -1 on failure.
 */
int fabric_init(int num_prefetch_threads);

/*
 * Shutdown the fabric library.
 * Call once at exit.
 */
void fabric_shutdown(void);

/*
 * Allocate memory with specified pattern hint.
 * 
 * size: Bytes to allocate
 * pattern: Access pattern hint (AUTO uses size heuristic)
 * 
 * For FABRIC_PATTERN_AUTO:
 *   - size < 16MB: uses cached
 *   - size >= 16MB + RANDOM: uses uncached
 *   - size >= 16MB + SEQUENTIAL: uses cached (HW prefetch)
 * 
 * Returns allocation handle, or NULL on failure.
 */
fabric_alloc_t *fabric_alloc(size_t size, fabric_pattern_t pattern);

/*
 * Free a fabric allocation.
 */
void fabric_free(fabric_alloc_t *alloc);

/*
 * Request asynchronous prefetch of a memory region.
 * Only works on UNCACHED allocations.
 * 
 * For cached allocations, this is a no-op (HW prefetch handles it).
 * 
 * Returns prefetch request handle. Check req->done for completion.
 * Returns NULL if prefetch threads not available.
 */
fabric_prefetch_req_t *fabric_prefetch_async(fabric_alloc_t *alloc, 
                                              size_t offset, 
                                              size_t length);

/*
 * Wait for a prefetch request to complete.
 */
void fabric_prefetch_wait(fabric_prefetch_req_t *req);

/*
 * Synchronous prefetch - blocks until complete.
 */
void fabric_prefetch(fabric_alloc_t *alloc, size_t offset, size_t length);

/*
 * Get statistics about fabric allocations.
 */
typedef struct fabric_stats {
    size_t cached_allocs;       /* Number of cached allocations */
    size_t cached_bytes;        /* Total cached bytes */
    size_t uncached_allocs;     /* Number of uncached allocations */
    size_t uncached_bytes;      /* Total uncached bytes */
    size_t prefetch_requests;   /* Total prefetch requests */
    size_t prefetch_completed;  /* Completed prefetch requests */
} fabric_stats_t;

void fabric_get_stats(fabric_stats_t *stats);

/*
 * Query if uncached memory is available on this platform.
 */
int fabric_uncached_available(void);

/*
 * Get the memory class of an allocation.
 */
fabric_mem_class_t fabric_get_mem_class(fabric_alloc_t *alloc);

/*
 * Get the raw pointer from an allocation.
 */
void *fabric_get_ptr(fabric_alloc_t *alloc);

/*
 * Get the size of an allocation.
 */
size_t fabric_get_size(fabric_alloc_t *alloc);

#ifdef __cplusplus
}
#endif

#endif /* FABRIC_H */
