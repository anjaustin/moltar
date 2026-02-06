/*
 * libfabric - Uncached Memory Fabric Implementation
 */

#define _GNU_SOURCE
#include "fabric.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#ifdef __ANDROID__
#include <sched.h>
#endif

/* Size threshold for uncached allocation (16MB) */
#define UNCACHED_THRESHOLD (16 * 1024 * 1024)

/* Maximum prefetch threads */
#define MAX_PREFETCH_THREADS 6

/* Maximum pending prefetch requests */
#define MAX_PREFETCH_QUEUE 64

/* dma_heap ioctl structure */
struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0, struct dma_heap_allocation_data)

/* Global state */
static struct {
    int initialized;
    int uncached_available;
    int num_prefetch_threads;
    
    pthread_t prefetch_threads[MAX_PREFETCH_THREADS];
    volatile int prefetch_running;
    
    /* Prefetch queue */
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_cond;
    fabric_prefetch_req_t *queue[MAX_PREFETCH_QUEUE];
    int queue_head;
    int queue_tail;
    int queue_count;
    
    /* Statistics */
    fabric_stats_t stats;
    pthread_mutex_t stats_mutex;
} g_fabric = {0};

/* Try to allocate uncached memory via dma_heap */
static void *alloc_uncached_impl(size_t size, int *heap_fd, int *buf_fd) {
    /* Try MTK uncached heap first, then system-uncached */
    const char *heap_paths[] = {
        "/dev/dma_heap/mtk_mm-uncached",
        "/dev/dma_heap/system-uncached",
        NULL
    };
    
    *heap_fd = -1;
    for (int i = 0; heap_paths[i] != NULL; i++) {
        *heap_fd = open(heap_paths[i], O_RDONLY);
        if (*heap_fd >= 0) break;
    }
    
    if (*heap_fd < 0) {
        return NULL;
    }
    
    struct dma_heap_allocation_data alloc = {
        .len = size,
        .fd_flags = O_RDWR | O_CLOEXEC,
        .heap_flags = 0,
    };
    
    if (ioctl(*heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
        close(*heap_fd);
        *heap_fd = -1;
        return NULL;
    }
    
    *buf_fd = alloc.fd;
    
    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, *buf_fd, 0);
    if (ptr == MAP_FAILED) {
        close(*buf_fd);
        close(*heap_fd);
        *heap_fd = -1;
        *buf_fd = -1;
        return NULL;
    }
    
    return ptr;
}

/* Prefetch worker thread */
static void *prefetch_worker(void *arg) {
    int thread_id = (int)(intptr_t)arg;
    
#ifdef __ANDROID__
    /* Pin to LITTLE core */
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(thread_id, &set);  /* Cores 0-5 are LITTLE on MT6855 */
    sched_setaffinity(0, sizeof(set), &set);
#endif
    
    while (g_fabric.prefetch_running) {
        fabric_prefetch_req_t *req = NULL;
        
        /* Get next request from queue */
        pthread_mutex_lock(&g_fabric.queue_mutex);
        while (g_fabric.queue_count == 0 && g_fabric.prefetch_running) {
            pthread_cond_wait(&g_fabric.queue_cond, &g_fabric.queue_mutex);
        }
        
        if (!g_fabric.prefetch_running) {
            pthread_mutex_unlock(&g_fabric.queue_mutex);
            break;
        }
        
        if (g_fabric.queue_count > 0) {
            req = g_fabric.queue[g_fabric.queue_head];
            g_fabric.queue_head = (g_fabric.queue_head + 1) % MAX_PREFETCH_QUEUE;
            g_fabric.queue_count--;
        }
        pthread_mutex_unlock(&g_fabric.queue_mutex);
        
        if (req == NULL) continue;
        
        /* Perform prefetch by reading the memory region */
        if (req->alloc && req->alloc->ptr && req->length > 0) {
            volatile float sum = 0;
            float *ptr = (float *)req->alloc->ptr;
            size_t start = req->offset / sizeof(float);
            size_t count = req->length / sizeof(float);
            
            /* Read with stride to warm DRAM rows */
            for (size_t i = 0; i < count; i += 16) {
                sum += ptr[start + i];
            }
            
            /* Prevent optimization */
            (void)sum;
        }
        
        /* Mark complete */
        req->done = 1;
        
        /* Update stats */
        pthread_mutex_lock(&g_fabric.stats_mutex);
        g_fabric.stats.prefetch_completed++;
        pthread_mutex_unlock(&g_fabric.stats_mutex);
    }
    
    return NULL;
}

int fabric_init(int num_prefetch_threads) {
    if (g_fabric.initialized) {
        return 0;
    }
    
    memset(&g_fabric, 0, sizeof(g_fabric));
    
    /* Check if uncached memory is available */
    int heap_fd, buf_fd;
    void *test = alloc_uncached_impl(4096, &heap_fd, &buf_fd);
    if (test) {
        munmap(test, 4096);
        close(buf_fd);
        close(heap_fd);
        g_fabric.uncached_available = 1;
    }
    
    /* Initialize mutexes */
    pthread_mutex_init(&g_fabric.queue_mutex, NULL);
    pthread_cond_init(&g_fabric.queue_cond, NULL);
    pthread_mutex_init(&g_fabric.stats_mutex, NULL);
    
    /* Start prefetch threads */
    if (num_prefetch_threads > MAX_PREFETCH_THREADS) {
        num_prefetch_threads = MAX_PREFETCH_THREADS;
    }
    if (num_prefetch_threads < 0) {
        num_prefetch_threads = 0;
    }
    
    g_fabric.num_prefetch_threads = num_prefetch_threads;
    g_fabric.prefetch_running = 1;
    
    for (int i = 0; i < num_prefetch_threads; i++) {
        pthread_create(&g_fabric.prefetch_threads[i], NULL, 
                       prefetch_worker, (void *)(intptr_t)i);
    }
    
    g_fabric.initialized = 1;
    return 0;
}

void fabric_shutdown(void) {
    if (!g_fabric.initialized) {
        return;
    }
    
    /* Stop prefetch threads */
    g_fabric.prefetch_running = 0;
    pthread_cond_broadcast(&g_fabric.queue_cond);
    
    for (int i = 0; i < g_fabric.num_prefetch_threads; i++) {
        pthread_join(g_fabric.prefetch_threads[i], NULL);
    }
    
    pthread_mutex_destroy(&g_fabric.queue_mutex);
    pthread_cond_destroy(&g_fabric.queue_cond);
    pthread_mutex_destroy(&g_fabric.stats_mutex);
    
    g_fabric.initialized = 0;
}

fabric_alloc_t *fabric_alloc(size_t size, fabric_pattern_t pattern) {
    if (!g_fabric.initialized) {
        if (fabric_init(2) != 0) {  /* Auto-init with 2 prefetch threads */
            return NULL;
        }
    }
    
    fabric_alloc_t *alloc = (fabric_alloc_t *)calloc(1, sizeof(fabric_alloc_t));
    if (!alloc) {
        return NULL;
    }
    
    alloc->size = size;
    alloc->heap_fd = -1;
    alloc->buf_fd = -1;
    
    /* Decide memory class based on pattern and size */
    int use_uncached = 0;
    
    switch (pattern) {
        case FABRIC_PATTERN_SEQUENTIAL:
        case FABRIC_PATTERN_REPEATED:
            /* Always use cached for sequential/repeated */
            use_uncached = 0;
            break;
            
        case FABRIC_PATTERN_RANDOM:
            /* Use uncached for large random access */
            use_uncached = (size >= UNCACHED_THRESHOLD) && g_fabric.uncached_available;
            break;
            
        case FABRIC_PATTERN_AUTO:
        default:
            /* Heuristic: large allocations get uncached */
            use_uncached = (size >= UNCACHED_THRESHOLD) && g_fabric.uncached_available;
            break;
    }
    
    if (use_uncached) {
        alloc->ptr = alloc_uncached_impl(size, &alloc->heap_fd, &alloc->buf_fd);
        if (alloc->ptr) {
            alloc->mem_class = FABRIC_MEM_UNCACHED;
            
            pthread_mutex_lock(&g_fabric.stats_mutex);
            g_fabric.stats.uncached_allocs++;
            g_fabric.stats.uncached_bytes += size;
            pthread_mutex_unlock(&g_fabric.stats_mutex);
            
            return alloc;
        }
        /* Fall back to cached if uncached fails */
    }
    
    /* Cached allocation */
    alloc->ptr = aligned_alloc(4096, size);
    if (!alloc->ptr) {
        free(alloc);
        return NULL;
    }
    
    alloc->mem_class = FABRIC_MEM_CACHED;
    
    pthread_mutex_lock(&g_fabric.stats_mutex);
    g_fabric.stats.cached_allocs++;
    g_fabric.stats.cached_bytes += size;
    pthread_mutex_unlock(&g_fabric.stats_mutex);
    
    return alloc;
}

void fabric_free(fabric_alloc_t *alloc) {
    if (!alloc) return;
    
    pthread_mutex_lock(&g_fabric.stats_mutex);
    if (alloc->mem_class == FABRIC_MEM_UNCACHED) {
        g_fabric.stats.uncached_allocs--;
        g_fabric.stats.uncached_bytes -= alloc->size;
    } else {
        g_fabric.stats.cached_allocs--;
        g_fabric.stats.cached_bytes -= alloc->size;
    }
    pthread_mutex_unlock(&g_fabric.stats_mutex);
    
    if (alloc->mem_class == FABRIC_MEM_UNCACHED) {
        if (alloc->ptr) {
            munmap(alloc->ptr, alloc->size);
        }
        if (alloc->buf_fd >= 0) {
            close(alloc->buf_fd);
        }
        if (alloc->heap_fd >= 0) {
            close(alloc->heap_fd);
        }
    } else {
        free(alloc->ptr);
    }
    
    free(alloc);
}

fabric_prefetch_req_t *fabric_prefetch_async(fabric_alloc_t *alloc, 
                                              size_t offset, 
                                              size_t length) {
    if (!alloc || !g_fabric.initialized) {
        return NULL;
    }
    
    /* Only prefetch uncached allocations */
    if (alloc->mem_class != FABRIC_MEM_UNCACHED) {
        return NULL;
    }
    
    /* No prefetch threads available */
    if (g_fabric.num_prefetch_threads == 0) {
        return NULL;
    }
    
    fabric_prefetch_req_t *req = (fabric_prefetch_req_t *)calloc(1, sizeof(fabric_prefetch_req_t));
    if (!req) {
        return NULL;
    }
    
    req->alloc = alloc;
    req->offset = offset;
    req->length = length;
    req->done = 0;
    
    /* Add to queue */
    pthread_mutex_lock(&g_fabric.queue_mutex);
    
    if (g_fabric.queue_count >= MAX_PREFETCH_QUEUE) {
        /* Queue full, drop request */
        pthread_mutex_unlock(&g_fabric.queue_mutex);
        free(req);
        return NULL;
    }
    
    g_fabric.queue[g_fabric.queue_tail] = req;
    g_fabric.queue_tail = (g_fabric.queue_tail + 1) % MAX_PREFETCH_QUEUE;
    g_fabric.queue_count++;
    
    g_fabric.stats.prefetch_requests++;
    
    pthread_cond_signal(&g_fabric.queue_cond);
    pthread_mutex_unlock(&g_fabric.queue_mutex);
    
    return req;
}

void fabric_prefetch_wait(fabric_prefetch_req_t *req) {
    if (!req) return;
    
    while (!req->done) {
        usleep(10);
    }
    
    free(req);
}

void fabric_prefetch(fabric_alloc_t *alloc, size_t offset, size_t length) {
    fabric_prefetch_req_t *req = fabric_prefetch_async(alloc, offset, length);
    if (req) {
        fabric_prefetch_wait(req);
    }
}

void fabric_get_stats(fabric_stats_t *stats) {
    if (!stats) return;
    
    pthread_mutex_lock(&g_fabric.stats_mutex);
    memcpy(stats, &g_fabric.stats, sizeof(fabric_stats_t));
    pthread_mutex_unlock(&g_fabric.stats_mutex);
}

int fabric_uncached_available(void) {
    if (!g_fabric.initialized) {
        if (fabric_init(0) != 0) {
            return 0;
        }
    }
    return g_fabric.uncached_available;
}

fabric_mem_class_t fabric_get_mem_class(fabric_alloc_t *alloc) {
    if (!alloc) return FABRIC_MEM_CACHED;
    return alloc->mem_class;
}

void *fabric_get_ptr(fabric_alloc_t *alloc) {
    if (!alloc) return NULL;
    return alloc->ptr;
}

size_t fabric_get_size(fabric_alloc_t *alloc) {
    if (!alloc) return 0;
    return alloc->size;
}
