/*
 * libfabric test harness v3
 * 
 * Tests the REAL benefit: cross-core prefetch overlapping with compute
 * 
 * Simulates LLM inference pattern:
 *   - BIG core computes on block N
 *   - LITTLE core prefetches block N+1 in parallel
 *   - Next iteration: BIG core accesses "warmed" block N+1
 */

#define _GNU_SOURCE
#include "fabric.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#ifdef __ANDROID__
#include <sched.h>
#endif

#define KV_CACHE_SIZE (64 * 1024 * 1024)  /* 64MB total */
#define BLOCK_SIZE (4 * 1024 * 1024)       /* 4MB per "layer" */
#define NUM_BLOCKS (KV_CACHE_SIZE / BLOCK_SIZE)  /* 16 blocks */
#define NUM_ACCESSES 2500                   /* Per block */
#define NUM_ITERATIONS 50
#define PAGE_SIZE 4096

/* Timing */
static inline uint64_t get_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

/* Cache flush for ARM */
static inline void dc_civac(void *addr) {
    __asm__ __volatile__("dc civac, %0" : : "r"(addr) : "memory");
}

static inline void dsb_sy(void) {
    __asm__ __volatile__("dsb sy" ::: "memory");
}

void flush_region(void *ptr, size_t size) {
    char *p = (char *)ptr;
    for (size_t i = 0; i < size; i += 64) {
        dc_civac(p + i);
    }
    dsb_sy();
}

/* Prefetch thread state */
typedef struct {
    volatile int should_exit;
    volatile int block_to_prefetch;
    volatile int prefetch_done;
    void *ptr;
    size_t block_size;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int cpu;  /* CPU to pin to */
} prefetch_state_t;

void *prefetch_thread(void *arg) {
    prefetch_state_t *state = (prefetch_state_t *)arg;
    
#ifdef __ANDROID__
    /* Pin to LITTLE core */
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(state->cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
#endif
    
    while (!state->should_exit) {
        pthread_mutex_lock(&state->lock);
        while (state->block_to_prefetch < 0 && !state->should_exit) {
            pthread_cond_wait(&state->cond, &state->lock);
        }
        int block = state->block_to_prefetch;
        state->block_to_prefetch = -1;
        pthread_mutex_unlock(&state->lock);
        
        if (state->should_exit) break;
        
        /* Do the prefetch - touch each cache line */
        volatile char *base = (volatile char *)state->ptr + (block * state->block_size);
        volatile char sum = 0;
        for (size_t i = 0; i < state->block_size; i += 64) {
            sum += base[i];  /* Touch to bring into memory subsystem */
        }
        
        __sync_synchronize();  /* Memory barrier */
        state->prefetch_done = 1;
    }
    
    return NULL;
}

/* Simulate compute on a block */
volatile float do_compute(float *ptr, size_t *offsets, int num_accesses, size_t block_offset) {
    volatile float sum = 0;
    for (int i = 0; i < num_accesses; i++) {
        sum += ptr[(block_offset + offsets[i]) / sizeof(float)];
    }
    return sum;
}

typedef struct {
    uint64_t p50;
    uint64_t p90;
    uint64_t p99;
    double mean;
    double cv;
} stats_t;

int compare_u64(const void *a, const void *b) {
    uint64_t ua = *(const uint64_t*)a;
    uint64_t ub = *(const uint64_t*)b;
    return (ua > ub) - (ua < ub);
}

stats_t calc_stats(uint64_t *times, int n) {
    qsort(times, n, sizeof(uint64_t), compare_u64);
    
    double mean = 0;
    for (int i = 0; i < n; i++) mean += times[i];
    mean /= n;
    
    double var = 0;
    for (int i = 0; i < n; i++) {
        var += (times[i] - mean) * (times[i] - mean);
    }
    
    stats_t st = {
        .p50 = times[n/2],
        .p90 = times[n*90/100],
        .p99 = times[n*99/100],
        .mean = mean,
        .cv = 100.0 * sqrt(var / n) / mean
    };
    return st;
}

void print_stats(const char *name, stats_t *st) {
    printf("%-40s p50=%6lu us  p90=%6lu us  p99=%6lu us  CV=%5.1f%%\n",
           name, st->p50, st->p90, st->p99, st->cv);
}

int main(int argc, char **argv) {
    printf("=== libfabric Test Harness v3 ===\n");
    printf("Testing cross-core prefetch pipeline\n\n");
    printf("Block size: %d MB, %d blocks total\n", BLOCK_SIZE / (1024*1024), NUM_BLOCKS);
    printf("Simulating layer-by-layer KV cache access\n\n");
    
#ifdef __ANDROID__
    /* Pin main thread to BIG core */
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(7, &set);
    sched_setaffinity(0, sizeof(set), &set);
    printf("Main thread pinned to CPU 7 (big core)\n");
#endif
    
    /* Initialize fabric (but we'll do manual prefetch thread for this test) */
    if (fabric_init(0) != 0) {  /* 0 prefetch threads - we manage our own */
        fprintf(stderr, "Failed to initialize fabric\n");
        return 1;
    }
    
    printf("Uncached memory available: %s\n\n", 
           fabric_uncached_available() ? "YES" : "NO");
    
    /* Generate random access pattern within a block */
    srand(42);
    size_t *offsets = (size_t *)malloc(NUM_ACCESSES * sizeof(size_t));
    for (int i = 0; i < NUM_ACCESSES; i++) {
        offsets[i] = (rand() % (BLOCK_SIZE / sizeof(float))) * sizeof(float);
    }
    
    uint64_t times[NUM_ITERATIONS];
    stats_t st;
    
    /* Allocate cached and uncached memory */
    float *cached_ptr = (float *)aligned_alloc(PAGE_SIZE, KV_CACHE_SIZE);
    memset(cached_ptr, 0, KV_CACHE_SIZE);
    
    fabric_alloc_t *fabric_uncached = fabric_alloc(KV_CACHE_SIZE, FABRIC_PATTERN_RANDOM);
    float *uncached_ptr = (float *)fabric_get_ptr(fabric_uncached);
    memset(uncached_ptr, 0, KV_CACHE_SIZE);
    
    printf("Cached ptr: %p\n", cached_ptr);
    printf("Uncached ptr: %p (class: %s)\n\n", uncached_ptr,
           fabric_get_mem_class(fabric_uncached) == FABRIC_MEM_UNCACHED ? "UNCACHED" : "CACHED");
    
    /* ============================================ */
    /* TEST 1: Cached, no prefetch (baseline)       */
    /* ============================================ */
    printf("TEST 1: Cached memory, NO prefetch (cold each block)\n");
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        /* Flush all cache first */
        flush_region(cached_ptr, KV_CACHE_SIZE);
        
        uint64_t t0 = get_us();
        for (int b = 0; b < NUM_BLOCKS; b++) {
            do_compute(cached_ptr, offsets, NUM_ACCESSES, b * BLOCK_SIZE);
        }
        uint64_t t1 = get_us();
        times[iter] = t1 - t0;
    }
    
    st = calc_stats(times, NUM_ITERATIONS);
    print_stats("Cached NO prefetch:", &st);
    uint64_t cached_no_prefetch = st.p50;
    
    /* ============================================ */
    /* TEST 2: Cached WITH cross-core prefetch      */
    /* ============================================ */
    printf("\nTEST 2: Cached memory, WITH cross-core prefetch (LITTLE->BIG)\n");
    
    prefetch_state_t pstate = {
        .should_exit = 0,
        .block_to_prefetch = -1,
        .prefetch_done = 0,
        .ptr = cached_ptr,
        .block_size = BLOCK_SIZE,
        .lock = PTHREAD_MUTEX_INITIALIZER,
        .cond = PTHREAD_COND_INITIALIZER,
        .cpu = 0  /* LITTLE core */
    };
    
    pthread_t prefetch_tid;
    pthread_create(&prefetch_tid, NULL, prefetch_thread, &pstate);
    usleep(10000);  /* Let thread start */
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        /* Flush all cache first */
        flush_region(cached_ptr, KV_CACHE_SIZE);
        
        uint64_t t0 = get_us();
        for (int b = 0; b < NUM_BLOCKS; b++) {
            /* Start prefetch of NEXT block */
            if (b < NUM_BLOCKS - 1) {
                pthread_mutex_lock(&pstate.lock);
                pstate.prefetch_done = 0;
                pstate.block_to_prefetch = b + 1;
                pthread_cond_signal(&pstate.cond);
                pthread_mutex_unlock(&pstate.lock);
            }
            
            /* Compute on current block */
            do_compute(cached_ptr, offsets, NUM_ACCESSES, b * BLOCK_SIZE);
            
            /* Wait for prefetch to complete (should already be done if overlap worked) */
            if (b < NUM_BLOCKS - 1) {
                while (!pstate.prefetch_done) {
                    __sync_synchronize();
                }
            }
        }
        uint64_t t1 = get_us();
        times[iter] = t1 - t0;
    }
    
    pstate.should_exit = 1;
    pthread_cond_signal(&pstate.cond);
    pthread_join(prefetch_tid, NULL);
    
    st = calc_stats(times, NUM_ITERATIONS);
    print_stats("Cached WITH cross-core prefetch:", &st);
    uint64_t cached_with_prefetch = st.p50;
    
    printf("  -> Speedup from prefetch: %.2fx\n", (double)cached_no_prefetch / cached_with_prefetch);
    
    /* ============================================ */
    /* TEST 3: Uncached, no prefetch                */
    /* ============================================ */
    printf("\nTEST 3: Uncached memory, NO prefetch\n");
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        uint64_t t0 = get_us();
        for (int b = 0; b < NUM_BLOCKS; b++) {
            do_compute(uncached_ptr, offsets, NUM_ACCESSES, b * BLOCK_SIZE);
        }
        uint64_t t1 = get_us();
        times[iter] = t1 - t0;
    }
    
    st = calc_stats(times, NUM_ITERATIONS);
    print_stats("Uncached NO prefetch:", &st);
    uint64_t uncached_no_prefetch = st.p50;
    
    /* ============================================ */
    /* TEST 4: Uncached WITH cross-core prefetch    */
    /* ============================================ */
    printf("\nTEST 4: Uncached memory, WITH cross-core prefetch (LITTLE->BIG)\n");
    
    prefetch_state_t pstate2 = {
        .should_exit = 0,
        .block_to_prefetch = -1,
        .prefetch_done = 0,
        .ptr = uncached_ptr,
        .block_size = BLOCK_SIZE,
        .lock = PTHREAD_MUTEX_INITIALIZER,
        .cond = PTHREAD_COND_INITIALIZER,
        .cpu = 0  /* LITTLE core */
    };
    
    pthread_create(&prefetch_tid, NULL, prefetch_thread, &pstate2);
    usleep(10000);
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        uint64_t t0 = get_us();
        for (int b = 0; b < NUM_BLOCKS; b++) {
            /* Start prefetch of NEXT block */
            if (b < NUM_BLOCKS - 1) {
                pthread_mutex_lock(&pstate2.lock);
                pstate2.prefetch_done = 0;
                pstate2.block_to_prefetch = b + 1;
                pthread_cond_signal(&pstate2.cond);
                pthread_mutex_unlock(&pstate2.lock);
            }
            
            /* Compute on current block */
            do_compute(uncached_ptr, offsets, NUM_ACCESSES, b * BLOCK_SIZE);
            
            /* Wait for prefetch to complete */
            if (b < NUM_BLOCKS - 1) {
                while (!pstate2.prefetch_done) {
                    __sync_synchronize();
                }
            }
        }
        uint64_t t1 = get_us();
        times[iter] = t1 - t0;
    }
    
    pstate2.should_exit = 1;
    pthread_cond_signal(&pstate2.cond);
    pthread_join(prefetch_tid, NULL);
    
    st = calc_stats(times, NUM_ITERATIONS);
    print_stats("Uncached WITH cross-core prefetch:", &st);
    uint64_t uncached_with_prefetch = st.p50;
    
    printf("  -> Speedup from prefetch: %.2fx\n", (double)uncached_no_prefetch / uncached_with_prefetch);
    
    /* ============================================ */
    /* SUMMARY                                       */
    /* ============================================ */
    printf("\n=== SUMMARY ===\n");
    printf("Testing %d blocks x %d accesses each\n\n", NUM_BLOCKS, NUM_ACCESSES);
    
    printf("CACHED memory:\n");
    printf("  Without prefetch: %lu us (baseline)\n", cached_no_prefetch);
    printf("  With prefetch:    %lu us (%.2fx)\n", cached_with_prefetch, 
           (double)cached_no_prefetch / cached_with_prefetch);
    
    printf("\nUNCACHED memory:\n");
    printf("  Without prefetch: %lu us (baseline)\n", uncached_no_prefetch);
    printf("  With prefetch:    %lu us (%.2fx)\n", uncached_with_prefetch,
           (double)uncached_no_prefetch / uncached_with_prefetch);
    
    printf("\nKEY INSIGHT:\n");
    if ((double)cached_no_prefetch / cached_with_prefetch < 
        (double)uncached_no_prefetch / uncached_with_prefetch) {
        printf("  Uncached gets MORE benefit from cross-core prefetch!\n");
        printf("  This confirms: cache coherency overhead hurts cached cross-core access.\n");
    } else {
        printf("  Cached gets more benefit (unexpected - investigate)\n");
    }
    
    printf("\nBest overall:\n");
    uint64_t best = cached_no_prefetch;
    const char *best_name = "Cached no prefetch";
    if (cached_with_prefetch < best) { best = cached_with_prefetch; best_name = "Cached WITH prefetch"; }
    if (uncached_no_prefetch < best) { best = uncached_no_prefetch; best_name = "Uncached no prefetch"; }
    if (uncached_with_prefetch < best) { best = uncached_with_prefetch; best_name = "Uncached WITH prefetch"; }
    printf("  %s: %lu us\n", best_name, best);
    
    /* Cleanup */
    free(cached_ptr);
    fabric_free(fabric_uncached);
    free(offsets);
    fabric_shutdown();
    
    printf("\n=== Test Complete ===\n");
    return 0;
}
