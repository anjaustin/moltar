/*
 * libfabric test harness v2
 * 
 * More careful testing with cache flushes between tests
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

#ifdef __ANDROID__
#include <sched.h>
#endif

#define KV_CACHE_SIZE (64 * 1024 * 1024)  /* 64MB simulated KV cache */
#define NUM_ACCESSES 10000
#define NUM_SAMPLES 100
#define PAGE_SIZE 4096

/* Cache flush for ARM */
static inline void dc_civac(void *addr) {
    __asm__ __volatile__("dc civac, %0" : : "r"(addr) : "memory");
}

static inline void dsb_sy(void) {
    __asm__ __volatile__("dsb sy" ::: "memory");
}

/* Flush entire region from cache */
void flush_region(void *ptr, size_t size) {
    char *p = (char *)ptr;
    for (size_t i = 0; i < size; i += 64) {
        dc_civac(p + i);
    }
    dsb_sy();
}

static inline uint64_t get_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
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
    printf("%-35s p50=%6lu us  p90=%6lu us  p99=%6lu us  CV=%5.1f%%\n",
           name, st->p50, st->p90, st->p99, st->cv);
}

int main(int argc, char **argv) {
    printf("=== libfabric Test Harness v2 ===\n");
    printf("Testing with cache flushes between samples\n\n");
    
#ifdef __ANDROID__
    /* Pin to BIG core */
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(7, &set);
    sched_setaffinity(0, sizeof(set), &set);
    printf("Pinned to CPU 7 (big core)\n");
#endif
    
    /* Initialize fabric */
    printf("Initializing fabric with 2 prefetch threads...\n");
    if (fabric_init(2) != 0) {
        fprintf(stderr, "Failed to initialize fabric\n");
        return 1;
    }
    
    printf("Uncached memory available: %s\n\n", 
           fabric_uncached_available() ? "YES" : "NO");
    
    /* Generate random access pattern - random offsets, not page-aligned */
    srand(42);
    size_t *offsets = (size_t *)malloc(NUM_ACCESSES * sizeof(size_t));
    for (int i = 0; i < NUM_ACCESSES; i++) {
        /* Random byte offset, aligned to 4 bytes (float) */
        offsets[i] = (rand() % (KV_CACHE_SIZE / sizeof(float))) * sizeof(float);
    }
    
    uint64_t times[NUM_SAMPLES];
    stats_t st;
    volatile float sum;
    
    /* Allocate regions */
    float *cached_ptr = (float *)aligned_alloc(PAGE_SIZE, KV_CACHE_SIZE);
    memset(cached_ptr, 0, KV_CACHE_SIZE);
    
    fabric_alloc_t *fabric_uncached = fabric_alloc(KV_CACHE_SIZE, FABRIC_PATTERN_RANDOM);
    float *uncached_ptr = (float *)fabric_get_ptr(fabric_uncached);
    memset(uncached_ptr, 0, KV_CACHE_SIZE);
    
    printf("Cached ptr: %p\n", cached_ptr);
    printf("Uncached ptr: %p (class: %s)\n\n", uncached_ptr,
           fabric_get_mem_class(fabric_uncached) == FABRIC_MEM_UNCACHED ? "UNCACHED" : "CACHED");
    
    /* ============================================ */
    /* TEST 1: Cached, COLD (flush before each)    */
    /* ============================================ */
    printf("TEST 1: Cached memory, COLD (flushed before each sample)\n");
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        /* Flush cache */
        flush_region(cached_ptr, KV_CACHE_SIZE);
        
        sum = 0;
        uint64_t t0 = get_us();
        for (int i = 0; i < NUM_ACCESSES; i++) {
            sum += cached_ptr[offsets[i] / sizeof(float)];
        }
        uint64_t t1 = get_us();
        times[s] = t1 - t0;
    }
    
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Cached COLD:", &st);
    uint64_t cached_cold_p50 = st.p50;
    
    /* ============================================ */
    /* TEST 2: Cached, WARM (no flush)             */
    /* ============================================ */
    printf("\nTEST 2: Cached memory, WARM (no flush - may hit cache)\n");
    
    /* Warm up */
    for (int i = 0; i < NUM_ACCESSES; i++) {
        sum += cached_ptr[offsets[i] / sizeof(float)];
    }
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        sum = 0;
        uint64_t t0 = get_us();
        for (int i = 0; i < NUM_ACCESSES; i++) {
            sum += cached_ptr[offsets[i] / sizeof(float)];
        }
        uint64_t t1 = get_us();
        times[s] = t1 - t0;
    }
    
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Cached WARM:", &st);
    uint64_t cached_warm_p50 = st.p50;
    
    /* ============================================ */
    /* TEST 3: Uncached (no flush needed)          */
    /* ============================================ */
    printf("\nTEST 3: Uncached memory (bypasses cache)\n");
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        sum = 0;
        uint64_t t0 = get_us();
        for (int i = 0; i < NUM_ACCESSES; i++) {
            sum += uncached_ptr[offsets[i] / sizeof(float)];
        }
        uint64_t t1 = get_us();
        times[s] = t1 - t0;
    }
    
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Uncached:", &st);
    uint64_t uncached_p50 = st.p50;
    
    printf("\nSpeedup vs Cached COLD: %.2fx\n", (double)cached_cold_p50 / uncached_p50);
    printf("Speedup vs Cached WARM: %.2fx\n", (double)cached_warm_p50 / uncached_p50);
    
    /* ============================================ */
    /* TEST 4: Uncached with prefetch              */
    /* ============================================ */
    printf("\nTEST 4: Uncached with prefetch thread\n");
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        /* Prefetch ahead */
        fabric_prefetch(fabric_uncached, 0, KV_CACHE_SIZE);
        
        sum = 0;
        uint64_t t0 = get_us();
        for (int i = 0; i < NUM_ACCESSES; i++) {
            sum += uncached_ptr[offsets[i] / sizeof(float)];
        }
        uint64_t t1 = get_us();
        times[s] = t1 - t0;
    }
    
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Uncached + Prefetch:", &st);
    printf("Speedup vs uncached alone: %.2fx\n", (double)uncached_p50 / st.p50);
    
    /* ============================================ */
    /* SUMMARY                                      */
    /* ============================================ */
    printf("\n=== SUMMARY ===\n");
    printf("For KV cache (64MB, random access):\n");
    printf("- Cached COLD (realistic):  ~%lu us per %d accesses\n", cached_cold_p50, NUM_ACCESSES);
    printf("- Cached WARM (best case):  ~%lu us per %d accesses\n", cached_warm_p50, NUM_ACCESSES);
    printf("- Uncached:                 ~%lu us per %d accesses\n", uncached_p50, NUM_ACCESSES);
    printf("\nFor LLM inference, KV cache is accessed COLD each layer.\n");
    printf("Compare Cached COLD vs Uncached for realistic assessment.\n");
    
    /* Cleanup */
    free(cached_ptr);
    fabric_free(fabric_uncached);
    free(offsets);
    fabric_shutdown();
    
    printf("\n=== Test Complete ===\n");
    return 0;
}
