/*
 * libfabric test harness
 * 
 * Tests fabric allocation and prefetch with KV cache-like access patterns.
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

/* Simulate KV cache access pattern: random reads across the cache */
void simulate_kv_access(float *ptr, size_t size, size_t *offsets, int num_accesses) {
    volatile float sum = 0;
    for (int i = 0; i < num_accesses; i++) {
        sum += ptr[offsets[i] / sizeof(float)];
    }
    (void)sum;
}

int main(int argc, char **argv) {
    printf("=== libfabric Test Harness ===\n\n");
    
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
    
    /* Generate random access pattern */
    srand(42);
    size_t *offsets = (size_t *)malloc(NUM_ACCESSES * sizeof(size_t));
    for (int i = 0; i < NUM_ACCESSES; i++) {
        offsets[i] = (rand() % (KV_CACHE_SIZE / 4096)) * 4096;  /* Page-aligned */
    }
    
    uint64_t times[NUM_SAMPLES];
    stats_t st;
    
    /* ============================================ */
    /* TEST 1: Standard malloc (cached)            */
    /* ============================================ */
    printf("TEST 1: Standard malloc (cached) - %d MB\n", KV_CACHE_SIZE/1024/1024);
    
    float *cached_ptr = (float *)aligned_alloc(4096, KV_CACHE_SIZE);
    memset(cached_ptr, 0, KV_CACHE_SIZE);
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        uint64_t t0 = get_us();
        simulate_kv_access(cached_ptr, KV_CACHE_SIZE, offsets, NUM_ACCESSES);
        uint64_t t1 = get_us();
        times[s] = t1 - t0;
    }
    
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Cached (malloc):", &st);
    uint64_t cached_p50 = st.p50;
    
    free(cached_ptr);
    
    /* ============================================ */
    /* TEST 2: Fabric alloc with RANDOM hint       */
    /* ============================================ */
    printf("\nTEST 2: Fabric alloc (RANDOM hint) - %d MB\n", KV_CACHE_SIZE/1024/1024);
    
    fabric_alloc_t *fabric_random = fabric_alloc(KV_CACHE_SIZE, FABRIC_PATTERN_RANDOM);
    if (!fabric_random) {
        fprintf(stderr, "Failed to allocate with RANDOM pattern\n");
        return 1;
    }
    
    printf("Memory class: %s\n", 
           fabric_get_mem_class(fabric_random) == FABRIC_MEM_UNCACHED ? "UNCACHED" : "CACHED");
    
    float *fabric_ptr = (float *)fabric_get_ptr(fabric_random);
    memset(fabric_ptr, 0, KV_CACHE_SIZE);
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        uint64_t t0 = get_us();
        simulate_kv_access(fabric_ptr, KV_CACHE_SIZE, offsets, NUM_ACCESSES);
        uint64_t t1 = get_us();
        times[s] = t1 - t0;
    }
    
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Fabric (RANDOM/uncached):", &st);
    printf("Speedup vs cached: %.2fx\n", (double)cached_p50 / st.p50);
    uint64_t uncached_p50 = st.p50;
    
    /* ============================================ */
    /* TEST 3: Fabric alloc with prefetch          */
    /* ============================================ */
    printf("\nTEST 3: Fabric with prefetch - %d MB\n", KV_CACHE_SIZE/1024/1024);
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        /* Prefetch ahead */
        fabric_prefetch(fabric_random, 0, KV_CACHE_SIZE);
        
        uint64_t t0 = get_us();
        simulate_kv_access(fabric_ptr, KV_CACHE_SIZE, offsets, NUM_ACCESSES);
        uint64_t t1 = get_us();
        times[s] = t1 - t0;
    }
    
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Fabric + Prefetch:", &st);
    printf("Speedup vs cached: %.2fx\n", (double)cached_p50 / st.p50);
    printf("Speedup vs uncached: %.2fx\n", (double)uncached_p50 / st.p50);
    
    fabric_free(fabric_random);
    
    /* ============================================ */
    /* TEST 4: Fabric alloc with SEQUENTIAL hint   */
    /* ============================================ */
    printf("\nTEST 4: Fabric alloc (SEQUENTIAL hint) - %d MB\n", KV_CACHE_SIZE/1024/1024);
    
    fabric_alloc_t *fabric_seq = fabric_alloc(KV_CACHE_SIZE, FABRIC_PATTERN_SEQUENTIAL);
    if (!fabric_seq) {
        fprintf(stderr, "Failed to allocate with SEQUENTIAL pattern\n");
        return 1;
    }
    
    printf("Memory class: %s\n", 
           fabric_get_mem_class(fabric_seq) == FABRIC_MEM_UNCACHED ? "UNCACHED" : "CACHED");
    
    fabric_ptr = (float *)fabric_get_ptr(fabric_seq);
    memset(fabric_ptr, 0, KV_CACHE_SIZE);
    
    /* Sequential access pattern */
    for (int s = 0; s < NUM_SAMPLES; s++) {
        volatile float sum = 0;
        uint64_t t0 = get_us();
        for (size_t i = 0; i < KV_CACHE_SIZE / sizeof(float); i += 16) {
            sum += fabric_ptr[i];
        }
        uint64_t t1 = get_us();
        times[s] = t1 - t0;
        (void)sum;
    }
    
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Fabric (SEQUENTIAL/cached):", &st);
    
    fabric_free(fabric_seq);
    
    /* ============================================ */
    /* Print statistics                            */
    /* ============================================ */
    printf("\n=== Fabric Statistics ===\n");
    fabric_stats_t stats;
    fabric_get_stats(&stats);
    printf("Cached allocs:   %zu (%zu bytes)\n", stats.cached_allocs, stats.cached_bytes);
    printf("Uncached allocs: %zu (%zu bytes)\n", stats.uncached_allocs, stats.uncached_bytes);
    printf("Prefetch reqs:   %zu (completed: %zu)\n", 
           stats.prefetch_requests, stats.prefetch_completed);
    
    /* Cleanup */
    free(offsets);
    fabric_shutdown();
    
    printf("\n=== Test Complete ===\n");
    return 0;
}
