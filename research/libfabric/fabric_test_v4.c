/*
 * libfabric test harness v4
 * 
 * Simpler test: just measure if touching data from LITTLE core 
 * helps subsequent BIG core access.
 * 
 * No synchronization overhead - just sequential: prefetch, then access
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

#define TEST_SIZE (16 * 1024 * 1024)  /* 16MB */
#define NUM_ACCESSES 5000
#define NUM_SAMPLES 30
#define PAGE_SIZE 4096

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

/* Prefetch helper - touches each cache line */
void prefetch_region(volatile char *ptr, size_t size) {
    volatile char sum = 0;
    for (size_t i = 0; i < size; i += 64) {
        sum += ptr[i];
    }
    __sync_synchronize();
}

typedef struct {
    volatile char *ptr;
    size_t size;
    volatile int done;
    int cpu;
} prefetch_job_t;

void *prefetch_worker(void *arg) {
    prefetch_job_t *job = (prefetch_job_t *)arg;
    
#ifdef __ANDROID__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(job->cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
#endif
    
    prefetch_region(job->ptr, job->size);
    job->done = 1;
    return NULL;
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
    printf("%-45s p50=%6lu us  p90=%6lu us  p99=%6lu us  CV=%5.1f%%\n",
           name, st->p50, st->p90, st->p99, st->cv);
}

/* Random access workload */
volatile float random_access(float *ptr, size_t *offsets, int n) {
    volatile float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += ptr[offsets[i] / sizeof(float)];
    }
    return sum;
}

int main(int argc, char **argv) {
    printf("=== libfabric Test Harness v4 ===\n");
    printf("Testing: Does pre-touching from LITTLE core help BIG core access?\n\n");
    
#ifdef __ANDROID__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(7, &set);
    sched_setaffinity(0, sizeof(set), &set);
    printf("Main thread pinned to CPU 7 (big core)\n");
#endif
    
    if (fabric_init(0) != 0) {
        fprintf(stderr, "Failed to initialize fabric\n");
        return 1;
    }
    
    printf("Uncached available: %s\n", fabric_uncached_available() ? "YES" : "NO");
    printf("Test size: %d MB, %d random accesses per sample\n\n", 
           TEST_SIZE / (1024*1024), NUM_ACCESSES);
    
    /* Random offsets */
    srand(42);
    size_t *offsets = (size_t *)malloc(NUM_ACCESSES * sizeof(size_t));
    for (int i = 0; i < NUM_ACCESSES; i++) {
        offsets[i] = (rand() % (TEST_SIZE / sizeof(float))) * sizeof(float);
    }
    
    uint64_t times[NUM_SAMPLES];
    stats_t st;
    
    /* Allocate memory */
    float *cached_ptr = (float *)aligned_alloc(PAGE_SIZE, TEST_SIZE);
    memset(cached_ptr, 0, TEST_SIZE);
    
    fabric_alloc_t *fabric_uncached = fabric_alloc(TEST_SIZE, FABRIC_PATTERN_RANDOM);
    float *uncached_ptr = (float *)fabric_get_ptr(fabric_uncached);
    memset(uncached_ptr, 0, TEST_SIZE);
    
    printf("Cached:   %p\n", cached_ptr);
    printf("Uncached: %p (%s)\n\n", uncached_ptr,
           fabric_get_mem_class(fabric_uncached) == FABRIC_MEM_UNCACHED ? "UNCACHED" : "CACHED");
    
    /* ============================================ */
    /* TEST 1a: Cached, cold (flush, then access)  */
    /* ============================================ */
    printf("TEST 1a: Cached COLD (flush then access from BIG core)\n");
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        flush_region(cached_ptr, TEST_SIZE);
        
        uint64_t t0 = get_us();
        random_access(cached_ptr, offsets, NUM_ACCESSES);
        times[s] = get_us() - t0;
    }
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Cached COLD (BIG core):", &st);
    uint64_t cached_cold = st.p50;
    
    /* ============================================ */
    /* TEST 1b: Cached, pre-touch from BIG core    */
    /* ============================================ */
    printf("\nTEST 1b: Cached + pre-touch from SAME core (BIG)\n");
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        flush_region(cached_ptr, TEST_SIZE);
        
        /* Pre-touch from BIG core (same core) */
        prefetch_region((volatile char *)cached_ptr, TEST_SIZE);
        
        uint64_t t0 = get_us();
        random_access(cached_ptr, offsets, NUM_ACCESSES);
        times[s] = get_us() - t0;
    }
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Cached + BIG pretouch:", &st);
    printf("  Speedup vs cold: %.2fx\n", (double)cached_cold / st.p50);
    uint64_t cached_big_pretouch = st.p50;
    
    /* ============================================ */
    /* TEST 1c: Cached, pre-touch from LITTLE core */
    /* ============================================ */
    printf("\nTEST 1c: Cached + pre-touch from DIFFERENT core (LITTLE)\n");
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        flush_region(cached_ptr, TEST_SIZE);
        
        /* Pre-touch from LITTLE core */
        prefetch_job_t job = { 
            .ptr = (volatile char *)cached_ptr, 
            .size = TEST_SIZE, 
            .done = 0,
            .cpu = 0  /* LITTLE core */
        };
        pthread_t tid;
        pthread_create(&tid, NULL, prefetch_worker, &job);
        pthread_join(tid, NULL);  /* Wait for prefetch to complete */
        
        uint64_t t0 = get_us();
        random_access(cached_ptr, offsets, NUM_ACCESSES);
        times[s] = get_us() - t0;
    }
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Cached + LITTLE pretouch:", &st);
    printf("  Speedup vs cold: %.2fx\n", (double)cached_cold / st.p50);
    uint64_t cached_little_pretouch = st.p50;
    
    /* ============================================ */
    /* TEST 2a: Uncached, cold                      */
    /* ============================================ */
    printf("\nTEST 2a: Uncached COLD (no flush needed)\n");
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        uint64_t t0 = get_us();
        random_access(uncached_ptr, offsets, NUM_ACCESSES);
        times[s] = get_us() - t0;
    }
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Uncached COLD (BIG core):", &st);
    uint64_t uncached_cold = st.p50;
    
    /* ============================================ */
    /* TEST 2b: Uncached, pre-touch from BIG core  */
    /* ============================================ */
    printf("\nTEST 2b: Uncached + pre-touch from SAME core (BIG)\n");
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        /* Pre-touch from BIG core */
        prefetch_region((volatile char *)uncached_ptr, TEST_SIZE);
        
        uint64_t t0 = get_us();
        random_access(uncached_ptr, offsets, NUM_ACCESSES);
        times[s] = get_us() - t0;
    }
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Uncached + BIG pretouch:", &st);
    printf("  Speedup vs cold: %.2fx\n", (double)uncached_cold / st.p50);
    
    /* ============================================ */
    /* TEST 2c: Uncached, pre-touch from LITTLE    */
    /* ============================================ */
    printf("\nTEST 2c: Uncached + pre-touch from DIFFERENT core (LITTLE)\n");
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        /* Pre-touch from LITTLE core */
        prefetch_job_t job = { 
            .ptr = (volatile char *)uncached_ptr, 
            .size = TEST_SIZE, 
            .done = 0,
            .cpu = 0  /* LITTLE core */
        };
        pthread_t tid;
        pthread_create(&tid, NULL, prefetch_worker, &job);
        pthread_join(tid, NULL);
        
        uint64_t t0 = get_us();
        random_access(uncached_ptr, offsets, NUM_ACCESSES);
        times[s] = get_us() - t0;
    }
    st = calc_stats(times, NUM_SAMPLES);
    print_stats("Uncached + LITTLE pretouch:", &st);
    printf("  Speedup vs cold: %.2fx\n", (double)uncached_cold / st.p50);
    
    /* ============================================ */
    /* SUMMARY                                       */
    /* ============================================ */
    printf("\n=== SUMMARY ===\n");
    printf("Question: Does pre-touching memory help subsequent access?\n\n");
    
    printf("CACHED memory:\n");
    printf("  Cold:              %lu us\n", cached_cold);
    printf("  BIG pretouch:      %lu us (%.2fx vs cold)\n", 
           cached_big_pretouch, (double)cached_cold / cached_big_pretouch);
    printf("  LITTLE pretouch:   %lu us (%.2fx vs cold)\n", 
           cached_little_pretouch, (double)cached_cold / cached_little_pretouch);
    
    printf("\nUNCACHED memory:\n");
    printf("  Cold:              %lu us\n", uncached_cold);
    printf("  (Pre-touch doesn't cache, so should be ~same)\n");
    
    printf("\n=== KEY INSIGHT ===\n");
    if (cached_little_pretouch > cached_big_pretouch * 1.1) {
        printf("LITTLE pretouch is SLOWER than BIG pretouch for cached memory.\n");
        printf("This confirms cache coherency overhead when crossing cores.\n");
    } else {
        printf("Cross-core pretouch works well for cached memory.\n");
    }
    
    /* Cleanup */
    free(cached_ptr);
    fabric_free(fabric_uncached);
    free(offsets);
    fabric_shutdown();
    
    printf("\n=== Test Complete ===\n");
    return 0;
}
