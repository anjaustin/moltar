/*
 * l3_prefetch_probe.c — Test whether LITTLE-core L3 prefetch helps BIG-core reads
 *
 * Strategy: LITTLE core (A55) issues PRFM PLDL3KEEP ahead of BIG core (A78) reads.
 * The DSU's shared L3 (~1MB) should hold prefetched lines, giving BIG cores L3 hits
 * instead of DRAM misses.
 *
 * Build: $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang \
 *   -O2 -march=armv8.2-a+dotprod -o l3_prefetch_probe l3_prefetch_probe.c -lpthread
 *
 * Run: taskset ff ./l3_prefetch_probe  (needs all cores available)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <malloc.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <arm_neon.h>
#include <unistd.h>

/* Simulated weight matrix sizes matching LFM2-350M matmuls */
#define CHUNK_SMALL  (256 * 1024)   /* 256 KB — like K/V projections */
#define CHUNK_MED    (512 * 1024)   /* 512 KB — like Q/O projections */
#define CHUNK_LARGE  (2400 * 1024)  /* 2.4 MB — like FFN gate/up */
#define NUM_CHUNKS   93             /* matmuls per token */
#define TOTAL_SIZE   (192 * 1024 * 1024)  /* 192 MB total weight data */

static void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

/* Chunk table: offsets into the weight buffer for each "matmul" */
typedef struct {
    size_t offset;
    size_t size;
} chunk_t;

static chunk_t chunks[NUM_CHUNKS];
static char *weight_buf;

/* Synchronization */
static atomic_int current_chunk;    /* BIG core's current chunk index */
static atomic_int prefetch_target;  /* LITTLE core's target chunk */
static volatile int done;

/* ================================================================
 * BIG CORE: Read weight data (simulates GEMV streaming)
 * ================================================================ */
static void big_core_read_chunk(const char *ptr, size_t size, volatile int64_t *sink) {
    const int8_t *p = (const int8_t *)ptr;
    const int8_t *end = p + size;

    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    int64x2_t acc2 = vdupq_n_s64(0);
    int64x2_t acc3 = vdupq_n_s64(0);

    while (p < end) {
        int64x2_t v0 = vld1q_s64((const int64_t *)(p));
        int64x2_t v1 = vld1q_s64((const int64_t *)(p + 16));
        int64x2_t v2 = vld1q_s64((const int64_t *)(p + 32));
        int64x2_t v3 = vld1q_s64((const int64_t *)(p + 48));
        acc0 = vaddq_s64(acc0, v0);
        acc1 = vaddq_s64(acc1, v1);
        acc2 = vaddq_s64(acc2, v2);
        acc3 = vaddq_s64(acc3, v3);
        p += 64;
    }

    int64x2_t sum = vaddq_s64(vaddq_s64(acc0, acc1), vaddq_s64(acc2, acc3));
    *sink = vgetq_lane_s64(sum, 0);
}

/* ================================================================
 * LITTLE CORE: Prefetch weight data to L3
 * ================================================================ */
static void *little_prefetcher(void *arg) {
    int cpu = *(int *)arg;
    pin_to_cpu(cpu);

    while (!done) {
        int target = atomic_load_explicit(&prefetch_target, memory_order_acquire);
        int cur = atomic_load_explicit(&current_chunk, memory_order_acquire);

        /* Prefetch chunks that are ahead of BIG core */
        for (int c = cur + 1; c <= target && c < NUM_CHUNKS; c++) {
            const char *ptr = weight_buf + chunks[c].offset;
            size_t size = chunks[c].size;

            /* Issue PRFM PLDL3KEEP every cache line */
            for (size_t i = 0; i < size; i += 64) {
                __asm__ volatile("prfm pldl3keep, [%0]" : : "r"(ptr + i));
            }
        }

        /* Yield briefly to avoid spinning too hard */
        usleep(5);
    }
    return NULL;
}

/* ================================================================
 * BIG CORE: Process all chunks with optional L3 prefetch
 * ================================================================ */
static double run_big_core(int with_prefetch, int prefetch_ahead) {
    volatile int64_t sink = 0;
    done = 0;
    atomic_store(&current_chunk, 0);
    atomic_store(&prefetch_target, prefetch_ahead);

    pthread_t little_thread;
    int little_cpu = 0;  /* A55 core 0 */

    if (with_prefetch) {
        pthread_create(&little_thread, NULL, little_prefetcher, &little_cpu);
        usleep(1000);  /* Let prefetcher start and get ahead */
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int c = 0; c < NUM_CHUNKS; c++) {
        atomic_store_explicit(&current_chunk, c, memory_order_release);

        /* Update prefetch target: stay N chunks ahead */
        int new_target = c + prefetch_ahead;
        if (new_target >= NUM_CHUNKS) new_target = NUM_CHUNKS - 1;
        atomic_store_explicit(&prefetch_target, new_target, memory_order_release);

        big_core_read_chunk(weight_buf + chunks[c].offset, chunks[c].size, &sink);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    done = 1;
    if (with_prefetch) {
        pthread_join(little_thread, NULL);
    }

    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    return dt;
}

/* ================================================================
 * 2-THREAD BIG CORE: Both A78 cores split the work
 * ================================================================ */
typedef struct {
    int cpu;
    int start_chunk;
    int end_chunk;
    volatile int64_t sink;
} big_thread_arg_t;

static void *big_core_worker(void *arg) {
    big_thread_arg_t *ta = (big_thread_arg_t *)arg;
    pin_to_cpu(ta->cpu);

    for (int c = ta->start_chunk; c < ta->end_chunk; c++) {
        /* For 2-thread mode, each thread reads the full chunk
         * (simulating our GEMV column split where both cores
         * read the same weight data) */
        big_core_read_chunk(weight_buf + chunks[c].offset,
                           chunks[c].size / 2,  /* each core reads half */
                           &ta->sink);

        if (ta->cpu == 6) {
            atomic_store_explicit(&current_chunk, c, memory_order_release);
            int new_target = c + 3;
            if (new_target >= NUM_CHUNKS) new_target = NUM_CHUNKS - 1;
            atomic_store_explicit(&prefetch_target, new_target, memory_order_release);
        }
    }
    return NULL;
}

static double run_two_big_cores(int with_prefetch, int prefetch_ahead) {
    done = 0;
    atomic_store(&current_chunk, 0);
    atomic_store(&prefetch_target, prefetch_ahead);

    pthread_t little_thread;
    int little_cpu = 0;

    if (with_prefetch) {
        pthread_create(&little_thread, NULL, little_prefetcher, &little_cpu);
        usleep(1000);
    }

    big_thread_arg_t args[2] = {
        { .cpu = 6, .start_chunk = 0, .end_chunk = NUM_CHUNKS, .sink = 0 },
        { .cpu = 7, .start_chunk = 0, .end_chunk = NUM_CHUNKS, .sink = 0 },
    };

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    pthread_t t6, t7;
    pthread_create(&t6, NULL, big_core_worker, &args[0]);
    pthread_create(&t7, NULL, big_core_worker, &args[1]);
    pthread_join(t6, NULL);
    pthread_join(t7, NULL);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    done = 1;
    if (with_prefetch) {
        pthread_join(little_thread, NULL);
    }

    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    return dt;
}

int main() {
    /* Allocate weight buffer */
    weight_buf = (char *)memalign(4096, TOTAL_SIZE);
    if (!weight_buf) { printf("alloc failed\n"); return 1; }
    memset(weight_buf, 0x42, TOTAL_SIZE);

    /* Build chunk table — mix of sizes like real LFM2-350M */
    size_t offset = 0;
    for (int c = 0; c < NUM_CHUNKS; c++) {
        size_t sz;
        /* Approximate the real matmul size distribution */
        if (c < 48) {
            /* FFN: gate/up ~2.4MB, down ~2.36MB */
            sz = (c % 3 == 2) ? 2360 * 1024 : 2380 * 1024;
        } else if (c < 68) {
            /* ShortConv: in_proj ~1.59MB, out_proj ~0.53MB */
            sz = (c % 2 == 0) ? 1590 * 1024 : 530 * 1024;
        } else if (c < 92) {
            /* Attention: Q ~0.53MB, K ~0.26MB, V ~0.26MB, O ~0.53MB */
            int sub = (c - 68) % 4;
            if (sub == 0 || sub == 3) sz = 530 * 1024;
            else sz = 260 * 1024;
        } else {
            /* Output logits: ~33.8MB */
            sz = 33816 * 1024;
        }
        /* Clamp to buffer */
        if (offset + sz > TOTAL_SIZE) sz = TOTAL_SIZE - offset;
        chunks[c].offset = offset;
        chunks[c].size = sz;
        offset += sz;
        if (offset >= TOTAL_SIZE) offset = 0;  /* wrap */
    }

    int reps = 5;

    printf("=== L3 Prefetch Probe ===\n");
    printf("Weight buffer: %zu MB, %d chunks\n\n", TOTAL_SIZE / (1024*1024), NUM_CHUNKS);

    /* ---- Test 1: Single BIG core, no prefetch ---- */
    printf("--- 1x A78 (core 6), NO prefetch ---\n");
    pin_to_cpu(6);
    for (int r = 0; r < reps; r++) {
        double dt = run_big_core(0, 0);
        size_t total = 0;
        for (int c = 0; c < NUM_CHUNKS; c++) total += chunks[c].size;
        printf("  rep %d: %.2f ms, %.2f GB/s\n", r, dt*1000, total / dt / 1e9);
    }

    /* ---- Test 2: Single BIG core, WITH L3 prefetch from LITTLE ---- */
    printf("\n--- 1x A78 (core 6), L3 prefetch from A55 (core 0), ahead=2 ---\n");
    pin_to_cpu(6);
    for (int r = 0; r < reps; r++) {
        double dt = run_big_core(1, 2);
        size_t total = 0;
        for (int c = 0; c < NUM_CHUNKS; c++) total += chunks[c].size;
        printf("  rep %d: %.2f ms, %.2f GB/s\n", r, dt*1000, total / dt / 1e9);
    }

    /* ---- Test 3: Single BIG core, L3 prefetch ahead=5 ---- */
    printf("\n--- 1x A78 (core 6), L3 prefetch from A55 (core 0), ahead=5 ---\n");
    pin_to_cpu(6);
    for (int r = 0; r < reps; r++) {
        double dt = run_big_core(1, 5);
        size_t total = 0;
        for (int c = 0; c < NUM_CHUNKS; c++) total += chunks[c].size;
        printf("  rep %d: %.2f ms, %.2f GB/s\n", r, dt*1000, total / dt / 1e9);
    }

    /* ---- Test 4: 2x BIG cores, no prefetch ---- */
    printf("\n--- 2x A78 (cores 6-7), NO prefetch ---\n");
    for (int r = 0; r < reps; r++) {
        double dt = run_two_big_cores(0, 0);
        size_t total = 0;
        for (int c = 0; c < NUM_CHUNKS; c++) total += chunks[c].size;
        printf("  rep %d: %.2f ms, %.2f GB/s\n", r, dt*1000, total / dt / 1e9);
    }

    /* ---- Test 5: 2x BIG cores, WITH L3 prefetch ---- */
    printf("\n--- 2x A78 (cores 6-7), L3 prefetch from A55 (core 0), ahead=3 ---\n");
    for (int r = 0; r < reps; r++) {
        double dt = run_two_big_cores(1, 3);
        size_t total = 0;
        for (int c = 0; c < NUM_CHUNKS; c++) total += chunks[c].size;
        printf("  rep %d: %.2f ms, %.2f GB/s\n", r, dt*1000, total / dt / 1e9);
    }

    printf("\nDone.\n");
    free(weight_buf);
    return 0;
}
