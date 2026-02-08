#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <malloc.h>
#include <pthread.h>
#include <arm_neon.h>

typedef struct {
    const char *buf;
    size_t size;
    volatile int64_t sink_val;
} thread_arg_t;

static void *stream_read(void *arg) {
    thread_arg_t *ta = (thread_arg_t *)arg;
    const int8_t *p = (const int8_t *)ta->buf;
    const int8_t *end = p + ta->size;

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
    ta->sink_val = vgetq_lane_s64(sum, 0);
    return NULL;
}

// Also test with multiple outstanding loads per iteration (deeper pipeline)
static void *stream_read_deep(void *arg) {
    thread_arg_t *ta = (thread_arg_t *)arg;
    const int8_t *p = (const int8_t *)ta->buf;
    const int8_t *end = p + ta->size;

    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    int64x2_t acc2 = vdupq_n_s64(0);
    int64x2_t acc3 = vdupq_n_s64(0);
    int64x2_t acc4 = vdupq_n_s64(0);
    int64x2_t acc5 = vdupq_n_s64(0);
    int64x2_t acc6 = vdupq_n_s64(0);
    int64x2_t acc7 = vdupq_n_s64(0);

    // 8 loads per iteration = 4 cache lines = 256 bytes
    while (p + 256 <= end) {
        int64x2_t v0 = vld1q_s64((const int64_t *)(p));
        int64x2_t v1 = vld1q_s64((const int64_t *)(p + 16));
        int64x2_t v2 = vld1q_s64((const int64_t *)(p + 32));
        int64x2_t v3 = vld1q_s64((const int64_t *)(p + 48));
        int64x2_t v4 = vld1q_s64((const int64_t *)(p + 64));
        int64x2_t v5 = vld1q_s64((const int64_t *)(p + 80));
        int64x2_t v6 = vld1q_s64((const int64_t *)(p + 96));
        int64x2_t v7 = vld1q_s64((const int64_t *)(p + 112));
        acc0 = vaddq_s64(acc0, v0);
        acc1 = vaddq_s64(acc1, v1);
        acc2 = vaddq_s64(acc2, v2);
        acc3 = vaddq_s64(acc3, v3);
        acc4 = vaddq_s64(acc4, v4);
        acc5 = vaddq_s64(acc5, v5);
        acc6 = vaddq_s64(acc6, v6);
        acc7 = vaddq_s64(acc7, v7);

        int64x2_t v8  = vld1q_s64((const int64_t *)(p + 128));
        int64x2_t v9  = vld1q_s64((const int64_t *)(p + 144));
        int64x2_t v10 = vld1q_s64((const int64_t *)(p + 160));
        int64x2_t v11 = vld1q_s64((const int64_t *)(p + 176));
        int64x2_t v12 = vld1q_s64((const int64_t *)(p + 192));
        int64x2_t v13 = vld1q_s64((const int64_t *)(p + 208));
        int64x2_t v14 = vld1q_s64((const int64_t *)(p + 224));
        int64x2_t v15 = vld1q_s64((const int64_t *)(p + 240));
        acc0 = vaddq_s64(acc0, v8);
        acc1 = vaddq_s64(acc1, v9);
        acc2 = vaddq_s64(acc2, v10);
        acc3 = vaddq_s64(acc3, v11);
        acc4 = vaddq_s64(acc4, v12);
        acc5 = vaddq_s64(acc5, v13);
        acc6 = vaddq_s64(acc6, v14);
        acc7 = vaddq_s64(acc7, v15);
        p += 256;
    }
    // Handle remainder
    while (p < end) {
        int64x2_t v0 = vld1q_s64((const int64_t *)(p));
        acc0 = vaddq_s64(acc0, v0);
        p += 16;
    }

    int64x2_t s01 = vaddq_s64(acc0, acc1);
    int64x2_t s23 = vaddq_s64(acc2, acc3);
    int64x2_t s45 = vaddq_s64(acc4, acc5);
    int64x2_t s67 = vaddq_s64(acc6, acc7);
    int64x2_t sum = vaddq_s64(vaddq_s64(s01, s23), vaddq_s64(s45, s67));
    ta->sink_val = vgetq_lane_s64(sum, 0);
    return NULL;
}

static double run_test(char *buf, size_t total_size, int nthreads, int deep) {
    pthread_t threads[8];
    thread_arg_t args[8];

    size_t per_thread = total_size / nthreads;
    // Align to 256 bytes
    per_thread = (per_thread / 256) * 256;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < nthreads; i++) {
        args[i].buf = buf + i * per_thread;
        args[i].size = per_thread;
        args[i].sink_val = 0;
        pthread_create(&threads[i], NULL, deep ? stream_read_deep : stream_read, &args[i]);
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double total_bytes = (double)per_thread * nthreads;
    return total_bytes / dt / 1e9;
}

int main(int argc, char **argv) {
    size_t size_mb = 192;
    if (argc > 1) size_mb = atoi(argv[1]);

    size_t size = size_mb * 1024 * 1024;
    char *buf = (char *)memalign(4096, size);
    if (!buf) { printf("alloc failed\n"); return 1; }
    memset(buf, 0x42, size);

    int reps = 5;

    printf("=== Memory Bandwidth Test (%zu MB) ===\n\n", size_mb);

    // Test 1: single thread, normal
    printf("--- 1 thread, 1 cache-line/iter ---\n");
    for (int r = 0; r < reps; r++) {
        double gbps = run_test(buf, size, 1, 0);
        printf("  rep %d: %.2f GB/s\n", r, gbps);
    }

    // Test 2: single thread, deep pipeline
    printf("\n--- 1 thread, 4 cache-lines/iter (deep) ---\n");
    for (int r = 0; r < reps; r++) {
        double gbps = run_test(buf, size, 1, 1);
        printf("  rep %d: %.2f GB/s\n", r, gbps);
    }

    // Test 3: 2 threads, normal
    printf("\n--- 2 threads, 1 cache-line/iter ---\n");
    for (int r = 0; r < reps; r++) {
        double gbps = run_test(buf, size, 2, 0);
        printf("  rep %d: %.2f GB/s\n", r, gbps);
    }

    // Test 4: 2 threads, deep pipeline
    printf("\n--- 2 threads, 4 cache-lines/iter (deep) ---\n");
    for (int r = 0; r < reps; r++) {
        double gbps = run_test(buf, size, 2, 1);
        printf("  rep %d: %.2f GB/s\n", r, gbps);
    }

    printf("\nDone.\n");
    free(buf);
    return 0;
}
