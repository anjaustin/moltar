#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

// Simple streaming read bandwidth test
// Mimics GEMV: sequential reads of weight data
int main(int argc, char **argv) {
    size_t size_mb = 192;  // ~Q4_0 model size
    if (argc > 1) size_mb = atoi(argv[1]);

    size_t size = size_mb * 1024 * 1024;
    char *buf = (char *)aligned_alloc(4096, size);
    if (!buf) { printf("alloc failed\n"); return 1; }

    // Touch all pages
    memset(buf, 0x42, size);

    int reps = 10;
    double best_gbps = 0;

    for (int r = 0; r < reps; r++) {
        // Flush from cache by reading a different large buffer
        // (not perfect but good enough)

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        // Streaming read with NEON — 64 bytes per iteration (1 cache line)
        volatile int64x2_t sink = vdupq_n_s64(0);
        const int8_t *p = (const int8_t *)buf;
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
        sink = vaddq_s64(vaddq_s64(acc0, acc1), vaddq_s64(acc2, acc3));

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        double gbps = (double)size / dt / 1e9;
        if (gbps > best_gbps) best_gbps = gbps;
        printf("rep %d: %.2f GB/s (%.3f ms for %zu MB)\n", r, gbps, dt * 1000, size_mb);
    }
    printf("Best: %.2f GB/s\n", best_gbps);

    free(buf);
    return 0;
}
