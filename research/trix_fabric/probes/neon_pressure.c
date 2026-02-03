/*
 * neon_pressure — NEON/FP pipeline stress generator
 *
 * Saturates the NEON/FP pipeline on specified cores to simulate real-world
 * background load (image processing, ML inference, camera pipeline, etc.).
 *
 * Three pressure modes:
 *   1. NEON-only: fmla/fmul in tight loop (FP pipeline saturation)
 *   2. Memory:    streaming reads through 4MB buffer (cache/bandwidth pressure)
 *   3. Mixed:     alternating NEON compute + memory streaming
 *
 * Usage:
 *   neon_pressure <mode> <duration_sec> [cores]
 *     mode: neon | mem | mixed
 *     cores: bitmask for taskset (default: 3f = little cores 0-5)
 *
 * Cross-compile:
 *   $CC -O2 -o neon_pressure neon_pressure.c -lm -lpthread
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <signal.h>
#include <sys/mman.h>

static volatile int running = 1;

static void sigint_handler(int sig) {
    (void)sig;
    running = 0;
}

/* ── NEON FP pressure: fused multiply-add in tight loop ── */
static void *neon_pressure_thread(void *arg) {
    (void)arg;
    /*
     * This loop does continuous FP multiply-accumulate using NEON.
     * On ARM, the compiler will use fmla instructions for this pattern.
     * 4 independent accumulators to saturate the FP pipeline width.
     */
    float a0 = 1.0001f, a1 = 1.0002f, a2 = 1.0003f, a3 = 1.0004f;
    float b = 0.99999f;
    while (running) {
        for (int i = 0; i < 10000; i++) {
            a0 = a0 * b + 0.0001f;
            a1 = a1 * b + 0.0001f;
            a2 = a2 * b + 0.0001f;
            a3 = a3 * b + 0.0001f;
        }
        /* Prevent dead code elimination */
        if (a0 < -1e30f) running = 0;
    }
    /* Use all accumulators so compiler doesn't optimize them away */
    volatile float sink = a0 + a1 + a2 + a3;
    (void)sink;
    return NULL;
}

/* ── Memory pressure: streaming reads through large buffer ── */
#define MEM_BUF_SIZE (4 * 1024 * 1024)  /* 4 MB — larger than L2 */

static void *mem_pressure_thread(void *arg) {
    (void)arg;
    char *buf = (char *)mmap(NULL, MEM_BUF_SIZE, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) return NULL;

    /* Initialize to prevent zero-page optimization */
    for (size_t i = 0; i < MEM_BUF_SIZE; i += 64) {
        buf[i] = (char)(i & 0xFF);
    }

    volatile long sum = 0;
    while (running) {
        /* Stream through buffer — forces cache line fetches */
        for (size_t i = 0; i < MEM_BUF_SIZE; i += 64) {
            sum += buf[i];
        }
    }
    (void)sum;
    munmap(buf, MEM_BUF_SIZE);
    return NULL;
}

/* ── Mixed pressure: alternating NEON + memory ── */
static void *mixed_pressure_thread(void *arg) {
    (void)arg;
    char *buf = (char *)mmap(NULL, MEM_BUF_SIZE, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) return NULL;

    for (size_t i = 0; i < MEM_BUF_SIZE; i += 64) {
        buf[i] = (char)(i & 0xFF);
    }

    float a0 = 1.0001f, a1 = 1.0002f;
    float b = 0.99999f;
    volatile long sum = 0;
    int phase = 0;

    while (running) {
        if (phase & 1) {
            /* NEON phase */
            for (int i = 0; i < 5000; i++) {
                a0 = a0 * b + 0.0001f;
                a1 = a1 * b + 0.0001f;
            }
            if (a0 < -1e30f) running = 0;
        } else {
            /* Memory phase */
            for (size_t i = 0; i < MEM_BUF_SIZE; i += 64) {
                sum += buf[i];
            }
        }
        phase++;
    }
    (void)sum;
    volatile float sink = a0 + a1;
    (void)sink;
    munmap(buf, MEM_BUF_SIZE);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <neon|mem|mixed> <duration_sec> [n_threads]\n", argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    int duration = atoi(argv[2]);
    int n_threads = (argc > 3) ? atoi(argv[3]) : 4;

    if (duration <= 0 || n_threads <= 0 || n_threads > 16) {
        fprintf(stderr, "Bad args: duration=%d n_threads=%d\n", duration, n_threads);
        return 1;
    }

    void *(*thread_fn)(void *) = NULL;
    if (strcmp(mode, "neon") == 0) {
        thread_fn = neon_pressure_thread;
    } else if (strcmp(mode, "mem") == 0) {
        thread_fn = mem_pressure_thread;
    } else if (strcmp(mode, "mixed") == 0) {
        thread_fn = mixed_pressure_thread;
    } else {
        fprintf(stderr, "Unknown mode: %s (use neon, mem, or mixed)\n", mode);
        return 1;
    }

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    fprintf(stderr, "[pressure] mode=%s threads=%d duration=%ds\n", mode, n_threads, duration);

    pthread_t threads[16];
    for (int i = 0; i < n_threads; i++) {
        pthread_create(&threads[i], NULL, thread_fn, NULL);
    }

    /* Run for specified duration then stop */
    sleep(duration);
    running = 0;

    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    fprintf(stderr, "[pressure] done\n");
    return 0;
}
