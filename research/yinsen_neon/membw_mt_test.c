/*
 * membw_mt_test.c - Multi-threaded memory bandwidth test
 *
 * Tests if we can achieve higher bandwidth with multiple cores.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

typedef struct {
    const int8_t* data;
    int64_t size;
    int iters;
} thread_args_t;

static void* read_thread(void* arg) {
    thread_args_t* a = (thread_args_t*)arg;
    
    for (int iter = 0; iter < a->iters; iter++) {
        int8x16_t acc = vdupq_n_s8(0);
        const int8_t* ptr = a->data;
        const int8_t* end = a->data + a->size;
        
        while (ptr < end) {
            int8x16_t v0 = vld1q_s8(ptr);
            int8x16_t v1 = vld1q_s8(ptr + 16);
            int8x16_t v2 = vld1q_s8(ptr + 32);
            int8x16_t v3 = vld1q_s8(ptr + 48);
            ptr += 64;
            
            acc = vaddq_s8(acc, v0);
            acc = vaddq_s8(acc, v1);
            acc = vaddq_s8(acc, v2);
            acc = vaddq_s8(acc, v3);
        }
        
        volatile int8_t sink = vgetq_lane_s8(acc, 0);
        (void)sink;
    }
    
    return NULL;
}

int main(int argc, char** argv) {
    int64_t size_mb = 16;
    int iters = 10;
    int num_threads = 2;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            size_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        }
    }
    
    int64_t total_size = size_mb * 1024 * 1024;
    int64_t per_thread = total_size / num_threads;
    
    printf("=== Multi-threaded Memory Bandwidth Test ===\n");
    printf("Total buffer: %lld MB\n", (long long)size_mb);
    printf("Threads: %d\n", num_threads);
    printf("Per thread: %lld MB\n", (long long)(per_thread / (1024*1024)));
    printf("Iterations: %d\n", iters);
    printf("\n");
    
    /* Allocate and initialize */
    int8_t* data = (int8_t*)malloc(total_size);
    if (!data) {
        printf("ERROR: Failed to allocate\n");
        return 1;
    }
    memset(data, 1, total_size);
    
    /* Warmup */
    thread_args_t warmup_args = {data, per_thread, 1};
    read_thread(&warmup_args);
    
    /* Setup thread args */
    pthread_t threads[8];
    thread_args_t args[8];
    
    for (int t = 0; t < num_threads; t++) {
        args[t].data = data + t * per_thread;
        args[t].size = per_thread;
        args[t].iters = iters;
    }
    
    /* Time parallel reads */
    uint64_t t0 = get_time_ns();
    
    /* Start worker threads */
    for (int t = 1; t < num_threads; t++) {
        pthread_create(&threads[t], NULL, read_thread, &args[t]);
    }
    
    /* Main thread does first chunk */
    read_thread(&args[0]);
    
    /* Join */
    for (int t = 1; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    
    uint64_t t1 = get_time_ns();
    
    double ns = (double)(t1 - t0) / iters;
    double gbps = (double)total_size / ns;
    
    printf("Total bandwidth (%d threads): %.2f GB/s\n", num_threads, gbps);
    printf("Per-thread: %.2f GB/s\n", gbps / num_threads);
    printf("\n");
    
    free(data);
    
    return 0;
}
