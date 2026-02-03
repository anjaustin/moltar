/*
 * raid0_big_probe.c — RAID 0 Striped MatVec on Big Cores (cpu6+7)
 *
 * Phase 1A: Confirms 2-way striped matvec throughput ceiling.
 * Baseline: single-core Kernel B = 762us on A78 (matvec_shootout)
 * Expected: ~380us with 2-way striping
 *
 * $CC -O2 -march=armv8.2-a+dotprod -o raid0_big_probe raid0_big_probe.c -lm -lpthread
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ── Timing ── */
static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ── Minimal GGUF Parser ── */
#define GGUF_MAGIC 0x46554747
#define GGUF_TYPE_Q4_0 2

typedef struct { char name[256]; uint32_t n_dims; uint64_t dims[4]; uint32_t type; uint64_t offset; } TensorInfo;

static const uint8_t *skip_str(const uint8_t *p) { return p + 8 + *(const uint64_t *)p; }
static void read_str(const uint8_t *p, char *o, int m) {
    uint64_t l = *(const uint64_t *)p; int c = l < (uint64_t)m-1 ? (int)l : m-1;
    memcpy(o, p+8, c); o[c] = 0;
}
static const uint8_t *skip_val(const uint8_t *p, uint32_t t) {
    switch(t) {
        case 0:case 1:case 7: return p+1; case 2:case 3: return p+2;
        case 4:case 5:case 6: return p+4; case 10:case 11:case 12: return p+8;
        case 8: return skip_str(p);
        case 9: { uint32_t et=*(const uint32_t*)p; p+=4; uint64_t n=*(const uint64_t*)p; p+=8;
                  for(uint64_t i=0;i<n;i++) p=skip_val(p,et); return p; }
        default: return p;
    }
}

static int find_tensor(const uint8_t *data, const char *name, TensorInfo *out, const uint8_t **tensor_data) {
    const uint8_t *p = data + 4; /* skip magic */
    p += 4; /* version */
    uint64_t n_tensors = *(const uint64_t *)p; p += 8;
    uint64_t n_kv = *(const uint64_t *)p; p += 8;
    for (uint64_t i = 0; i < n_kv; i++) { p = skip_str(p); uint32_t vt = *(const uint32_t *)p; p += 4; p = skip_val(p, vt); }

    TensorInfo *all = calloc(n_tensors, sizeof(TensorInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        read_str(p, all[i].name, 256); p = skip_str(p);
        all[i].n_dims = *(const uint32_t *)p; p += 4;
        for (uint32_t d = 0; d < all[i].n_dims; d++) { all[i].dims[d] = *(const uint64_t *)p; p += 8; }
        all[i].type = *(const uint32_t *)p; p += 4;
        all[i].offset = *(const uint64_t *)p; p += 8;
    }
    uint64_t base = ((p - data) + 31) & ~31ULL;
    int found = -1;
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (strcmp(all[i].name, name) == 0 && all[i].type == GGUF_TYPE_Q4_0) { *out = all[i]; found = 0; break; }
    }
    if (found == 0) *tensor_data = data + base + out->offset;
    free(all);
    return found;
}

/* ── f16 → f32 ── */
static float f16_to_f32(uint16_t h) {
    uint32_t s=(h&0x8000)<<16, e=(h>>10)&0x1F, m=h&0x3FF, f;
    if(!e){if(!m)f=s;else{e=1;while(!(m&0x400)){m<<=1;e--;}m&=0x3FF;f=s|((e+112)<<23)|(m<<13);}}
    else if(e==31)f=s|0x7F800000|(m<<13); else f=s|((e+112)<<23)|(m<<13);
    float r; memcpy(&r,&f,4); return r;
}

/* ── Q4_0 NEON SDOT kernel ── */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void matvec_q4_neon(float *out, const uint8_t *q4, const int8_t *act, int N, int K) {
    const int bpr = K/32, bs = 18;
    const int8x16_t eight = vdupq_n_s8(8);
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        const uint8_t *row = q4 + (size_t)n * bpr * bs;
        for (int b = 0; b < bpr; b++) {
            uint16_t sh; memcpy(&sh, row, 2); float sc = f16_to_f32(sh);
            uint8x16_t raw = vld1q_u8(row + 2);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
            int8x16x2_t ap = vld2q_s8(act + b * 32);
            int32x4_t a0 = vdotq_s32(vdupq_n_s32(0), lo, ap.val[0]);
            int32x4_t a1 = vdotq_s32(vdupq_n_s32(0), hi, ap.val[1]);
            sum += sc * (float)(vaddvq_s32(a0) + vaddvq_s32(a1)) / 64.0f;
            row += bs;
        }
        out[n] = sum;
    }
}
#endif

/* ── Stats ── */
static int cmpd(const void *a, const void *b) { double d=*(const double*)a-*(const double*)b; return (d>0)-(d<0); }
static void stats(const char *label, double *t, int n) {
    qsort(t, n, sizeof(double), cmpd);
    double s=0; for(int i=0;i<n;i++) s+=t[i];
    printf("  %-38s mean=%7.1f  min=%7.1f  p50=%7.1f  p99=%7.1f  max=%7.1f us\n",
           label, s/n, t[0], t[n/2], t[(int)(n*0.99)], t[n-1]);
}

/* ── RAID 0 worker ── */
typedef struct {
    int cpu_id; float *out; const uint8_t *wgt; const int8_t *act;
    int N, K, n_iters;
    pthread_barrier_t *bar_start, *bar_done;
    double *times;
} WorkerArg;

static void *worker(void *a) {
    WorkerArg *w = (WorkerArg *)a;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(w->cpu_id, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
    for (int i = 0; i < w->n_iters; i++) {
        pthread_barrier_wait(w->bar_start);
        double t0 = now_us();
        matvec_q4_neon(w->out, w->wgt, w->act, w->N, w->K);
        w->times[i] = now_us() - t0;
        pthread_barrier_wait(w->bar_done);
    }
    return NULL;
}

/* ── Main ── */
int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    TensorInfo ti; const uint8_t *wgt;
    if (find_tensor(data, "blk.0.ffn_gate.weight", &ti, &wgt) != 0) { fprintf(stderr, "tensor not found\n"); return 1; }
    int K = (int)ti.dims[0], N = (int)ti.dims[1];
    size_t row_bytes = (size_t)(K/32) * 18;

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  PROBE 1A: RAID 0 on Big Cores (cpu6+7)\n");
    printf("  Tensor: %s [N=%d, K=%d] = %.2f MiB\n", ti.name, N, K, (double)N*row_bytes/(1024*1024));
    printf("  Stripe: 2 × %d rows = %.1f KiB each\n", N/2, (double)(N/2)*row_bytes/1024);
    printf("════════════════════════════════════════════════════════════\n\n");

    int8_t *act = calloc(K, 1); float *out = calloc(N, sizeof(float));
    srand(42); for (int i=0;i<K;i++) act[i]=(int8_t)(((float)rand()/RAND_MAX-0.5f)*128);

    int ITERS = 1000;

    /* ── Single core baseline on cpu6 ── */
    { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(6, &cs); sched_setaffinity(0, sizeof(cs), &cs); }
    for (int i=0;i<20;i++) matvec_q4_neon(out, wgt, act, N, K); /* warmup */
    double *t1 = calloc(ITERS, sizeof(double));
    for (int i=0;i<ITERS;i++) { double t=now_us(); matvec_q4_neon(out, wgt, act, N, K); t1[i]=now_us()-t; }
    stats("Single core (cpu6, N=4608)", t1, ITERS);

    /* ── Single core baseline on cpu0 (A55) ── */
    { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(0, &cs); sched_setaffinity(0, sizeof(cs), &cs); }
    for (int i=0;i<10;i++) matvec_q4_neon(out, wgt, act, N, K);
    double *t1a = calloc(ITERS, sizeof(double));
    for (int i=0;i<ITERS;i++) { double t=now_us(); matvec_q4_neon(out, wgt, act, N, K); t1a[i]=now_us()-t; }
    stats("Single core (cpu0/A55, N=4608)", t1a, ITERS);

    /* ── RAID 0: 2-way on big cores ── */
    printf("\n");
    pthread_barrier_t bs2, bd2;
    pthread_barrier_init(&bs2, NULL, 3); pthread_barrier_init(&bd2, NULL, 3);
    double *tt0 = calloc(ITERS, sizeof(double)), *tt1 = calloc(ITERS, sizeof(double)), *tw = calloc(ITERS, sizeof(double));
    int hN = N/2;
    WorkerArg a0 = {6, out, wgt, act, hN, K, ITERS, &bs2, &bd2, tt0};
    WorkerArg a1 = {7, out+hN, wgt+(size_t)hN*row_bytes, act, hN, K, ITERS, &bs2, &bd2, tt1};
    pthread_t th[2];
    pthread_create(&th[0], NULL, worker, &a0);
    pthread_create(&th[1], NULL, worker, &a1);
    for (int i=0;i<ITERS;i++) {
        double t=now_us();
        pthread_barrier_wait(&bs2);
        pthread_barrier_wait(&bd2);
        tw[i]=now_us()-t;
    }
    pthread_join(th[0], NULL); pthread_join(th[1], NULL);
    stats("RAID 0 thread 0 (cpu6, N/2)", tt0, ITERS);
    stats("RAID 0 thread 1 (cpu7, N/2)", tt1, ITERS);
    stats("RAID 0 wall clock (2-way)", tw, ITERS);

    /* Summary */
    qsort(t1, ITERS, sizeof(double), cmpd);
    qsort(tw, ITERS, sizeof(double), cmpd);
    double s1=0, sw=0; for(int i=0;i<ITERS;i++){s1+=t1[i];sw+=tw[i];}
    printf("\n  ── Summary ──\n");
    printf("  Single A78:     %.1f us (%.2f GB/s)\n", s1/ITERS, (double)N*row_bytes/(s1/ITERS*1e3));
    printf("  RAID 0 (2×A78): %.1f us (%.2f GB/s)\n", sw/ITERS, (double)N*row_bytes/(sw/ITERS*1e3));
    printf("  Speedup:        %.2fx\n", (s1/ITERS)/(sw/ITERS));
    printf("\n════════════════════════════════════════════════════════════\n\n");

    free(t1); free(t1a); free(tt0); free(tt1); free(tw); free(act); free(out);
    pthread_barrier_destroy(&bs2); pthread_barrier_destroy(&bd2);
    munmap((void*)data, st.st_size); close(fd);
    return 0;
}
