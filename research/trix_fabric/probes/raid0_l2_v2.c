/*
 * raid0_l2_v2.c — L2-Resident RAID 0, Fixed Barrier
 *
 * Per-thread times from v1 confirmed perfect 2x scaling.
 * This version fixes the wall-clock measurement by:
 * 1. Using pthread_barrier for zero-overhead sync
 * 2. Measuring wall time from inside a worker thread
 * 3. Pinning the main thread to a big core too
 *
 * $CC -O2 -march=armv8.2-a+dotprod -o raid0_l2_v2 raid0_l2_v2.c -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void pin(int cpu) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}

/* ── GGUF Parser ── */
#define GGUF_MAGIC 0x46554747
#define GGUF_TYPE_Q4_0 2
typedef struct { char name[256]; uint32_t n_dims; uint64_t dims[4]; uint32_t type; uint64_t offset; } TensorInfo;
static const uint8_t *skip_str(const uint8_t *p) { return p+8+*(const uint64_t*)p; }
static void read_str(const uint8_t *p, char *o, int m) { uint64_t l=*(const uint64_t*)p; int c=l<(uint64_t)m-1?(int)l:m-1; memcpy(o,p+8,c); o[c]=0; }
static const uint8_t *skip_val(const uint8_t *p, uint32_t t) {
    switch(t){case 0:case 1:case 7:return p+1;case 2:case 3:return p+2;case 4:case 5:case 6:return p+4;
    case 10:case 11:case 12:return p+8;case 8:return skip_str(p);
    case 9:{uint32_t et=*(const uint32_t*)p;p+=4;uint64_t n=*(const uint64_t*)p;p+=8;for(uint64_t i=0;i<n;i++)p=skip_val(p,et);return p;}
    default:return p;}
}
static int find_tensor(const uint8_t *data, const char *name, TensorInfo *out, const uint8_t **td) {
    const uint8_t *p=data+8; uint64_t nt=*(const uint64_t*)p; p+=8; uint64_t nk=*(const uint64_t*)p; p+=8;
    for(uint64_t i=0;i<nk;i++){p=skip_str(p);uint32_t vt=*(const uint32_t*)p;p+=4;p=skip_val(p,vt);}
    TensorInfo *all=calloc(nt,sizeof(TensorInfo));
    for(uint64_t i=0;i<nt;i++){read_str(p,all[i].name,256);p=skip_str(p);all[i].n_dims=*(const uint32_t*)p;p+=4;
        for(uint32_t d=0;d<all[i].n_dims;d++){all[i].dims[d]=*(const uint64_t*)p;p+=8;}all[i].type=*(const uint32_t*)p;p+=4;all[i].offset=*(const uint64_t*)p;p+=8;}
    uint64_t base=((p-data)+31)&~31ULL; int found=-1;
    for(uint64_t i=0;i<nt;i++){if(strcmp(all[i].name,name)==0&&all[i].type==GGUF_TYPE_Q4_0){*out=all[i];found=0;break;}}
    if(found==0)*td=data+base+out->offset; free(all); return found;
}

static float f16_to_f32(uint16_t h) {
    uint32_t s=(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,f;
    if(!e){if(!m)f=s;else{e=1;while(!(m&0x400)){m<<=1;e--;}m&=0x3FF;f=s|((e+112)<<23)|(m<<13);}}
    else if(e==31)f=s|0x7F800000|(m<<13);else f=s|((e+112)<<23)|(m<<13);
    float r;memcpy(&r,&f,4);return r;
}

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void matvec_q4_neon_slice(float *out, const uint8_t *q4, const int8_t *act,
                                  int row_start, int n_rows, int K) {
    const int bpr = K / 32, bs = 18;
    const int8x16_t eight = vdupq_n_s8(8);
    for (int r = 0; r < n_rows; r++) {
        int n = row_start + r;
        float sum = 0;
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
        out[r] = sum;
    }
}
#endif

/* ── Stats ── */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static void print_stats(const char *label, double *t, int n) {
    double sum = 0, mn = 1e18, mx = 0;
    for (int i = 0; i < n; i++) { sum += t[i]; if (t[i] < mn) mn = t[i]; if (t[i] > mx) mx = t[i]; }
    double s[n]; memcpy(s, t, n * sizeof(double));
    qsort(s, n, sizeof(double), cmp_double);
    printf("  %-42s mean=%7.1f  min=%7.1f  p50=%7.1f  p99=%7.1f  max=%7.1f us\n",
           label, sum / n, mn, s[n / 2], s[(int)(n * 0.99)], mx);
}

/* ═══════════════════════════════════════════════════════════
 * RAID 0 with pthread_barrier — thread 0 measures wall time
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    const uint8_t *q4;
    const int8_t  *act;
    float         *out;
    int            row_start;
    int            n_rows;
    int            K;
    int            cpu;
    int            tid;       /* 0 = also measures wall time */
    int            iters;
    int            warmup;
    double        *times;     /* per-thread times */
    double        *wall;      /* wall times (only tid==0 writes) */
    volatile float *sink;
    pthread_barrier_t *bar_start;  /* sync before compute */
    pthread_barrier_t *bar_end;    /* sync after compute */
} WorkerArg;

static void *raid0_worker(void *arg) {
    WorkerArg *a = (WorkerArg*)arg;
    pin(a->cpu);

    /* Warmup */
    for (int i = 0; i < a->warmup; i++) {
        pthread_barrier_wait(a->bar_start);
        matvec_q4_neon_slice(a->out, a->q4, a->act, a->row_start, a->n_rows, a->K);
        *a->sink += a->out[0];
        pthread_barrier_wait(a->bar_end);
    }

    /* Measured iterations */
    for (int i = 0; i < a->iters; i++) {
        pthread_barrier_wait(a->bar_start);

        double t0 = now_us();
        matvec_q4_neon_slice(a->out, a->q4, a->act, a->row_start, a->n_rows, a->K);
        a->times[i] = now_us() - t0;
        *a->sink += a->out[0];

        pthread_barrier_wait(a->bar_end);

        /* Thread 0 measures wall clock (time between barriers) */
        if (a->tid == 0) {
            a->wall[i] = now_us() - t0;
        }
    }
    return NULL;
}

/* Run a single-thread baseline */
static void run_single(const char *label, const uint8_t *q4, const int8_t *act,
                       int N, int K, int cpu, int iters, volatile float *sink) {
    pin(cpu);
    float *out = calloc(N, sizeof(float));
    /* warmup */
    for (int i = 0; i < 10; i++) { matvec_q4_neon_slice(out, q4, act, 0, N, K); *sink += out[0]; }
    double *times = calloc(iters, sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t = now_us();
        matvec_q4_neon_slice(out, q4, act, 0, N, K);
        *sink += out[0];
        times[i] = now_us() - t;
    }
    print_stats(label, times, iters);
    free(out); free(times);
}

/* Run RAID 0 with 2 threads */
static void run_raid0(const char *label, const uint8_t *q4, const int8_t *act,
                      int N, int K, int cpu0, int cpu1, int iters, volatile float *sink) {
    int half = N / 2;
    float *out0 = calloc(half, sizeof(float));
    float *out1 = calloc(half, sizeof(float));
    double *t0 = calloc(iters, sizeof(double));
    double *t1 = calloc(iters, sizeof(double));
    double *tw = calloc(iters, sizeof(double));

    pthread_barrier_t bar_s, bar_e;
    pthread_barrier_init(&bar_s, NULL, 2);
    pthread_barrier_init(&bar_e, NULL, 2);

    WorkerArg a0 = { q4, act, out0, 0, half, K, cpu0, 0, iters, 10, t0, tw, sink, &bar_s, &bar_e };
    WorkerArg a1 = { q4, act, out1, half, half, K, cpu1, 1, iters, 10, t1, NULL, sink, &bar_s, &bar_e };

    pthread_t th0, th1;
    pthread_create(&th0, NULL, raid0_worker, &a0);
    pthread_create(&th1, NULL, raid0_worker, &a1);
    pthread_join(th0, NULL);
    pthread_join(th1, NULL);

    char buf0[80], buf1[80], bufw[80];
    snprintf(buf0, 80, "%s thread0 (cpu%d)", label, cpu0);
    snprintf(buf1, 80, "%s thread1 (cpu%d)", label, cpu1);
    snprintf(bufw, 80, "%s wall clock", label);
    print_stats(buf0, t0, iters);
    print_stats(buf1, t1, iters);
    print_stats(bufw, tw, iters);

    /* Compute speedup from summary */
    double sum_w = 0; for (int i = 0; i < iters; i++) sum_w += tw[i];
    double mean_w = sum_w / iters;
    printf("    → %.2f GB/s effective\n", (double)N * (K / 32) * 18 / (mean_w * 1e-6) / 1e9);

    pthread_barrier_destroy(&bar_s);
    pthread_barrier_destroy(&bar_e);
    free(out0); free(out1); free(t0); free(t1); free(tw);
}

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    TensorInfo ti; const uint8_t *wgt;
    if (find_tensor(data, "blk.0.ffn_gate.weight", &ti, &wgt) != 0) {
        fprintf(stderr, "Tensor not found\n"); return 1;
    }
    int K = (int)ti.dims[0], N = (int)ti.dims[1];
    int row_bytes = (K / 32) * 18;

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  PROBE: L2-Resident RAID 0 v2 (pthread_barrier)\n");
    printf("  Tensor: %s [N=%d, K=%d] = %.2f MiB\n", ti.name, N, K,
           (double)N * row_bytes / (1024.0 * 1024.0));
    printf("  Q4_0 row: %d bytes | L2 cliff: ~256-384KB\n", row_bytes);
    printf("════════════════════════════════════════════════════════════\n");

    int8_t *act = calloc(K, 1);
    srand(42);
    for (int i = 0; i < K; i++) act[i] = (int8_t)(((float)rand() / RAND_MAX - 0.5f) * 128);
    volatile float sink = 0;
    int ITERS = 1000;

    /* ── Full tensor baselines ── */
    printf("\n  === Full tensor (N=%d, %.1fKB) ===\n", N, (double)N * row_bytes / 1024.0);
    run_single("Single A78 (cpu6)", wgt, act, N, K, 6, ITERS, &sink);
    run_raid0("RAID 0 (2×A78)", wgt, act, N, K, 6, 7, ITERS, &sink);

    /* ── L2-resident tile sizes ── */
    int tile_rows[] = {128, 256, 384, 448, 512};
    for (int ti_idx = 0; ti_idx < 5; ti_idx++) {
        int tr = tile_rows[ti_idx];
        int tile_kb = tr * row_bytes / 1024;
        printf("\n  === Tile: %d rows (%dKB, %s L2) ===\n",
               tr, tile_kb, tile_kb <= 256 ? "fits" : "exceeds");
        char lbl1[80], lbl2[80];
        snprintf(lbl1, 80, "Single A78 (N=%d)", tr);
        snprintf(lbl2, 80, "RAID 0 2×A78 (N=%d)", tr);
        run_single(lbl1, wgt, act, tr, K, 6, ITERS, &sink);
        run_raid0(lbl2, wgt, act, tr, K, 6, 7, ITERS, &sink);
    }

    /* ── Summary table ── */
    printf("\n  ══════════════════════════════════════════\n");
    printf("  (sink=%.1f)\n", (double)sink);
    printf("\n════════════════════════════════════════════════════════════\n\n");

    free(act);
    munmap((void*)data, st.st_size);
    close(fd);
    return 0;
}
