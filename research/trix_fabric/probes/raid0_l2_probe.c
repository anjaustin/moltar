/*
 * raid0_l2_probe.c — RAID 0 from within L2 cache
 *
 * Two experiments:
 * 1. Bandwidth cliff: sweep working set sizes to find L1/L2/L3 boundaries
 * 2. L2-tiled RAID 0: tile the matvec so each chunk fits in L2, stripe tiles
 *    across cpu6+7 which share the L2
 *
 * $CC -O2 -march=armv8.2-a+dotprod -o raid0_l2_probe raid0_l2_probe.c -lm
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

/* ── GGUF Parser (same as other probes) ── */
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

/*
 * matvec_q4_neon_slice: compute rows [row_start, row_start+n_rows) of a Q4_0 matvec
 * q4 points to start of full tensor, act is Kx1 activation, out is per-row output
 */
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

/* Full matvec for baseline */
static void matvec_q4_neon(float *out, const uint8_t *q4, const int8_t *act, int N, int K) {
    matvec_q4_neon_slice(out, q4, act, 0, N, K);
}
#endif

/* ═══════════════════════════════════════════════════════════
 * Experiment 1: Bandwidth cliff — find L1/L2/L3 boundaries
 * ═══════════════════════════════════════════════════════════ */
static void bandwidth_cliff(void) {
    printf("\n  === Experiment 1: Bandwidth Cliff (cpu6) ===\n");
    printf("  Sequential read bandwidth vs working set size\n\n");
    pin(6);

    /* Test sizes from 4KB to 8MB */
    int sizes_kb[] = {4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    int n_sizes = sizeof(sizes_kb) / sizeof(sizes_kb[0]);
    volatile uint64_t dummy = 0;

    printf("  %8s  %8s  %8s\n", "Size", "BW", "Latency");
    printf("  %8s  %8s  %8s\n", "(KB)", "(GB/s)", "(ns/CL)");
    printf("  ────────  ────────  ────────\n");

    for (int si = 0; si < n_sizes; si++) {
        size_t sz = (size_t)sizes_kb[si] * 1024;
        uint8_t *buf = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        memset(buf, 0xAA, sz); /* touch all pages */

        /* Warmup: iterate 3x */
        for (int w = 0; w < 3; w++) {
            for (size_t i = 0; i < sz; i += 64)
                dummy += *(volatile uint64_t*)(buf + i);
        }

        /* Measure: iterate enough to get stable timing */
        int iters = (32 * 1024 * 1024) / (int)sz; /* ~32MB total read */
        if (iters < 4) iters = 4;
        if (iters > 2000) iters = 2000;

        double t0 = now_us();
        for (int it = 0; it < iters; it++) {
            for (size_t i = 0; i < sz; i += 64)
                dummy += *(volatile uint64_t*)(buf + i);
        }
        double elapsed = now_us() - t0;

        double total_bytes = (double)sz * iters;
        double bw_gbps = total_bytes / (elapsed * 1e-6) / 1e9;
        double n_cls = (double)(sz / 64) * iters;
        double ns_per_cl = (elapsed * 1000.0) / n_cls;

        printf("  %6dKB  %7.2f  %7.1f\n", sizes_kb[si], bw_gbps, ns_per_cl);
        munmap(buf, sz);
    }
    printf("  (dummy=%lu)\n", (unsigned long)dummy);
}

/* ═══════════════════════════════════════════════════════════
 * Experiment 2: L2-Tiled RAID 0
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    const uint8_t *q4;
    const int8_t *act;
    float *out;
    int row_start;
    int n_rows;
    int K;
    int cpu;
    int iters;
    double *times;       /* per-iteration wall times */
    volatile int *go;    /* spin barrier */
    volatile int *done;
} TileArg;

static void *tile_worker(void *arg) {
    TileArg *a = (TileArg*)arg;
    pin(a->cpu);

    for (int it = 0; it < a->iters; it++) {
        /* Spin-wait for go signal */
        while (__atomic_load_n(a->go, __ATOMIC_ACQUIRE) != it + 1)
            ;

        double t0 = now_us();
        matvec_q4_neon_slice(a->out, a->q4, a->act, a->row_start, a->n_rows, a->K);
        a->times[it] = now_us() - t0;

        __atomic_fetch_add(a->done, 1, __ATOMIC_RELEASE);
    }
    return NULL;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static void print_stats(const char *label, double *times, int n) {
    double sum = 0, mn = 1e18, mx = 0;
    for (int i = 0; i < n; i++) { sum += times[i]; if (times[i] < mn) mn = times[i]; if (times[i] > mx) mx = times[i]; }
    double sorted[n]; memcpy(sorted, times, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double);
    printf("  %-40s mean=%7.1f  min=%7.1f  p50=%7.1f  p99=%7.1f  max=%7.1f us\n",
           label, sum / n, mn, sorted[n / 2], sorted[(int)(n * 0.99)], mx);
}

static void l2_tiled_raid0(const uint8_t *q4, const int8_t *act, int N, int K,
                           int l2_rows, int full_rows) {
    int ITERS = 500;
    volatile float sink = 0;

    /* ── Baseline: single A78, full tensor ── */
    printf("\n  --- Baseline: single A78 (cpu6), full N=%d ---\n", full_rows);
    pin(6);
    float *out_full = calloc(full_rows, sizeof(float));
    /* warmup */
    for (int i = 0; i < 5; i++) { matvec_q4_neon_slice(out_full, q4, act, 0, full_rows, K); sink += out_full[0]; }
    double *t_base = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        double t = now_us();
        matvec_q4_neon_slice(out_full, q4, act, 0, full_rows, K);
        sink += out_full[0];
        t_base[i] = now_us() - t;
    }
    print_stats("Single A78 (full)", t_base, ITERS);

    /* ── Baseline: single A78, L2-sized tile only ── */
    printf("\n  --- Baseline: single A78 (cpu6), L2 tile N=%d ---\n", l2_rows);
    float *out_tile = calloc(l2_rows, sizeof(float));
    for (int i = 0; i < 5; i++) { matvec_q4_neon_slice(out_tile, q4, act, 0, l2_rows, K); sink += out_tile[0]; }
    double *t_tile1 = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        double t = now_us();
        matvec_q4_neon_slice(out_tile, q4, act, 0, l2_rows, K);
        sink += out_tile[0];
        t_tile1[i] = now_us() - t;
    }
    print_stats("Single A78 (L2 tile)", t_tile1, ITERS);

    /* ── RAID 0: two A78s, each does half of L2 tile ── */
    int half = l2_rows / 2;
    printf("\n  --- RAID 0: 2×A78 (cpu6+7), L2 tile N=%d, %d rows each ---\n", l2_rows, half);

    float *out0 = calloc(half, sizeof(float));
    float *out1 = calloc(half, sizeof(float));
    double *t0_arr = calloc(ITERS, sizeof(double));
    double *t1_arr = calloc(ITERS, sizeof(double));
    double *t_wall = calloc(ITERS, sizeof(double));

    volatile int go_flag = 0;
    volatile int done_count = 0;

    TileArg arg0 = { q4, act, out0, 0, half, K, 6, ITERS, t0_arr, &go_flag, &done_count };
    TileArg arg1 = { q4, act, out1, half, half, K, 7, ITERS, t1_arr, &go_flag, &done_count };

    pthread_t th0, th1;
    pthread_create(&th0, NULL, tile_worker, &arg0);
    pthread_create(&th1, NULL, tile_worker, &arg1);

    /* Drive iterations from main thread */
    for (int it = 0; it < ITERS; it++) {
        __atomic_store_n(&done_count, 0, __ATOMIC_RELEASE);
        double tw0 = now_us();
        __atomic_store_n(&go_flag, it + 1, __ATOMIC_RELEASE);

        /* Wait for both threads to finish */
        while (__atomic_load_n(&done_count, __ATOMIC_ACQUIRE) < 2)
            ;
        t_wall[it] = now_us() - tw0;
    }

    pthread_join(th0, NULL);
    pthread_join(th1, NULL);

    print_stats("RAID 0 thread 0 (cpu6)", t0_arr, ITERS);
    print_stats("RAID 0 thread 1 (cpu7)", t1_arr, ITERS);
    print_stats("RAID 0 wall clock", t_wall, ITERS);
    sink += out0[0] + out1[0];

    /* ── RAID 0: full tensor, tiled into L2 chunks ── */
    int n_tiles = (full_rows + l2_rows - 1) / l2_rows;
    printf("\n  --- RAID 0 Tiled: 2×A78, full N=%d in %d tiles of %d rows ---\n",
           full_rows, n_tiles, l2_rows);

    /* Single-threaded tiled baseline first */
    pin(6);
    double *t_tiled_1t = calloc(ITERS, sizeof(double));
    for (int i = 0; i < 5; i++) { matvec_q4_neon_slice(out_full, q4, act, 0, full_rows, K); sink += out_full[0]; }
    for (int it = 0; it < ITERS; it++) {
        double t = now_us();
        for (int tile = 0; tile < n_tiles; tile++) {
            int rs = tile * l2_rows;
            int nr = (rs + l2_rows <= full_rows) ? l2_rows : (full_rows - rs);
            matvec_q4_neon_slice(out_full + rs, q4, act, rs, nr, K);
        }
        sink += out_full[0];
        t_tiled_1t[it] = now_us() - t;
    }
    print_stats("Tiled 1-thread (cpu6)", t_tiled_1t, ITERS);

    /* 2-thread tiled: for each tile, stripe half to each core */
    /* Use simple sequential tile approach — each tile is RAID 0'd */
    float *out_r0 = calloc(full_rows, sizeof(float));
    double *t_tiled_2t = calloc(ITERS, sizeof(double));

    /* For the tiled version, we'll do it synchronously with barriers per tile */
    volatile int go2 = 0;
    volatile int done2 = 0;

    /* We need a different approach: process all tiles in sequence,
     * but within each tile, RAID 0 across cores.
     * Simplest: just do the whole matvec as one RAID 0 (N/2 each) */
    int half_full = full_rows / 2;
    float *out_f0 = calloc(half_full, sizeof(float));
    float *out_f1 = calloc(half_full, sizeof(float));
    double *tf0 = calloc(ITERS, sizeof(double));
    double *tf1 = calloc(ITERS, sizeof(double));
    double *tfw = calloc(ITERS, sizeof(double));

    volatile int go3 = 0;
    volatile int done3 = 0;

    TileArg farg0 = { q4, act, out_f0, 0, half_full, K, 6, ITERS, tf0, &go3, &done3 };
    TileArg farg1 = { q4, act, out_f1, half_full, half_full, K, 7, ITERS, tf1, &go3, &done3 };

    pthread_t fth0, fth1;
    pthread_create(&fth0, NULL, tile_worker, &farg0);
    pthread_create(&fth1, NULL, tile_worker, &farg1);

    printf("\n  --- RAID 0: 2×A78, full N=%d, %d rows each (no tiling) ---\n",
           full_rows, half_full);

    for (int it = 0; it < ITERS; it++) {
        __atomic_store_n(&done3, 0, __ATOMIC_RELEASE);
        double tw = now_us();
        __atomic_store_n(&go3, it + 1, __ATOMIC_RELEASE);
        while (__atomic_load_n(&done3, __ATOMIC_ACQUIRE) < 2)
            ;
        tfw[it] = now_us() - tw;
    }
    pthread_join(fth0, NULL);
    pthread_join(fth1, NULL);

    print_stats("RAID 0 full thread 0 (cpu6)", tf0, ITERS);
    print_stats("RAID 0 full thread 1 (cpu7)", tf1, ITERS);
    print_stats("RAID 0 full wall clock", tfw, ITERS);
    sink += out_f0[0] + out_f1[0];

    /* ── Summary ── */
    double base_mean = 0; for (int i = 0; i < ITERS; i++) base_mean += t_base[i]; base_mean /= ITERS;
    double tile1_mean = 0; for (int i = 0; i < ITERS; i++) tile1_mean += t_tile1[i]; tile1_mean /= ITERS;
    double wall_l2_mean = 0; for (int i = 0; i < ITERS; i++) wall_l2_mean += t_wall[i]; wall_l2_mean /= ITERS;
    double wall_full_mean = 0; for (int i = 0; i < ITERS; i++) wall_full_mean += tfw[i]; wall_full_mean /= ITERS;

    printf("\n  ── Summary ──\n");
    printf("  Single A78 (full N=%d):      %7.1f us\n", full_rows, base_mean);
    printf("  Single A78 (L2 tile N=%d):   %7.1f us\n", l2_rows, tile1_mean);
    printf("  RAID 0 L2-tile (N=%d):       %7.1f us  (%.2fx vs single L2 tile)\n",
           l2_rows, wall_l2_mean, tile1_mean / wall_l2_mean);
    printf("  RAID 0 full (N=%d):          %7.1f us  (%.2fx vs single full)\n",
           full_rows, wall_full_mean, base_mean / wall_full_mean);
    printf("  (sink=%.1f)\n", (double)sink);

    free(out_full); free(out_tile); free(out0); free(out1); free(out_r0);
    free(out_f0); free(out_f1);
    free(t_base); free(t_tile1); free(t0_arr); free(t1_arr); free(t_wall);
    free(t_tiled_1t); free(t_tiled_2t); free(tf0); free(tf1); free(tfw);
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

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  PROBE: L2-Resident RAID 0\n");
    printf("  Tensor: %s [N=%d, K=%d] = %.2f MiB\n", ti.name, N, K,
           (double)N * (K / 32) * 18 / (1024.0 * 1024.0));
    printf("  Q4_0 row size: %d bytes\n", (K / 32) * 18);
    printf("════════════════════════════════════════════════════════════\n");

    int8_t *act = calloc(K, 1);
    srand(42);
    for (int i = 0; i < K; i++) act[i] = (int8_t)(((float)rand() / RAND_MAX - 0.5f) * 128);

    /* Experiment 1: find cache boundaries */
    bandwidth_cliff();

    /* Experiment 2: L2-tiled RAID 0
     * A78 L2 is likely 512KB. Each Q4_0 row = (1024/32)*18 = 576 bytes.
     * 512KB / 576 = ~910 rows fit in L2.
     * But we also need the activation vector (1024 bytes) and output buffer.
     * Conservative: use 768 rows = 442KB of weights + 1KB act + 3KB out = ~446KB.
     * Also test with smaller tiles to see effect.
     */
    int row_bytes = (K / 32) * 18;
    printf("\n  Row size: %d bytes. Testing tile sizes:\n", row_bytes);

    /* Test several tile sizes to find optimal L2 utilization */
    int tile_rows[] = {256, 384, 512, 768, 1024, 1536};
    int n_tiles = sizeof(tile_rows) / sizeof(tile_rows[0]);

    for (int ti_idx = 0; ti_idx < n_tiles; ti_idx++) {
        int tr = tile_rows[ti_idx];
        if (tr > N) continue;
        int tile_kb = (tr * row_bytes) / 1024;
        printf("\n  ╔══════════════════════════════════════════╗\n");
        printf("  ║ Tile: %d rows = %dKB weight data          \n", tr, tile_kb);
        printf("  ╚══════════════════════════════════════════╝\n");
        l2_tiled_raid0(wgt, act, N, K, tr, N);
    }

    printf("\n════════════════════════════════════════════════════════════\n\n");
    free(act);
    munmap((void*)data, st.st_size);
    close(fd);
    return 0;
}
