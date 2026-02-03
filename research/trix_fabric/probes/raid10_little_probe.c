/*
 * raid10_little_probe.c — RAID 10 Striped+Mirrored MatVec on Little Cores
 *
 * Phase 1B: 3 mirrored pairs on cpu0-5. Each pair computes the same
 * stripe (1536 rows). Both threads write to the same output. Measures
 * throughput and per-pair jitter reduction from mirroring.
 *
 * $CC -O2 -march=armv8.2-a+dotprod -o raid10_little_probe raid10_little_probe.c -lm -lpthread
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
#include <stdatomic.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static double now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e6+ts.tv_nsec/1e3; }

/* ── GGUF Parser (shared) ── */
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
static void matvec_q4_neon(float *out, const uint8_t *q4, const int8_t *act, int N, int K) {
    const int bpr=K/32,bs=18; const int8x16_t eight=vdupq_n_s8(8);
    for(int n=0;n<N;n++){float sum=0;const uint8_t *row=q4+(size_t)n*bpr*bs;
        for(int b=0;b<bpr;b++){uint16_t sh;memcpy(&sh,row,2);float sc=f16_to_f32(sh);
            uint8x16_t raw=vld1q_u8(row+2);
            int8x16_t lo=vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw,vdupq_n_u8(0x0F))),eight);
            int8x16_t hi=vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw,4)),eight);
            int8x16x2_t ap=vld2q_s8(act+b*32);
            int32x4_t a0=vdotq_s32(vdupq_n_s32(0),lo,ap.val[0]);
            int32x4_t a1=vdotq_s32(vdupq_n_s32(0),hi,ap.val[1]);
            sum+=sc*(float)(vaddvq_s32(a0)+vaddvq_s32(a1))/64.0f;row+=bs;}out[n]=sum;}
}
#endif

static int cmpd(const void *a,const void *b){double d=*(const double*)a-*(const double*)b;return(d>0)-(d<0);}
static void pstats(const char *label,double *t,int n){
    qsort(t,n,sizeof(double),cmpd);double s=0;for(int i=0;i<n;i++)s+=t[i];
    printf("  %-38s mean=%7.1f  min=%7.1f  p50=%7.1f  p99=%7.1f  max=%7.1f us\n",label,s/n,t[0],t[n/2],t[(int)(n*0.99)],t[n-1]);
}

/* ── RAID 10 worker ── */
typedef struct {
    int thread_id, cpu_id, pair_id;
    float *out; const uint8_t *wgt; const int8_t *act;
    int N, K, n_iters;
    pthread_barrier_t *bar_start, *bar_done;
    double *times;
} R10Arg;

static void *r10_worker(void *a) {
    R10Arg *w = (R10Arg *)a;
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

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TensorInfo ti; const uint8_t *wgt;
    if (find_tensor(data, "blk.0.ffn_gate.weight", &ti, &wgt) != 0) { fprintf(stderr, "not found\n"); return 1; }
    int K=(int)ti.dims[0], N=(int)ti.dims[1];
    size_t rb = (size_t)(K/32)*18;
    int stripe = N/3; /* 1536 rows per stripe */

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  PROBE 1B: RAID 10 on Little Cores (cpu0-5)\n");
    printf("  Tensor: %s [N=%d, K=%d] = %.2f MiB\n", ti.name, N, K, (double)N*rb/(1024*1024));
    printf("  RAID 10: 3 mirrored pairs × %d rows = %.1f KiB/stripe\n", stripe, (double)stripe*rb/1024);
    printf("  Pairs: (cpu0+1) (cpu2+3) (cpu4+5)\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    int8_t *act=calloc(K,1); float *out=calloc(N,sizeof(float));
    srand(42); for(int i=0;i<K;i++) act[i]=(int8_t)(((float)rand()/RAND_MAX-0.5f)*128);

    int ITERS = 1000;

    /* ── Single A55 core baseline ── */
    { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(0, &cs); sched_setaffinity(0, sizeof(cs), &cs); }
    for(int i=0;i<5;i++) matvec_q4_neon(out, wgt, act, N, K);
    double *t_single = calloc(ITERS, sizeof(double));
    for(int i=0;i<ITERS;i++){double t=now_us();matvec_q4_neon(out,wgt,act,N,K);t_single[i]=now_us()-t;}
    pstats("Single A55 (cpu0, N=4608)", t_single, ITERS);

    /* ── RAID 0 on 3 little cores (no mirroring) ── */
    printf("\n  --- RAID 0 (3-way, no mirror, cpu0+2+4) ---\n");
    {
        pthread_barrier_t bs3, bd3;
        pthread_barrier_init(&bs3, NULL, 4); pthread_barrier_init(&bd3, NULL, 4);
        double *tw3 = calloc(ITERS, sizeof(double));
        double *tt[3]; for(int i=0;i<3;i++) tt[i]=calloc(ITERS,sizeof(double));
        int cpus[3] = {0, 2, 4};
        R10Arg args[3];
        pthread_t threads[3];
        for (int p=0;p<3;p++) {
            args[p] = (R10Arg){p, cpus[p], p, out+p*stripe, wgt+(size_t)p*stripe*rb, act, stripe, K, ITERS, &bs3, &bd3, tt[p]};
            pthread_create(&threads[p], NULL, r10_worker, &args[p]);
        }
        for(int i=0;i<ITERS;i++){double t=now_us();pthread_barrier_wait(&bs3);pthread_barrier_wait(&bd3);tw3[i]=now_us()-t;}
        for(int p=0;p<3;p++) pthread_join(threads[p],NULL);
        for(int p=0;p<3;p++){char lb[64];snprintf(lb,64,"RAID 0 stripe %d (cpu%d)",p,cpus[p]);pstats(lb,tt[p],ITERS);}
        pstats("RAID 0 wall clock (3-way)", tw3, ITERS);
        for(int p=0;p<3;p++) free(tt[p]); free(tw3);
        pthread_barrier_destroy(&bs3); pthread_barrier_destroy(&bd3);
    }

    /* ── RAID 10 on 6 little cores (3 mirrored pairs) ── */
    printf("\n  --- RAID 10 (3 mirrored pairs, cpu0-5) ---\n");
    {
        pthread_barrier_t bs6, bd6;
        pthread_barrier_init(&bs6, NULL, 7); /* 6 workers + main */
        pthread_barrier_init(&bd6, NULL, 7);
        double *tw6 = calloc(ITERS, sizeof(double));
        double *tt[6]; for(int i=0;i<6;i++) tt[i]=calloc(ITERS,sizeof(double));
        /* Pair 0: cpu0+cpu1, rows 0-1535
         * Pair 1: cpu2+cpu3, rows 1536-3071
         * Pair 2: cpu4+cpu5, rows 3072-4607 */
        int pair_cpus[6] = {0,1, 2,3, 4,5};
        R10Arg args[6];
        pthread_t threads[6];
        for(int i=0;i<6;i++){
            int pair = i/2;
            args[i] = (R10Arg){i, pair_cpus[i], pair,
                out + pair*stripe,
                wgt + (size_t)pair*stripe*rb,
                act, stripe, K, ITERS, &bs6, &bd6, tt[i]};
            pthread_create(&threads[i], NULL, r10_worker, &args[i]);
        }
        for(int i=0;i<ITERS;i++){double t=now_us();pthread_barrier_wait(&bs6);pthread_barrier_wait(&bd6);tw6[i]=now_us()-t;}
        for(int i=0;i<6;i++) pthread_join(threads[i],NULL);

        /* Per-thread stats */
        for(int i=0;i<6;i++){char lb[64];snprintf(lb,64,"Pair %d, cpu%d",i/2,pair_cpus[i]);pstats(lb,tt[i],ITERS);}
        pstats("RAID 10 wall clock (6 threads)", tw6, ITERS);

        /* Mirror analysis: per-pair, how often does one thread finish before the other? */
        printf("\n  ── Mirror jitter analysis ──\n");
        for(int p=0;p<3;p++){
            int t0=p*2, t1=p*2+1;
            int wins_a=0, wins_b=0;
            double sum_jitter=0, max_jitter=0;
            for(int i=0;i<ITERS;i++){
                double ja = tt[t0][i], jb = tt[t1][i];
                double diff = fabs(ja - jb);
                sum_jitter += diff;
                if(diff > max_jitter) max_jitter = diff;
                if(ja < jb) wins_a++; else wins_b++;
            }
            printf("  Pair %d (cpu%d vs cpu%d): wins=%d/%d  avg_jitter=%.1f us  max_jitter=%.1f us\n",
                   p, pair_cpus[t0], pair_cpus[t1], wins_a, wins_b, sum_jitter/ITERS, max_jitter);
        }

        /* Per-pair "effective" time (min of two threads = what RAID 10 delivers) */
        printf("\n  ── RAID 10 effective (min per pair) ──\n");
        double *eff = calloc(ITERS, sizeof(double));
        for(int i=0;i<ITERS;i++){
            double worst_pair = 0;
            for(int p=0;p<3;p++){
                double best = tt[p*2][i] < tt[p*2+1][i] ? tt[p*2][i] : tt[p*2+1][i];
                if(best > worst_pair) worst_pair = best;
            }
            eff[i] = worst_pair;
        }
        pstats("RAID 10 effective (min-of-pair)", eff, ITERS);

        /* Summary */
        double s_single=0, s_r10=0, s_eff=0;
        for(int i=0;i<ITERS;i++){s_single+=t_single[i];s_r10+=tw6[i];s_eff+=eff[i];}
        printf("\n  ── Summary ──\n");
        printf("  Single A55:              %.1f us\n", s_single/ITERS);
        printf("  RAID 10 wall clock:      %.1f us\n", s_r10/ITERS);
        printf("  RAID 10 effective:       %.1f us\n", s_eff/ITERS);
        printf("  Speedup vs single A55:   %.2fx (wall)  %.2fx (effective)\n",
               (s_single/ITERS)/(s_r10/ITERS), (s_single/ITERS)/(s_eff/ITERS));

        free(eff); for(int i=0;i<6;i++) free(tt[i]); free(tw6);
        pthread_barrier_destroy(&bs6); pthread_barrier_destroy(&bd6);
    }

    printf("\n════════════════════════════════════════════════════════════\n\n");
    free(t_single); free(act); free(out);
    munmap((void*)data,st.st_size); close(fd);
    return 0;
}
