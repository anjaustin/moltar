/*
 * prefetch_assist_probe.c — Little-Core Prefetch Farm for Big-Core MatVec
 *
 * Phase 1D: Tests whether little cores prefetching weight data into shared
 * L3/SLC improves big-core matvec performance. This directly tests whether
 * the Dimensity 930's cache hierarchy shares L3 across clusters.
 *
 * Setup: two weight tensors (blk.0 and blk.1 ffn_gate), alternating layers
 * to prevent L2 caching from hiding the effect.
 *
 * Control:    big cores alternate layers, no prefetch help
 * Experiment: little cores prefetch next layer while big cores compute current
 *
 * $CC -O2 -march=armv8.2-a+dotprod -o prefetch_assist_probe prefetch_assist_probe.c -lm -lpthread
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
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static double now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e6+ts.tv_nsec/1e3; }

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

/* ── Prefetch thread state ── */
static atomic_int pf_go = 0;       /* 1 = prefetch, 0 = idle, -1 = exit */
static const uint8_t *pf_target;   /* data to prefetch */
static size_t pf_size;             /* bytes to prefetch */
static atomic_int pf_done = 0;

static void *prefetch_worker(void *arg) {
    int cpu_id = *(int *)arg;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu_id, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    while (1) {
        /* Spin-wait for signal */
        while (atomic_load(&pf_go) == 0) { /* spin */ }
        if (atomic_load(&pf_go) < 0) break;  /* exit signal */

        /* Prefetch: touch every cache line (64 bytes) */
        volatile uint8_t sum = 0;
        const uint8_t *p = pf_target;
        size_t sz = pf_size;
        for (size_t off = 0; off < sz; off += 64) {
            sum += p[off];
        }
        (void)sum;

        atomic_store(&pf_done, 1);
        atomic_store(&pf_go, 0);
    }
    return NULL;
}

/* ── Compute worker for RAID 0 big-core ── */
typedef struct {
    float *out; const uint8_t *wgt; const int8_t *act;
    int N, K, cpu_id;
    pthread_barrier_t *bar_start, *bar_done;
    int n_iters;
    const uint8_t *wgt_alt;  /* alternating layer */
    double *times;
} ComputeArg;

static void *compute_worker(void *arg) {
    ComputeArg *c = (ComputeArg *)arg;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(c->cpu_id, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    for (int i = 0; i < c->n_iters; i++) {
        pthread_barrier_wait(c->bar_start);
        const uint8_t *w = (i & 1) ? c->wgt_alt : c->wgt;
        double t0 = now_us();
        matvec_q4_neon(c->out, w, c->act, c->N, c->K);
        c->times[i] = now_us() - t0;
        pthread_barrier_wait(c->bar_done);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    /* Load two tensors */
    TensorInfo ti0, ti1; const uint8_t *wgt0, *wgt1;
    if (find_tensor(data, "blk.0.ffn_gate.weight", &ti0, &wgt0) != 0) { fprintf(stderr, "blk.0 not found\n"); return 1; }
    if (find_tensor(data, "blk.1.ffn_gate.weight", &ti1, &wgt1) != 0) { fprintf(stderr, "blk.1 not found\n"); return 1; }
    int K=(int)ti0.dims[0], N=(int)ti0.dims[1];
    size_t tensor_bytes = (size_t)N * (K/32) * 18;

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  PROBE 1D: Prefetch Assist (Little Cores → Big Cores)\n");
    printf("  Layer 0: %s [N=%d, K=%d] = %.2f MiB\n", ti0.name, N, K, (double)tensor_bytes/(1024*1024));
    printf("  Layer 1: %s [N=%d, K=%d] = %.2f MiB\n", ti1.name, N, K, (double)tensor_bytes/(1024*1024));
    printf("  Alternating layers to prevent L2 caching\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    int8_t *act=calloc(K,1); float *out=calloc(N,sizeof(float));
    srand(42); for(int i=0;i<K;i++) act[i]=(int8_t)(((float)rand()/RAND_MAX-0.5f)*128);

    int ITERS = 500;
    int hN = N/2;
    size_t rb = (size_t)(K/32)*18;

    /* ══════════════════════════════════════════════════════════
     * CONTROL: Single big core, alternating layers, no prefetch
     * ══════════════════════════════════════════════════════════ */
    printf("  === Control: single big core (cpu6), no prefetch ===\n");
    {
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(6, &cs);
        sched_setaffinity(0, sizeof(cs), &cs);

        /* warmup */
        for(int i=0;i<10;i++){matvec_q4_neon(out,wgt0,act,N,K);matvec_q4_neon(out,wgt1,act,N,K);}

        double *tc = calloc(ITERS, sizeof(double));
        for(int i=0;i<ITERS;i++){
            const uint8_t *w = (i & 1) ? wgt1 : wgt0;
            double t=now_us();
            matvec_q4_neon(out, w, act, N, K);
            tc[i]=now_us()-t;
        }
        pstats("Control (1×A78, alternating)", tc, ITERS);
        free(tc);
    }

    /* ══════════════════════════════════════════════════════════
     * CONTROL 2: RAID 0 big cores, alternating layers, no prefetch
     * ══════════════════════════════════════════════════════════ */
    printf("\n  === Control: RAID 0 big (cpu6+7), no prefetch ===\n");
    {
        pthread_barrier_t bs, bd;
        pthread_barrier_init(&bs, NULL, 3); pthread_barrier_init(&bd, NULL, 3);
        double *tt0=calloc(ITERS,sizeof(double)), *tt1=calloc(ITERS,sizeof(double)), *tw=calloc(ITERS,sizeof(double));

        ComputeArg c0 = {out, wgt0, act, hN, K, 6, &bs, &bd, ITERS, wgt1, tt0};
        ComputeArg c1 = {out+hN, wgt0+(size_t)hN*rb, act, hN, K, 7, &bs, &bd, ITERS, wgt1+(size_t)hN*rb, tt1};
        pthread_t th[2];
        pthread_create(&th[0], NULL, compute_worker, &c0);
        pthread_create(&th[1], NULL, compute_worker, &c1);
        for(int i=0;i<ITERS;i++){double t=now_us();pthread_barrier_wait(&bs);pthread_barrier_wait(&bd);tw[i]=now_us()-t;}
        pthread_join(th[0],NULL); pthread_join(th[1],NULL);
        pstats("RAID 0 (2×A78, alternating)", tw, ITERS);

        pthread_barrier_destroy(&bs); pthread_barrier_destroy(&bd);
        free(tt0); free(tt1); free(tw);
    }

    /* ══════════════════════════════════════════════════════════
     * EXPERIMENT: Single big core + prefetch on little cores
     * ══════════════════════════════════════════════════════════ */
    printf("\n  === Experiment: single big core + 4 little prefetch ===\n");
    {
        /* Spawn 4 prefetch threads on little cores */
        int pf_cpus[4] = {0, 1, 2, 3};
        pthread_t pf_threads[4];
        /* We use a simple model: one prefetch thread touches the full tensor.
         * Multiple threads would partition it, but for this probe we test
         * the basic concept with a single prefetch thread first. */
        int pf_cpu = 0;
        pthread_create(&pf_threads[0], NULL, prefetch_worker, &pf_cpu);

        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(6, &cs);
        sched_setaffinity(0, sizeof(cs), &cs);

        /* warmup */
        for(int i=0;i<10;i++){matvec_q4_neon(out,wgt0,act,N,K);matvec_q4_neon(out,wgt1,act,N,K);}

        double *te = calloc(ITERS, sizeof(double));
        for(int i=0;i<ITERS;i++){
            const uint8_t *w_curr = (i & 1) ? wgt1 : wgt0;
            const uint8_t *w_next = (i & 1) ? wgt0 : wgt1;

            /* Signal prefetch thread to start on next layer */
            pf_target = w_next;
            pf_size = tensor_bytes;
            atomic_store(&pf_done, 0);
            atomic_store(&pf_go, 1);

            /* Compute current layer on big core */
            double t=now_us();
            matvec_q4_neon(out, w_curr, act, N, K);
            te[i]=now_us()-t;

            /* Wait for prefetch to complete (should already be done) */
            while (!atomic_load(&pf_done)) { /* spin */ }
        }
        pstats("Big + prefetch (1×A78 + 1×A55)", te, ITERS);

        /* Shutdown prefetch thread */
        atomic_store(&pf_go, -1);
        pthread_join(pf_threads[0], NULL);
        free(te);
    }

    /* ══════════════════════════════════════════════════════════
     * EXPERIMENT 2: RAID 0 big + prefetch on little cores
     * ══════════════════════════════════════════════════════════ */
    printf("\n  === Experiment: RAID 0 big + 2 little prefetch ===\n");
    {
        /* 2 prefetch threads, each covering half the next tensor */
        int pf_cpu0 = 0, pf_cpu1 = 2;
        atomic_store(&pf_go, 0); atomic_store(&pf_done, 0);
        pthread_t pft0, pft1;
        pthread_create(&pft0, NULL, prefetch_worker, &pf_cpu0);

        /* For RAID 0 + prefetch, use single prefetch thread (simpler) */
        pthread_barrier_t bs, bd;
        pthread_barrier_init(&bs, NULL, 3); pthread_barrier_init(&bd, NULL, 3);
        double *tt0=calloc(ITERS,sizeof(double)), *tt1=calloc(ITERS,sizeof(double)), *tw=calloc(ITERS,sizeof(double));

        ComputeArg c0 = {out, wgt0, act, hN, K, 6, &bs, &bd, ITERS, wgt1, tt0};
        ComputeArg c1 = {out+hN, wgt0+(size_t)hN*rb, act, hN, K, 7, &bs, &bd, ITERS, wgt1+(size_t)hN*rb, tt1};
        pthread_t th[2];
        pthread_create(&th[0], NULL, compute_worker, &c0);
        pthread_create(&th[1], NULL, compute_worker, &c1);

        for(int i=0;i<ITERS;i++){
            const uint8_t *w_next = (i & 1) ? wgt0 : wgt1;
            pf_target = w_next;
            pf_size = tensor_bytes;
            atomic_store(&pf_done, 0);
            atomic_store(&pf_go, 1);

            double t=now_us();
            pthread_barrier_wait(&bs);
            pthread_barrier_wait(&bd);
            tw[i]=now_us()-t;

            while (!atomic_load(&pf_done)) { /* spin */ }
        }
        pthread_join(th[0],NULL); pthread_join(th[1],NULL);
        pstats("RAID 0 + prefetch (2×A78+1×A55)", tw, ITERS);

        atomic_store(&pf_go, -1);
        pthread_join(pft0, NULL);
        pthread_barrier_destroy(&bs); pthread_barrier_destroy(&bd);
        free(tt0); free(tt1); free(tw);
    }

    /* ══════════════════════════════════════════════════════════
     * Summary
     * ══════════════════════════════════════════════════════════ */
    printf("\n  If experiment < control → L3/SLC is shared across clusters!\n");
    printf("  If experiment ≈ control → L3 is partitioned, prefetch doesn't help.\n");

    printf("\n════════════════════════════════════════════════════════════\n\n");
    free(act); free(out);
    munmap((void*)data,st.st_size); close(fd);
    return 0;
}
