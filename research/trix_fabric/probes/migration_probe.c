/*
 * migration_probe.c — Phase Transition Cost: Little → Big Core Migration
 *
 * Phase 1C: Measures sched_setaffinity() latency, cache cold-start penalty,
 * and warmup curve when migrating a matvec workload between core clusters.
 *
 * $CC -O2 -march=armv8.2-a+dotprod -o migration_probe migration_probe.c -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
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

static void pin(int cpu) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TensorInfo ti; const uint8_t *wgt;
    if (find_tensor(data, "blk.0.ffn_gate.weight", &ti, &wgt) != 0) { fprintf(stderr, "not found\n"); return 1; }
    int K=(int)ti.dims[0], N=(int)ti.dims[1];

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  PROBE 1C: Core Migration Cost\n");
    printf("  Tensor: %s [N=%d, K=%d]\n", ti.name, N, K);
    printf("════════════════════════════════════════════════════════════\n\n");

    int8_t *act=calloc(K,1); float *out=calloc(N,sizeof(float));
    volatile float sink = 0.0f;  /* prevent dead-code elimination of matvec */
    srand(42); for(int i=0;i<K;i++) act[i]=(int8_t)(((float)rand()/RAND_MAX-0.5f)*128);

    int SS_ITERS = 200;  /* steady-state iterations */
    int WARMUP = 20;     /* warmup curve points after migration */

    /* ══════════════════════════════════════════════════════════
     * TEST 1: Little → Big migration
     * ══════════════════════════════════════════════════════════ */
    printf("  === Little (cpu0) → Big (cpu6) migration ===\n\n");

    /* Phase 1: Steady state on little core */
    pin(0);
    for(int i=0;i<10;i++){matvec_q4_neon(out, wgt, act, N, K);sink+=out[0];} /* warmup */
    double *t_little = calloc(SS_ITERS, sizeof(double));
    for(int i=0;i<SS_ITERS;i++){double t=now_us();matvec_q4_neon(out,wgt,act,N,K);sink+=out[0];t_little[i]=now_us()-t;}
    double sum_l=0; for(int i=0;i<SS_ITERS;i++) sum_l+=t_little[i];
    printf("  Little steady-state (cpu0): %.1f us/matvec (mean of %d)\n", sum_l/SS_ITERS, SS_ITERS);

    /* Phase 2: Migration event */
    double t_pre = now_us();
    pin(6);  /* migrate to big core */
    double t_post = now_us();
    double syscall_us = t_post - t_pre;
    printf("  sched_setaffinity() latency: %.1f us\n\n", syscall_us);

    /* Phase 3: Warmup curve on big core (first N matvecs after migration) */
    printf("  Big-core warmup curve (first %d matvecs after migration):\n", WARMUP);
    double *t_warmup = calloc(WARMUP, sizeof(double));
    for(int i=0;i<WARMUP;i++){
        double t=now_us();
        matvec_q4_neon(out, wgt, act, N, K);sink+=out[0];
        t_warmup[i]=now_us()-t;
        printf("    matvec[%2d] = %7.1f us%s\n", i, t_warmup[i],
               i==0 ? "  ← cold cache" : "");
    }

    /* Phase 4: Steady state on big core */
    double *t_big = calloc(SS_ITERS, sizeof(double));
    for(int i=0;i<SS_ITERS;i++){double t=now_us();matvec_q4_neon(out,wgt,act,N,K);sink+=out[0];t_big[i]=now_us()-t;}
    double sum_b=0; for(int i=0;i<SS_ITERS;i++) sum_b+=t_big[i];
    printf("\n  Big steady-state (cpu6): %.1f us/matvec (mean of %d)\n", sum_b/SS_ITERS, SS_ITERS);

    /* Transition cost analysis */
    double cold_penalty = t_warmup[0] - (sum_b/SS_ITERS);
    double transition_gap = syscall_us + cold_penalty;
    printf("\n  ── Little → Big transition cost ──\n");
    printf("  syscall:         %7.1f us\n", syscall_us);
    printf("  cold-cache tax:  %7.1f us (first matvec - steady state)\n", cold_penalty);
    printf("  total gap:       %7.1f us\n", transition_gap);
    printf("  = %.2f tokens of generation time (at 44 tok/s = 22.7ms/tok)\n", transition_gap / 22700.0);

    /* ══════════════════════════════════════════════════════════
     * TEST 2: Big → Little migration (reverse)
     * ══════════════════════════════════════════════════════════ */
    printf("\n  === Big (cpu6) → Little (cpu0) migration ===\n\n");

    /* Already on big core from previous test, steady state */
    /* Phase 2: Migration event */
    t_pre = now_us();
    pin(0);  /* migrate to little core */
    t_post = now_us();
    syscall_us = t_post - t_pre;
    printf("  sched_setaffinity() latency: %.1f us\n\n", syscall_us);

    /* Phase 3: Warmup on little core */
    printf("  Little-core warmup curve (first %d matvecs after migration):\n", WARMUP);
    for(int i=0;i<WARMUP;i++){
        double t=now_us();
        matvec_q4_neon(out, wgt, act, N, K);sink+=out[0];
        t_warmup[i]=now_us()-t;
        printf("    matvec[%2d] = %7.1f us%s\n", i, t_warmup[i],
               i==0 ? "  ← cold cache" : "");
    }

    /* Phase 4: Steady state on little core (should match Phase 1) */
    pin(0);
    double *t_little2 = calloc(SS_ITERS, sizeof(double));
    for(int i=0;i<SS_ITERS;i++){double t=now_us();matvec_q4_neon(out,wgt,act,N,K);sink+=out[0];t_little2[i]=now_us()-t;}
    double sum_l2=0; for(int i=0;i<SS_ITERS;i++) sum_l2+=t_little2[i];
    printf("\n  Little steady-state (cpu0): %.1f us/matvec (mean of %d)\n", sum_l2/SS_ITERS, SS_ITERS);

    cold_penalty = t_warmup[0] - (sum_l2/SS_ITERS);
    transition_gap = syscall_us + cold_penalty;
    printf("\n  ── Big → Little transition cost ──\n");
    printf("  syscall:         %7.1f us\n", syscall_us);
    printf("  cold-cache tax:  %7.1f us\n", cold_penalty);
    printf("  total gap:       %7.1f us\n", transition_gap);

    printf("\n  (sink=%.1f)\n", (double)sink);  /* prevent DCE */
    printf("\n════════════════════════════════════════════════════════════\n\n");
    free(t_little); free(t_big); free(t_little2); free(t_warmup); free(act); free(out);
    munmap((void*)data,st.st_size); close(fd);
    return 0;
}
