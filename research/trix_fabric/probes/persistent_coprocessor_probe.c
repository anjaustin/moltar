/*
 * persistent_coprocessor_probe.c — PowerVR as Always-On Fabric Coprocessor
 *
 * Instead of dispatch-per-call, the GPU runs a PERSISTENT shader that
 * spins waiting for commands via mapped coherent memory. Zero dispatch
 * overhead — communication is just an atomic store + spin-wait.
 *
 * Measures:
 *   1. NOP round-trip: CPU writes flag → GPU sees it → GPU writes done → CPU sees it
 *   2. SiLU activation on 1024 elements (D_MODEL) entirely in shared memory
 *   3. RMSNorm on 1024 elements entirely in shared memory
 *   4. Concurrent: CPU matvec running WHILE GPU does activation (bus contention test)
 *
 * Build:
 *   glslc --target-env=vulkan1.1 -o persistent_coprocessor.spv persistent_coprocessor.comp
 *   $CC -O2 -march=armv8.2-a+dotprod -o persistent_coprocessor_probe persistent_coprocessor_probe.c -lvulkan -lm
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
#include <vulkan/vulkan.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define CHECK_VK(call) do { VkResult _r = (call); \
    if (_r != VK_SUCCESS) { fprintf(stderr, "VK error %d at %s:%d\n", _r, __FILE__, __LINE__); exit(1); } } while(0)

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
static void pin(int cpu) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}

/* ── Stats ── */
static int cmp_dbl(const void *a, const void *b) {
    double da=*(const double*)a, db=*(const double*)b; return (da>db)-(da<db); }
static void pstats(const char *l, double *t, int n) {
    double s=0,mn=1e18,mx=0;
    for(int i=0;i<n;i++){s+=t[i];if(t[i]<mn)mn=t[i];if(t[i]>mx)mx=t[i];}
    double q[n]; memcpy(q,t,n*sizeof(double)); qsort(q,n,sizeof(double),cmp_dbl);
    printf("  %-44s mean=%7.1f  min=%7.1f  p50=%7.1f  p99=%7.1f  max=%7.1f us\n",
           l, s/n, mn, q[n/2], q[(int)(n*0.99)], mx);
}

/* ── GGUF parser (for concurrent matvec test) ── */
#define GGUF_TYPE_Q4_0 2
typedef struct { char name[256]; uint32_t n_dims; uint64_t dims[4]; uint32_t type; uint64_t offset; } TI;
static const uint8_t *skip_str(const uint8_t *p){return p+8+*(const uint64_t*)p;}
static void read_str(const uint8_t *p,char *o,int m){uint64_t l=*(const uint64_t*)p;int c=l<(uint64_t)m-1?(int)l:m-1;memcpy(o,p+8,c);o[c]=0;}
static const uint8_t *skip_val(const uint8_t *p,uint32_t t){
    switch(t){case 0:case 1:case 7:return p+1;case 2:case 3:return p+2;case 4:case 5:case 6:return p+4;
    case 10:case 11:case 12:return p+8;case 8:return skip_str(p);
    case 9:{uint32_t et=*(const uint32_t*)p;p+=4;uint64_t n=*(const uint64_t*)p;p+=8;for(uint64_t i=0;i<n;i++)p=skip_val(p,et);return p;}
    default:return p;}
}
static int find_tensor(const uint8_t *data,const char *name,TI *out,const uint8_t **td){
    const uint8_t *p=data+8;uint64_t nt=*(const uint64_t*)p;p+=8;uint64_t nk=*(const uint64_t*)p;p+=8;
    for(uint64_t i=0;i<nk;i++){p=skip_str(p);uint32_t vt=*(const uint32_t*)p;p+=4;p=skip_val(p,vt);}
    TI *all=calloc(nt,sizeof(TI));
    for(uint64_t i=0;i<nt;i++){read_str(p,all[i].name,256);p=skip_str(p);all[i].n_dims=*(const uint32_t*)p;p+=4;
        for(uint32_t d=0;d<all[i].n_dims;d++){all[i].dims[d]=*(const uint64_t*)p;p+=8;}all[i].type=*(const uint32_t*)p;p+=4;all[i].offset=*(const uint64_t*)p;p+=8;}
    uint64_t base=((p-data)+31)&~31ULL;int found=-1;
    for(uint64_t i=0;i<nt;i++){if(strcmp(all[i].name,name)==0&&all[i].type==GGUF_TYPE_Q4_0){*out=all[i];found=0;break;}}
    if(found==0)*td=data+base+out->offset;free(all);return found;
}

/* ── Q4_0 matvec ── */
static float f16_to_f32(uint16_t h){
    uint32_t s=(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,f;
    if(!e){if(!m)f=s;else{e=1;while(!(m&0x400)){m<<=1;e--;}m&=0x3FF;f=s|((e+112)<<23)|(m<<13);}}
    else if(e==31)f=s|0x7F800000|(m<<13);else f=s|((e+112)<<23)|(m<<13);
    float r;memcpy(&r,&f,4);return r;
}
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static void matvec_q4(float *out,const uint8_t *q4,const int8_t *act,int N,int K){
    const int bpr=K/32,bs=18;const int8x16_t eight=vdupq_n_s8(8);
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

/* ── Vulkan setup ── */
typedef struct {
    VkInstance inst; VkPhysicalDevice pdev; VkDevice dev;
    uint32_t qf; VkQueue queue; VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties mprops;
} Vk;
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; VkDeviceSize sz; } Buf;

static uint32_t find_mem(Vk *v,uint32_t bits,VkMemoryPropertyFlags f){
    for(uint32_t i=0;i<v->mprops.memoryTypeCount;i++)
        if((bits&(1u<<i))&&(v->mprops.memoryTypes[i].propertyFlags&f)==f) return i;
    fprintf(stderr,"no mem\n");exit(1);
}
static Buf make_buf(Vk *v,VkDeviceSize sz){
    Buf b={.sz=sz};
    VkBufferCreateInfo ci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=sz,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    CHECK_VK(vkCreateBuffer(v->dev,&ci,NULL,&b.buf));
    VkMemoryRequirements req;vkGetBufferMemoryRequirements(v->dev,b.buf,&req);
    VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=req.size,
        .memoryTypeIndex=find_mem(v,req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    CHECK_VK(vkAllocateMemory(v->dev,&ai,NULL,&b.mem));
    CHECK_VK(vkBindBufferMemory(v->dev,b.buf,b.mem,0));
    CHECK_VK(vkMapMemory(v->dev,b.mem,0,sz,0,&b.map));
    return b;
}
static void init_vk(Vk *v){
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    CHECK_VK(vkCreateInstance(&ici,NULL,&v->inst));
    uint32_t n=1;CHECK_VK(vkEnumeratePhysicalDevices(v->inst,&n,&v->pdev));
    vkGetPhysicalDeviceMemoryProperties(v->pdev,&v->mprops);
    uint32_t qfc=0;vkGetPhysicalDeviceQueueFamilyProperties(v->pdev,&qfc,NULL);
    VkQueueFamilyProperties *qfp=malloc(qfc*sizeof(*qfp));
    vkGetPhysicalDeviceQueueFamilyProperties(v->pdev,&qfc,qfp);
    v->qf=UINT32_MAX;for(uint32_t i=0;i<qfc;i++)if(qfp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){v->qf=i;break;}free(qfp);
    float prio=1.0f;
    VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=v->qf,.queueCount=1,.pQueuePriorities=&prio};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
    CHECK_VK(vkCreateDevice(v->pdev,&dci,NULL,&v->dev));
    vkGetDeviceQueue(v->dev,v->qf,0,&v->queue);
    VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=v->qf};
    CHECK_VK(vkCreateCommandPool(v->dev,&pci,NULL,&v->pool));
}

/* ── Issue command to persistent coprocessor ── */
static inline void coprocessor_cmd(volatile uint32_t *cmd, uint32_t op, uint32_t n_elem) {
    cmd[1] = op;
    cmd[2] = n_elem;
    __sync_synchronize();  /* full memory barrier */
    cmd[0] = 1;            /* signal: go */
}

/* ── Wait for coprocessor completion ── */
static inline void coprocessor_wait(volatile uint32_t *cmd) {
    while (cmd[0] != 2) {
        /* spin — could add sched_yield() but that adds latency */
        __sync_synchronize();
    }
    /* Reset to idle for next command */
    cmd[0] = 0;
    __sync_synchronize();
}

/* ── Round-trip: issue + wait ── */
static inline double coprocessor_roundtrip(volatile uint32_t *cmd, uint32_t op, uint32_t n_elem) {
    double t0 = now_us();
    coprocessor_cmd(cmd, op, n_elem);
    coprocessor_wait(cmd);
    return now_us() - t0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    /* Map model for concurrent matvec test */
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TI ti; const uint8_t *wgt;
    find_tensor(data, "blk.0.ffn_gate.weight", &ti, &wgt);
    int K = (int)ti.dims[0], N = (int)ti.dims[1];

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  PERSISTENT COPROCESSOR PROBE\n");
    printf("  PowerVR BXM-8-256 as always-on fabric chip\n");
    printf("  Zero dispatch — communication via mapped coherent atomics\n");
    printf("══════════════════════════════════════════════════════════════\n");

    pin(6);

    /* Init Vulkan */
    Vk v; init_vk(&v);

    /* Create buffers */
    Buf ctrl_buf = make_buf(&v, 256);       /* control words */
    Buf data_buf = make_buf(&v, 32768);     /* 8K floats = 32KB (data + weights) */

    volatile uint32_t *cmd = (volatile uint32_t *)ctrl_buf.map;
    volatile float *gpu_data = (volatile float *)data_buf.map;

    /* Clear control */
    memset(ctrl_buf.map, 0, 256);

    /* Fill data buffer with test values */
    srand(42);
    for (int i = 0; i < 2048; i++) {
        ((float*)data_buf.map)[i] = ((float)rand() / (float)0x7fffffff - 0.5f) * 4.0f;
    }

    /* Load shader */
    FILE *f = fopen("persistent_coprocessor.spv", "rb");
    if (!f) { fprintf(stderr, "Cannot open persistent_coprocessor.spv\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sz); fread(spv, 1, sz, f); fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=spv};
    CHECK_VK(vkCreateShaderModule(v.dev, &smci, NULL, &sm)); free(spv);

    /* Descriptor layout: 2 storage buffers (ctrl + data) */
    VkDescriptorSetLayoutBinding binds[2] = {
        {.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=2,.pBindings=binds};
    VkDescriptorSetLayout dsl;
    CHECK_VK(vkCreateDescriptorSetLayout(v.dev, &dslci, NULL, &dsl));

    VkPipelineLayoutCreateInfo plci = {.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&dsl};
    VkPipelineLayout pl;
    CHECK_VK(vkCreatePipelineLayout(v.dev, &plci, NULL, &pl));

    VkComputePipelineCreateInfo cpci = {.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},
        .layout=pl};
    VkPipeline pipe;
    CHECK_VK(vkCreateComputePipelines(v.dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    vkDestroyShaderModule(v.dev, sm, NULL);

    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&dps};
    VkDescriptorPool dp;
    CHECK_VK(vkCreateDescriptorPool(v.dev, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl};
    VkDescriptorSet ds;
    CHECK_VK(vkAllocateDescriptorSets(v.dev, &dsai, &ds));

    VkDescriptorBufferInfo dbis[2] = {
        {.buffer=ctrl_buf.buf,.offset=0,.range=256},
        {.buffer=data_buf.buf,.offset=0,.range=32768},
    };
    VkWriteDescriptorSet wds[2];
    for (int i = 0; i < 2; i++) {
        wds[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=ds,.dstBinding=i,.descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbis[i]};
    }
    vkUpdateDescriptorSets(v.dev, 2, wds, 0, NULL);

    /* Record and launch the persistent shader — it will spin forever */
    VkCommandBufferAllocateInfo cbai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v.pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkCommandBuffer cmdbuf;
    CHECK_VK(vkAllocateCommandBuffers(v.dev, &cbai, &cmdbuf));

    VkCommandBufferBeginInfo bi = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK_VK(vkBeginCommandBuffer(cmdbuf, &bi));
    vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmdbuf, 1, 1, 1);  /* single workgroup of 128 threads */
    CHECK_VK(vkEndCommandBuffer(cmdbuf));

    /* Launch — fire and forget. Shader is now spinning. */
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmdbuf};
    CHECK_VK(vkQueueSubmit(v.queue, 1, &si, VK_NULL_HANDLE));

    printf("  Persistent shader launched. Testing communication...\n\n");

    /* Give GPU time to start spinning */
    usleep(10000);

    int WARMUP = 100;
    int ITERS = 1000;

    /* ═══════════════════════════════════════════════════════
     * TEST 1: NOP round-trip latency
     * ═══════════════════════════════════════════════════════ */
    printf("  === Test 1: NOP round-trip (flag flip only) ===\n");
    for (int i = 0; i < WARMUP; i++) coprocessor_roundtrip(cmd, 0, 0);
    double *t = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) t[i] = coprocessor_roundtrip(cmd, 0, 0);
    pstats("NOP round-trip", t, ITERS);

    /* ═══════════════════════════════════════════════════════
     * TEST 2: SiLU on 1024 elements (D_MODEL)
     * ═══════════════════════════════════════════════════════ */
    printf("\n  === Test 2: SiLU activation (1024 elements, shared mem) ===\n");
    for (int i = 0; i < WARMUP; i++) coprocessor_roundtrip(cmd, 2, 1024);
    for (int i = 0; i < ITERS; i++) t[i] = coprocessor_roundtrip(cmd, 2, 1024);
    pstats("SiLU 1024 (persistent)", t, ITERS);

    /* ═══════════════════════════════════════════════════════
     * TEST 3: RMSNorm on 1024 elements
     * ═══════════════════════════════════════════════════════ */
    printf("\n  === Test 3: RMSNorm (1024 elements, shared mem) ===\n");
    /* Set up weights in data_buf[1024..2047] */
    for (int i = 1024; i < 2048; i++) ((float*)data_buf.map)[i] = 1.0f;
    for (int i = 0; i < WARMUP; i++) coprocessor_roundtrip(cmd, 3, 1024);
    for (int i = 0; i < ITERS; i++) t[i] = coprocessor_roundtrip(cmd, 3, 1024);
    pstats("RMSNorm 1024 (persistent)", t, ITERS);

    /* ═══════════════════════════════════════════════════════
     * TEST 4: Reduce-sum on 1024 elements
     * ═══════════════════════════════════════════════════════ */
    printf("\n  === Test 4: Reduce-sum (1024 elements) ===\n");
    for (int i = 0; i < WARMUP; i++) coprocessor_roundtrip(cmd, 1, 1024);
    for (int i = 0; i < ITERS; i++) t[i] = coprocessor_roundtrip(cmd, 1, 1024);
    pstats("Reduce-sum 1024 (persistent)", t, ITERS);

    /* ═══════════════════════════════════════════════════════
     * TEST 5: Concurrent — GPU SiLU while CPU does matvec
     * KEY TEST: does the persistent GPU work steal bus bandwidth?
     * ═══════════════════════════════════════════════════════ */
    printf("\n  === Test 5: Concurrent CPU matvec + GPU SiLU ===\n");
    int8_t *act = calloc(K, 1);
    float *cpu_out = calloc(N, sizeof(float));
    volatile float sink = 0;
    for (int i = 0; i < K; i++) act[i] = (int8_t)(rand() % 256 - 128);

    /* 5a: CPU matvec alone (baseline) */
    for (int i = 0; i < 20; i++) { matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0]; }
    double *t_cpu = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS/2; i++) {
        double t0 = now_us();
        matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0];
        t_cpu[i] = now_us() - t0;
    }
    pstats("CPU matvec alone (baseline)", t_cpu, ITERS/2);

    /* 5b: CPU matvec while GPU does SiLU continuously */
    /* Kick GPU SiLU, do CPU matvec, wait GPU, repeat */
    double *t_concurrent = calloc(ITERS, sizeof(double));
    double *t_gpu_only = calloc(ITERS, sizeof(double));
    for (int i = 0; i < 20; i++) coprocessor_roundtrip(cmd, 2, 1024); /* warmup */
    for (int i = 0; i < ITERS/2; i++) {
        double t0 = now_us();
        /* Kick GPU */
        coprocessor_cmd(cmd, 2, 1024);
        /* CPU matvec runs concurrently */
        matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0];
        double t_cpu_done = now_us();
        /* Wait for GPU to finish (may already be done) */
        coprocessor_wait(cmd);
        double t_all = now_us();
        t_concurrent[i] = t_all - t0;
        t_gpu_only[i] = t_all - t_cpu_done; /* extra wait after CPU done */
    }
    pstats("Concurrent (CPU matvec + GPU SiLU)", t_concurrent, ITERS/2);
    pstats("GPU extra wait after CPU done", t_gpu_only, ITERS/2);

    /* ═══════════════════════════════════════════════════════
     * Summary
     * ═══════════════════════════════════════════════════════ */
    printf("\n  ── Summary ──\n");
    double nop_mean = 0, silu_mean = 0;
    for (int i = 0; i < ITERS; i++) { nop_mean += t[i]; }  /* t still has reduce-sum, but let's just report NOP */
    printf("  Compare vs dispatch path: Probe A timeline = 160us min\n");
    printf("  Compare vs CPU: SiLU(1024) on CPU = ~6us, RMSNorm(1024) on CPU = ~1us\n");
    printf("  (sink=%.1f, jobs=%u)\n", (double)sink, cmd[3]);

    printf("\n══════════════════════════════════════════════════════════════\n\n");

    /* Shutdown the persistent shader */
    cmd[0] = 0xDEAD;
    __sync_synchronize();
    usleep(50000);  /* give GPU time to see it and exit */
    vkDeviceWaitIdle(v.dev);

    free(t); free(t_cpu); free(t_concurrent); free(t_gpu_only);
    free(act); free(cpu_out);
    munmap((void*)data, st.st_size); close(fd);
    return 0;
}
