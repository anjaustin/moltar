/*
 * gpu_matmul_probe.c — Probe C: GPU Matmul for Prefill
 *
 * During prefill (prompt evaluation), llama.cpp processes B tokens at once.
 * This turns matvec into matmul: Out[B,N] = X[B,K] × W[N,K]^T
 *
 * For LFM2-350M: K=1024, N=4608 (FFN gate), typical B=32..128 tokens.
 * CPU does B sequential matvecs at ~758us each → B*758us total.
 * GPU can parallelize across B×N output elements simultaneously.
 *
 * Tests at multiple batch sizes:
 *   1. CPU Q4_0 matvec×B (NEON SDOT, single A78)
 *   2. CPU Q4_0 matvec×B (NEON SDOT, 2×A78 RAID 0)
 *   3. GPU FP32 matmul (throughput ceiling, pre-dequantized weights)
 *   4. GPU Q4_0 matmul (real scenario, dequant on GPU)
 *
 * Build:
 *   glslc --target-env=vulkan1.1 -o matmul_f32.spv matmul_f32.comp
 *   glslc --target-env=vulkan1.1 -o q4_matmul.spv q4_matmul.comp
 *   $CC -O2 -march=armv8.2-a+dotprod -o gpu_matmul_probe gpu_matmul_probe.c -lvulkan -lm
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
#include <vulkan/vulkan.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ── Timeline semaphore KHR ── */
static PFN_vkSignalSemaphoreKHR pfn_vkSignalSemaphore;
static PFN_vkWaitSemaphoresKHR pfn_vkWaitSemaphores;
#define vkSignalSemaphore pfn_vkSignalSemaphore
#define vkWaitSemaphores pfn_vkWaitSemaphores

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
    printf("  %-44s mean=%9.1f  min=%9.1f  p50=%9.1f  max=%9.1f us\n",
           l, s/n, mn, q[n/2], mx);
}

/* ── GGUF parser ── */
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

/* ── Q4_0 matvec (NEON SDOT) ── */
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

/* ── Batch-parallel worker for 2×A78 ── */
typedef struct {
    const uint8_t *q4; const int8_t *act; float *out;
    int N, K, cpu;
    int batch_start, batch_count;
    pthread_barrier_t *bar_start, *bar_end;
    int iters;
    volatile float sink;
} BatchArg;

static void *batch_worker(void *arg) {
    BatchArg *a = (BatchArg*)arg;
    pin(a->cpu);
    float local_sink = 0;
    for (int iter = 0; iter < a->iters; iter++) {
        pthread_barrier_wait(a->bar_start);
        for (int b = 0; b < a->batch_count; b++) {
            int bi = a->batch_start + b;
            matvec_q4(a->out + bi * a->N, a->q4, a->act + bi * a->K, a->N, a->K);
            local_sink += a->out[bi * a->N];
        }
        pthread_barrier_wait(a->bar_end);
    }
    a->sink = local_sink;
    return NULL;
}

/* ── Vulkan setup (reused from Probe B) ── */
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
static Buf make_buf(Vk *v,VkDeviceSize sz,VkBufferUsageFlags usage){
    Buf b={.sz=sz};
    VkBufferCreateInfo ci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=sz,.usage=usage};
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
    VkPhysicalDeviceTimelineSemaphoreFeatures tsf={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,.timelineSemaphore=VK_TRUE};
    const char *exts[]={"VK_KHR_timeline_semaphore"};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&tsf,
        .queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=1,.ppEnabledExtensionNames=exts};
    CHECK_VK(vkCreateDevice(v->pdev,&dci,NULL,&v->dev));
    vkGetDeviceQueue(v->dev,v->qf,0,&v->queue);
    VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=v->qf};
    CHECK_VK(vkCreateCommandPool(v->dev,&pci,NULL,&v->pool));
    pfn_vkSignalSemaphore=(PFN_vkSignalSemaphoreKHR)vkGetDeviceProcAddr(v->dev,"vkSignalSemaphoreKHR");
    pfn_vkWaitSemaphores=(PFN_vkWaitSemaphoresKHR)vkGetDeviceProcAddr(v->dev,"vkWaitSemaphoresKHR");
}

/* ── GPU pipeline helper ── */
typedef struct {
    VkPipeline pipe; VkPipelineLayout layout;
    VkDescriptorSetLayout dsl; VkDescriptorPool dp; VkDescriptorSet ds;
    Buf w_buf, x_buf, out_buf;
    VkCommandBuffer cmd;
    VkFence fence;
} GpuMatmul;

static VkShaderModule load_shader(Vk *v, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sz); fread(spv, 1, sz, f); fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=spv};
    CHECK_VK(vkCreateShaderModule(v->dev, &ci, NULL, &sm)); free(spv);
    return sm;
}

static void init_gpu_matmul(Vk *v, GpuMatmul *g, const char *shader_path,
                            VkDeviceSize w_sz, VkDeviceSize x_sz, VkDeviceSize out_sz,
                            uint32_t N, uint32_t K, uint32_t B) {
    VkShaderModule sm = load_shader(v, shader_path);

    /* 3 storage buffer bindings */
    VkDescriptorSetLayoutBinding binds[3] = {
        {.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding=2,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=3,.pBindings=binds};
    CHECK_VK(vkCreateDescriptorSetLayout(v->dev, &dslci, NULL, &g->dsl));

    /* Push constants: N, K, B (3 × uint32) */
    VkPushConstantRange pcr = {.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=12};
    VkPipelineLayoutCreateInfo plci = {.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&g->dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
    CHECK_VK(vkCreatePipelineLayout(v->dev, &plci, NULL, &g->layout));

    VkComputePipelineCreateInfo cpci = {.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},
        .layout=g->layout};
    CHECK_VK(vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &g->pipe));
    vkDestroyShaderModule(v->dev, sm, NULL);

    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&dps};
    CHECK_VK(vkCreateDescriptorPool(v->dev, &dpci, NULL, &g->dp));
    VkDescriptorSetAllocateInfo dsai = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=g->dp,.descriptorSetCount=1,.pSetLayouts=&g->dsl};
    CHECK_VK(vkAllocateDescriptorSets(v->dev, &dsai, &g->ds));

    /* Buffers */
    g->w_buf   = make_buf(v, w_sz,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    g->x_buf   = make_buf(v, x_sz,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    g->out_buf = make_buf(v, out_sz,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    /* Bind to descriptor set */
    VkDescriptorBufferInfo dbis[3] = {
        {.buffer=g->w_buf.buf,.offset=0,.range=w_sz},
        {.buffer=g->x_buf.buf,.offset=0,.range=x_sz},
        {.buffer=g->out_buf.buf,.offset=0,.range=out_sz},
    };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=g->ds,.dstBinding=i,.descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbis[i]};
    }
    vkUpdateDescriptorSets(v->dev, 3, wds, 0, NULL);

    /* Command buffer */
    VkCommandBufferAllocateInfo cbai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    CHECK_VK(vkAllocateCommandBuffers(v->dev, &cbai, &g->cmd));

    /* Record: dispatch N×B workgroups */
    VkCommandBufferBeginInfo bi = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK_VK(vkBeginCommandBuffer(g->cmd, &bi));
    vkCmdBindPipeline(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->pipe);
    vkCmdBindDescriptorSets(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->layout, 0, 1, &g->ds, 0, NULL);
    uint32_t pc[3] = {N, K, B};
    vkCmdPushConstants(g->cmd, g->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, pc);
    vkCmdDispatch(g->cmd, N, B, 1);  /* N workgroups in X, B in Y */
    CHECK_VK(vkEndCommandBuffer(g->cmd));

    VkFenceCreateInfo fci = {.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK_VK(vkCreateFence(v->dev, &fci, NULL, &g->fence));
}

static double gpu_matmul_sync(Vk *v, GpuMatmul *g) {
    CHECK_VK(vkResetFences(v->dev, 1, &g->fence));
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&g->cmd};
    double t0 = now_us();
    CHECK_VK(vkQueueSubmit(v->queue, 1, &si, g->fence));
    CHECK_VK(vkWaitForFences(v->dev, 1, &g->fence, VK_TRUE, UINT64_MAX));
    return now_us() - t0;
}

/* ── Dequant Q4_0 row to FP32 ── */
static void dequant_row(float *out, const uint8_t *q4, int K) {
    int bpr = K / 32;
    for (int b = 0; b < bpr; b++) {
        uint16_t sh; memcpy(&sh, q4, 2);
        float sc = f16_to_f32(sh);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = q4[2 + j];
            int lo = (byte & 0x0F) - 8;
            int hi = (byte >> 4) - 8;
            out[b * 32 + j * 2]     = sc * (float)lo;
            out[b * 32 + j * 2 + 1] = sc * (float)hi;
        }
        q4 += 18;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    /* Map model */
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TI ti; const uint8_t *wgt;
    if (find_tensor(data, "blk.0.ffn_gate.weight", &ti, &wgt) != 0) {
        fprintf(stderr, "tensor not found\n"); return 1; }
    int K = (int)ti.dims[0], N = (int)ti.dims[1];

    printf("\n════════════════════════════════════════════════════════════════════\n");
    printf("  PROBE C: GPU Matmul for Prefill\n");
    printf("  Weight matrix: blk.0.ffn_gate [%d×%d] Q4_0\n", N, K);
    printf("  Question: For batched prefill, is GPU matmul faster than CPU?\n");
    printf("════════════════════════════════════════════════════════════════════\n");

    /* Batch sizes to test */
    int batches[] = {1, 8, 16, 32, 64, 128};
    int n_batches = sizeof(batches) / sizeof(batches[0]);
    int max_B = 128;

    int WARMUP = 10;
    int ITERS = 50;

    /* Prepare activations: random int8 for CPU, random float for GPU */
    srand(42);
    int8_t *act_i8 = calloc(max_B * K, 1);
    float  *act_f32 = calloc(max_B * K, sizeof(float));
    for (int i = 0; i < max_B * K; i++) {
        act_i8[i] = (int8_t)(rand() % 256 - 128);
        act_f32[i] = (float)act_i8[i] / 64.0f;  /* scale to reasonable range */
    }
    float *cpu_out = calloc(max_B * N, sizeof(float));
    volatile float sink = 0;

    /* ═══════════════════════════════════════════════════════
     * Init Vulkan
     * ═══════════════════════════════════════════════════════ */
    Vk v; init_vk(&v);

    /* Dequant weights to FP32 for the FP32 GPU test */
    int bpr = K / 32;
    size_t q4_row_bytes = (size_t)bpr * 18;
    float *w_f32 = calloc((size_t)N * K, sizeof(float));
    for (int n = 0; n < N; n++) {
        dequant_row(w_f32 + (size_t)n * K, wgt + (size_t)n * q4_row_bytes, K);
    }

    printf("\n  Batch sizes: 1, 8, 16, 32, 64, 128\n");
    printf("  WARMUP=%d, ITERS=%d per batch size\n\n", WARMUP, ITERS);

    /* ═══════════════════════════════════════════════════════
     * TEST 1: CPU matvec×B (single A78)
     * ═══════════════════════════════════════════════════════ */
    printf("  ── Test 1: CPU Q4_0 matvec×B (single A78, cpu6) ──\n");
    pin(6);
    for (int bi = 0; bi < n_batches; bi++) {
        int B = batches[bi];
        char label[64]; snprintf(label, 64, "CPU 1×A78, B=%d", B);

        /* Warmup */
        for (int w = 0; w < WARMUP; w++) {
            for (int b = 0; b < B; b++) {
                matvec_q4(cpu_out + b * N, wgt, act_i8 + b * K, N, K);
                sink += cpu_out[b * N];
            }
        }
        /* Measure */
        double *times = calloc(ITERS, sizeof(double));
        for (int it = 0; it < ITERS; it++) {
            double t0 = now_us();
            for (int b = 0; b < B; b++) {
                matvec_q4(cpu_out + b * N, wgt, act_i8 + b * K, N, K);
                sink += cpu_out[b * N];
            }
            times[it] = now_us() - t0;
        }
        pstats(label, times, ITERS);
        free(times);
    }

    /* ═══════════════════════════════════════════════════════
     * TEST 2: CPU batch-parallel (2×A78, split B across cores)
     * This is what llama.cpp -t 2 actually does for prefill.
     * ═══════════════════════════════════════════════════════ */
    printf("\n  ── Test 2: CPU Q4_0 2×A78 batch-parallel (split B) ──\n");
    for (int bi = 0; bi < n_batches; bi++) {
        int B = batches[bi];
        if (B < 2) {
            printf("  %-44s (skipped, B=1 can't split)\n", "CPU 2×A78 batch, B=1");
            continue;
        }
        char label[64]; snprintf(label, 64, "CPU 2×A78 batch, B=%d", B);

        pthread_barrier_t bar_s, bar_e;
        pthread_barrier_init(&bar_s, NULL, 2);
        pthread_barrier_init(&bar_e, NULL, 2);

        int total_iters = WARMUP + ITERS;
        int half_B = B / 2;

        /* Worker 1 handles second half of batch on cpu7 */
        BatchArg a1 = {wgt, act_i8, cpu_out, N, K, 7,
                       half_B, B - half_B, &bar_s, &bar_e, total_iters, 0};
        pthread_t th;
        pthread_create(&th, NULL, batch_worker, &a1);

        /* Main thread handles first half on cpu6 */
        pin(6);
        double *times = calloc(ITERS, sizeof(double));
        int time_idx = 0;

        for (int iter = 0; iter < total_iters; iter++) {
            double t0 = now_us();
            pthread_barrier_wait(&bar_s);
            for (int b = 0; b < half_B; b++) {
                matvec_q4(cpu_out + b * N, wgt, act_i8 + b * K, N, K);
                sink += cpu_out[b * N];
            }
            pthread_barrier_wait(&bar_e);
            double dt = now_us() - t0;
            if (iter >= WARMUP) times[time_idx++] = dt;
        }
        pthread_join(th, NULL);
        sink += a1.sink;
        pstats(label, times, ITERS);
        free(times);
        pthread_barrier_destroy(&bar_s); pthread_barrier_destroy(&bar_e);
    }

    /* ═══════════════════════════════════════════════════════
     * TEST 3: GPU FP32 matmul (throughput ceiling)
     * ═══════════════════════════════════════════════════════ */
    printf("\n  ── Test 3: GPU FP32 matmul (throughput ceiling) ──\n");
    for (int bi = 0; bi < n_batches; bi++) {
        int B = batches[bi];
        char label[64]; snprintf(label, 64, "GPU FP32, B=%d", B);

        VkDeviceSize w_sz = (VkDeviceSize)N * K * sizeof(float);
        VkDeviceSize x_sz = (VkDeviceSize)B * K * sizeof(float);
        VkDeviceSize o_sz = (VkDeviceSize)B * N * sizeof(float);

        GpuMatmul g;
        init_gpu_matmul(&v, &g, "matmul_f32.spv", w_sz, x_sz, o_sz, N, K, B);

        /* Copy weights and activations */
        memcpy(g.w_buf.map, w_f32, (size_t)N * K * sizeof(float));
        memcpy(g.x_buf.map, act_f32, (size_t)B * K * sizeof(float));

        /* Warmup */
        for (int w = 0; w < WARMUP; w++) gpu_matmul_sync(&v, &g);

        /* Measure */
        double *times = calloc(ITERS, sizeof(double));
        for (int it = 0; it < ITERS; it++) {
            times[it] = gpu_matmul_sync(&v, &g);
        }
        pstats(label, times, ITERS);
        free(times);

        /* Cleanup this batch's pipeline */
        vkDestroyFence(v.dev, g.fence, NULL);
        vkDestroyPipeline(v.dev, g.pipe, NULL);
        vkDestroyPipelineLayout(v.dev, g.layout, NULL);
        vkDestroyDescriptorPool(v.dev, g.dp, NULL);
        vkDestroyDescriptorSetLayout(v.dev, g.dsl, NULL);
        vkDestroyBuffer(v.dev, g.w_buf.buf, NULL); vkFreeMemory(v.dev, g.w_buf.mem, NULL);
        vkDestroyBuffer(v.dev, g.x_buf.buf, NULL); vkFreeMemory(v.dev, g.x_buf.mem, NULL);
        vkDestroyBuffer(v.dev, g.out_buf.buf, NULL); vkFreeMemory(v.dev, g.out_buf.mem, NULL);
    }

    /* ═══════════════════════════════════════════════════════
     * TEST 4: GPU Q4_0 matmul (real scenario)
     * ═══════════════════════════════════════════════════════ */
    printf("\n  ── Test 4: GPU Q4_0 matmul (dequant on GPU) ──\n");
    for (int bi = 0; bi < n_batches; bi++) {
        int B = batches[bi];
        char label[64]; snprintf(label, 64, "GPU Q4_0, B=%d", B);

        VkDeviceSize w_sz = (VkDeviceSize)N * q4_row_bytes;  /* raw Q4_0 bytes */
        VkDeviceSize x_sz = (VkDeviceSize)B * K * sizeof(float);
        VkDeviceSize o_sz = (VkDeviceSize)B * N * sizeof(float);

        GpuMatmul g;
        init_gpu_matmul(&v, &g, "q4_matmul.spv", w_sz, x_sz, o_sz, N, K, B);

        /* Copy raw Q4_0 weights and float activations */
        memcpy(g.w_buf.map, wgt, (size_t)N * q4_row_bytes);
        memcpy(g.x_buf.map, act_f32, (size_t)B * K * sizeof(float));

        /* Warmup */
        for (int w = 0; w < WARMUP; w++) gpu_matmul_sync(&v, &g);

        /* Measure */
        double *times = calloc(ITERS, sizeof(double));
        for (int it = 0; it < ITERS; it++) {
            times[it] = gpu_matmul_sync(&v, &g);
        }
        pstats(label, times, ITERS);
        free(times);

        /* Cleanup */
        vkDestroyFence(v.dev, g.fence, NULL);
        vkDestroyPipeline(v.dev, g.pipe, NULL);
        vkDestroyPipelineLayout(v.dev, g.layout, NULL);
        vkDestroyDescriptorPool(v.dev, g.dp, NULL);
        vkDestroyDescriptorSetLayout(v.dev, g.dsl, NULL);
        vkDestroyBuffer(v.dev, g.w_buf.buf, NULL); vkFreeMemory(v.dev, g.w_buf.mem, NULL);
        vkDestroyBuffer(v.dev, g.x_buf.buf, NULL); vkFreeMemory(v.dev, g.x_buf.mem, NULL);
        vkDestroyBuffer(v.dev, g.out_buf.buf, NULL); vkFreeMemory(v.dev, g.out_buf.mem, NULL);
    }

    /* ═══════════════════════════════════════════════════════
     * Summary Table
     * ═══════════════════════════════════════════════════════ */
    printf("\n  ── Breakeven Analysis ──\n");
    printf("  CPU 1×A78 matvec = ~758us per token.\n");
    printf("  CPU RAID0 2×A78  = ~403us per token.\n");
    printf("  GPU dispatch overhead = ~368us (Probe B sync), ~160us (timeline).\n");
    printf("  If GPU matmul scales with B, crossover is where GPU_time < CPU_time × B.\n");
    printf("  (sink=%.1f)\n", (double)sink);

    printf("\n════════════════════════════════════════════════════════════════════\n\n");

    free(w_f32); free(act_i8); free(act_f32); free(cpu_out);
    munmap((void*)data, st.st_size); close(fd);
    return 0;
}
