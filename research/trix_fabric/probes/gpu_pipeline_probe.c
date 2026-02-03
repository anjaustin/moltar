/*
 * gpu_pipeline_probe.c — Probe B: Async CPU+GPU Pipeline Overlap
 *
 * Simulates the LFM2 FFN pipeline:
 *   CPU: Q4_0 matvec (gate projection, 1024→4608)
 *   GPU: SwiGLU activation (SiLU(gate) * up, 4608 elements)
 *
 * Measures 4 scenarios:
 *   1. CPU matvec only (baseline)
 *   2. GPU SwiGLU only (dispatch + compute)
 *   3. Sequential: CPU matvec then GPU SwiGLU
 *   4. Overlapped: CPU matvec(layer N) || GPU SwiGLU(layer N-1)
 *
 * If overlapped ≈ CPU-only → GPU activations are FREE (hidden behind matvec)
 *
 * Build:
 *   glslc --target-env=vulkan1.1 -o swiglu.spv swiglu.comp
 *   $CC -O2 -march=armv8.2-a+dotprod -o gpu_pipeline_probe gpu_pipeline_probe.c -lvulkan -lm
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
    printf("  %-44s mean=%7.1f  min=%7.1f  p50=%7.1f  p99=%7.1f  max=%7.1f us\n",
           l, s/n, mn, q[n/2], q[(int)(n*0.99)], mx);
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
    VkQueueFamilyProperties *qf=malloc(qfc*sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(v->pdev,&qfc,qf);
    v->qf=UINT32_MAX;for(uint32_t i=0;i<qfc;i++)if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){v->qf=i;break;}free(qf);
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

/* ── GPU SwiGLU pipeline ── */
typedef struct {
    VkPipeline pipe; VkPipelineLayout layout;
    VkDescriptorSetLayout dsl; VkDescriptorPool dp; VkDescriptorSet ds;
    Buf gate_buf, up_buf, out_buf;
    VkCommandBuffer cmd;
    VkSemaphore timeline;
    VkFence fence;
    uint64_t counter;
    uint32_t n_elements;
} GpuSwiglu;

static void init_swiglu(Vk *v, GpuSwiglu *g, uint32_t n_elements) {
    g->n_elements = n_elements;
    g->counter = 0;

    /* Load shader */
    FILE *f = fopen("swiglu.spv", "rb");
    if (!f) { fprintf(stderr, "Cannot open swiglu.spv\n"); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sz); fread(spv, 1, sz, f); fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=spv};
    CHECK_VK(vkCreateShaderModule(v->dev, &smci, NULL, &sm)); free(spv);

    /* Descriptor layout: 3 storage buffers */
    VkDescriptorSetLayoutBinding binds[3] = {
        {.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding=2,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=3,.pBindings=binds};
    CHECK_VK(vkCreateDescriptorSetLayout(v->dev, &dslci, NULL, &g->dsl));

    /* Push constant for n_elements */
    VkPushConstantRange pcr = {.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=4};
    VkPipelineLayoutCreateInfo plci = {.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&g->dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
    CHECK_VK(vkCreatePipelineLayout(v->dev, &plci, NULL, &g->layout));

    /* Pipeline */
    VkComputePipelineCreateInfo cpci = {.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},
        .layout=g->layout};
    CHECK_VK(vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &g->pipe));
    vkDestroyShaderModule(v->dev, sm, NULL);

    /* Descriptor pool + set */
    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&dps};
    CHECK_VK(vkCreateDescriptorPool(v->dev, &dpci, NULL, &g->dp));
    VkDescriptorSetAllocateInfo dsai = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=g->dp,.descriptorSetCount=1,.pSetLayouts=&g->dsl};
    CHECK_VK(vkAllocateDescriptorSets(v->dev, &dsai, &g->ds));

    /* Buffers */
    VkDeviceSize bsz = n_elements * sizeof(float);
    g->gate_buf = make_buf(v, bsz);
    g->up_buf   = make_buf(v, bsz);
    g->out_buf  = make_buf(v, bsz);

    /* Fill with test data */
    float *gate = (float*)g->gate_buf.map;
    float *up   = (float*)g->up_buf.map;
    for (uint32_t i = 0; i < n_elements; i++) {
        gate[i] = ((float)rand()/RAND_MAX - 0.5f) * 4.0f;
        up[i]   = ((float)rand()/RAND_MAX - 0.5f) * 4.0f;
    }

    /* Bind buffers to descriptor set */
    VkDescriptorBufferInfo dbis[3] = {
        {.buffer=g->gate_buf.buf,.offset=0,.range=bsz},
        {.buffer=g->up_buf.buf,.offset=0,.range=bsz},
        {.buffer=g->out_buf.buf,.offset=0,.range=bsz},
    };
    VkWriteDescriptorSet wds[3];
    for (int i = 0; i < 3; i++) {
        wds[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=g->ds,.dstBinding=i,.descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbis[i]};
    }
    vkUpdateDescriptorSets(v->dev, 3, wds, 0, NULL);

    /* Pre-record command buffer */
    VkCommandBufferAllocateInfo cbai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    CHECK_VK(vkAllocateCommandBuffers(v->dev, &cbai, &g->cmd));

    VkCommandBufferBeginInfo bi = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK_VK(vkBeginCommandBuffer(g->cmd, &bi));
    vkCmdBindPipeline(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->pipe);
    vkCmdBindDescriptorSets(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->layout, 0, 1, &g->ds, 0, NULL);
    vkCmdPushConstants(g->cmd, g->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n_elements);
    uint32_t groups = (n_elements + 127) / 128;  /* local_size_x = 128 */
    vkCmdDispatch(g->cmd, groups, 1, 1);
    CHECK_VK(vkEndCommandBuffer(g->cmd));

    /* Timeline semaphore */
    VkSemaphoreTypeCreateInfo stci = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE,.initialValue=0};
    VkSemaphoreCreateInfo sci = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,.pNext=&stci};
    CHECK_VK(vkCreateSemaphore(v->dev, &sci, NULL, &g->timeline));

    /* Fence for non-timeline path */
    VkFenceCreateInfo fci = {.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK_VK(vkCreateFence(v->dev, &fci, NULL, &g->fence));
}

/* Fire GPU SwiGLU and wait (fence path) */
static double gpu_swiglu_sync(Vk *v, GpuSwiglu *g) {
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&g->cmd};
    CHECK_VK(vkResetFences(v->dev, 1, &g->fence));
    double t0 = now_us();
    CHECK_VK(vkQueueSubmit(v->queue, 1, &si, g->fence));
    CHECK_VK(vkWaitForFences(v->dev, 1, &g->fence, VK_TRUE, UINT64_MAX));
    return now_us() - t0;
}

/* Pre-submit GPU work (waits on timeline), returns immediately */
static void gpu_swiglu_presubmit(Vk *v, GpuSwiglu *g) {
    uint64_t wait_val = g->counter;
    uint64_t sig_val = g->counter + 1;
    VkTimelineSemaphoreSubmitInfo tssi = {.sType=VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount=1,.pWaitSemaphoreValues=&wait_val,
        .signalSemaphoreValueCount=1,.pSignalSemaphoreValues=&sig_val};
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.pNext=&tssi,
        .waitSemaphoreCount=1,.pWaitSemaphores=&g->timeline,.pWaitDstStageMask=&ws,
        .commandBufferCount=1,.pCommandBuffers=&g->cmd,
        .signalSemaphoreCount=1,.pSignalSemaphores=&g->timeline};
    CHECK_VK(vkQueueSubmit(v->queue, 1, &si, VK_NULL_HANDLE));
}

/* Signal GPU to start (kick timeline) */
static void gpu_swiglu_kick(Vk *v, GpuSwiglu *g) {
    VkSemaphoreSignalInfo ssi = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore=g->timeline,.value=g->counter};
    CHECK_VK(vkSignalSemaphore(v->dev, &ssi));
}

/* Wait for GPU completion */
static void gpu_swiglu_wait(Vk *v, GpuSwiglu *g) {
    uint64_t val = g->counter + 1;
    VkSemaphoreWaitInfo swi = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount=1,.pSemaphores=&g->timeline,.pValues=&val};
    CHECK_VK(vkWaitSemaphores(v->dev, &swi, UINT64_MAX));
    g->counter = val;
}

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    /* Map model */
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TI ti; const uint8_t *wgt;
    if (find_tensor(data, "blk.0.ffn_gate.weight", &ti, &wgt) != 0) {
        fprintf(stderr, "tensor not found\n"); return 1; }
    int K = (int)ti.dims[0], N = (int)ti.dims[1];  /* N=4608, K=1024 */

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  PROBE B: Async CPU+GPU Pipeline Overlap\n");
    printf("  CPU: Q4_0 matvec [%d×%d] on A78 big cores\n", N, K);
    printf("  GPU: SwiGLU [%d elements] on PowerVR BXM-8-256\n", N);
    printf("  Question: Is GPU activation FREE when overlapped with CPU matvec?\n");
    printf("════════════════════════════════════════════════════════════\n");

    int8_t *act = calloc(K, 1);
    float *cpu_out = calloc(N, sizeof(float));
    volatile float sink = 0;
    srand(42);
    for (int i = 0; i < K; i++) act[i] = (int8_t)(((float)rand()/RAND_MAX - 0.5f) * 128);

    /* Init Vulkan + SwiGLU pipeline */
    Vk v; init_vk(&v);
    GpuSwiglu g; init_swiglu(&v, &g, N);

    int WARMUP = 20;
    int ITERS = 200;

    /* ═══════════════════════════════════════════════════════
     * TEST 1: CPU matvec only (pinned to cpu6, single A78)
     * ═══════════════════════════════════════════════════════ */
    printf("\n  === Test 1: CPU matvec only (1×A78, cpu6) ===\n");
    pin(6);
    for (int i = 0; i < WARMUP; i++) { matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0]; }
    double *t_cpu = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        double t = now_us();
        matvec_q4(cpu_out, wgt, act, N, K);
        sink += cpu_out[0];
        t_cpu[i] = now_us() - t;
    }
    pstats("CPU matvec (1×A78)", t_cpu, ITERS);

    /* ═══════════════════════════════════════════════════════
     * TEST 2: GPU SwiGLU only (sync dispatch)
     * ═══════════════════════════════════════════════════════ */
    printf("\n  === Test 2: GPU SwiGLU only (sync dispatch) ===\n");
    for (int i = 0; i < WARMUP; i++) gpu_swiglu_sync(&v, &g);
    double *t_gpu = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        t_gpu[i] = gpu_swiglu_sync(&v, &g);
    }
    pstats("GPU SwiGLU (sync)", t_gpu, ITERS);

    /* ═══════════════════════════════════════════════════════
     * TEST 3: Sequential (CPU matvec THEN GPU SwiGLU)
     * ═══════════════════════════════════════════════════════ */
    printf("\n  === Test 3: Sequential (CPU matvec then GPU SwiGLU) ===\n");
    pin(6);
    for (int i = 0; i < WARMUP; i++) {
        matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0];
        gpu_swiglu_sync(&v, &g);
    }
    double *t_seq = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        double t = now_us();
        matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0];
        gpu_swiglu_sync(&v, &g);
        t_seq[i] = now_us() - t;
    }
    pstats("Sequential (CPU+GPU)", t_seq, ITERS);

    /* ═══════════════════════════════════════════════════════
     * TEST 4: Overlapped (CPU matvec || GPU SwiGLU via timeline)
     *
     * Pipeline: pre-submit GPU work, kick GPU, do CPU matvec,
     * then wait for GPU. If GPU finishes during matvec → free.
     * ═══════════════════════════════════════════════════════ */
    printf("\n  === Test 4: Overlapped (CPU matvec || GPU SwiGLU) ===\n");
    pin(6);

    /* Warmup the pipeline */
    for (int i = 0; i < WARMUP; i++) {
        gpu_swiglu_presubmit(&v, &g);
        gpu_swiglu_kick(&v, &g);
        matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0];
        gpu_swiglu_wait(&v, &g);
    }

    double *t_overlap = calloc(ITERS, sizeof(double));
    double *t_gpu_hidden = calloc(ITERS, sizeof(double));  /* time from kick to GPU done */
    for (int i = 0; i < ITERS; i++) {
        /* Pre-submit GPU work (GPU stalls waiting for timeline signal) */
        gpu_swiglu_presubmit(&v, &g);

        double t0 = now_us();

        /* Kick GPU and start CPU simultaneously */
        gpu_swiglu_kick(&v, &g);
        double t_kick = now_us();

        /* CPU matvec runs while GPU computes */
        matvec_q4(cpu_out, wgt, act, N, K);
        sink += cpu_out[0];
        double t_cpu_done = now_us();

        /* Wait for GPU (might already be done) */
        gpu_swiglu_wait(&v, &g);
        double t_all = now_us();

        t_overlap[i] = t_all - t0;
        t_gpu_hidden[i] = t_all - t_kick;  /* how long from kick to GPU done */
    }
    pstats("Overlapped wall clock", t_overlap, ITERS);
    pstats("GPU time (kick→done)", t_gpu_hidden, ITERS);

    /* ═══════════════════════════════════════════════════════
     * TEST 5: Overlapped with RAID 0 (2×A78 matvec || GPU)
     * ═══════════════════════════════════════════════════════ */
    printf("\n  === Test 5: RAID 0 + GPU (theoretical) ===\n");
    printf("  RAID 0 2×A78 matvec ≈ 403us (from raid0_l2_v2 probe)\n");
    printf("  GPU SwiGLU dispatch  ≈ 160us (from vk_persistent_dispatch probe)\n");
    printf("  Theoretical overlap: max(403, 160) = 403us — GPU is FREE\n");

    /* ═══════════════════════════════════════════════════════
     * Summary
     * ═══════════════════════════════════════════════════════ */
    double cpu_mean = 0, gpu_mean = 0, seq_mean = 0, ovl_mean = 0;
    for (int i = 0; i < ITERS; i++) { cpu_mean += t_cpu[i]; gpu_mean += t_gpu[i]; seq_mean += t_seq[i]; ovl_mean += t_overlap[i]; }
    cpu_mean /= ITERS; gpu_mean /= ITERS; seq_mean /= ITERS; ovl_mean /= ITERS;

    printf("\n  ── Summary ──\n");
    printf("  CPU matvec alone:     %7.1f us\n", cpu_mean);
    printf("  GPU SwiGLU alone:     %7.1f us\n", gpu_mean);
    printf("  Sequential:           %7.1f us  (expected: %.1f)\n", seq_mean, cpu_mean + gpu_mean);
    printf("  Overlapped:           %7.1f us  (expected: %.1f = max)\n", ovl_mean, fmax(cpu_mean, gpu_mean));
    printf("  GPU hidden:           %5.1f%%  (1 - (overlap - cpu) / gpu)\n",
           (1.0 - (ovl_mean - cpu_mean) / gpu_mean) * 100.0);
    printf("  (sink=%.1f)\n", (double)sink);

    printf("\n════════════════════════════════════════════════════════════\n\n");

    free(t_cpu); free(t_gpu); free(t_seq); free(t_overlap); free(t_gpu_hidden);
    free(act); free(cpu_out);
    munmap((void*)data, st.st_size); close(fd);
    return 0;
}
