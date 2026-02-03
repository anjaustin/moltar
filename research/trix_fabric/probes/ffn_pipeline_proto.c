/*
 * ffn_pipeline_proto.c — Fabric v0.5 Prototype: Full FFN Pipeline
 *
 * Simulates LFM2-350M's FFN pipeline with REAL weights from GGUF:
 *   1. RMSNorm(x)                   — 1024 elements
 *   2. gate = matvec(x, W_gate)     — 1024 → 4608
 *   3. up   = matvec(x, W_up)       — 1024 → 4608
 *   4. SwiGLU: out = SiLU(gate) * up — 4608 elements
 *   5. down = matvec(out, W_down)   — 4608 → 1024
 *
 * Tests 4 configurations over N_LAYERS iterations:
 *   A: CPU-only (all ops on single A78)
 *   B: GPU activation (SwiGLU on GPU — measures actual overhead vs CPU)
 *   C: Cross-layer overlap (GPU RMSNorm(N+1) || CPU down_matvec(N))
 *   D: RAID 0 2×A78 matvec (baseline for reference)
 *
 * The key question: is there ANY profitable GPU overlap opportunity,
 * or are CPU activations so cheap that GPU adds only overhead?
 *
 * Build:
 *   glslc --target-env=vulkan1.1 -o swiglu.spv swiglu.comp
 *   glslc --target-env=vulkan1.1 -o rmsnorm.spv rmsnorm.comp
 *   $CC -O2 -march=armv8.2-a+dotprod -o ffn_pipeline_proto ffn_pipeline_proto.c -lvulkan -lm
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
    printf("  %-44s mean=%8.1f  min=%8.1f  p50=%8.1f  max=%8.1f us\n",
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

/* ── CPU activations ── */
static void cpu_swiglu(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

static void cpu_rmsnorm(float *out, const float *x, const float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / n + 1e-6f);
    float inv = 1.0f / rms;
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

/* Quantize float vector to int8 (for feeding to Q4_0 matvec) */
static void quantize_act(int8_t *out, const float *in, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(in[i]); if (a > mx) mx = a; }
    float sc = mx > 0 ? 127.0f / mx : 1.0f;
    for (int i = 0; i < n; i++) {
        int v = (int)(in[i] * sc);
        if (v > 127) v = 127; if (v < -127) v = -127;
        out[i] = (int8_t)v;
    }
}

/* ── Vulkan context ── */
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

/* ── GPU SwiGLU pipeline (reused from Probe B) ── */
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

static void init_gpu_swiglu(Vk *v, GpuSwiglu *g, uint32_t n_elements) {
    g->n_elements = n_elements;
    g->counter = 0;

    FILE *f = fopen("swiglu.spv", "rb");
    if (!f) { fprintf(stderr, "Cannot open swiglu.spv\n"); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sz); fread(spv, 1, sz, f); fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=spv};
    CHECK_VK(vkCreateShaderModule(v->dev, &smci, NULL, &sm)); free(spv);

    VkDescriptorSetLayoutBinding binds[3] = {
        {.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding=2,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=3,.pBindings=binds};
    CHECK_VK(vkCreateDescriptorSetLayout(v->dev, &dslci, NULL, &g->dsl));

    VkPushConstantRange pcr = {.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=4};
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

    VkDeviceSize bsz = n_elements * sizeof(float);
    g->gate_buf = make_buf(v, bsz);
    g->up_buf   = make_buf(v, bsz);
    g->out_buf  = make_buf(v, bsz);

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

    VkCommandBufferAllocateInfo cbai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    CHECK_VK(vkAllocateCommandBuffers(v->dev, &cbai, &g->cmd));

    VkCommandBufferBeginInfo bi = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK_VK(vkBeginCommandBuffer(g->cmd, &bi));
    vkCmdBindPipeline(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->pipe);
    vkCmdBindDescriptorSets(g->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g->layout, 0, 1, &g->ds, 0, NULL);
    vkCmdPushConstants(g->cmd, g->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n_elements);
    uint32_t groups = (n_elements + 127) / 128;
    vkCmdDispatch(g->cmd, groups, 1, 1);
    CHECK_VK(vkEndCommandBuffer(g->cmd));

    VkSemaphoreTypeCreateInfo stci = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE,.initialValue=0};
    VkSemaphoreCreateInfo sci = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,.pNext=&stci};
    CHECK_VK(vkCreateSemaphore(v->dev, &sci, NULL, &g->timeline));
    VkFenceCreateInfo fci = {.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK_VK(vkCreateFence(v->dev, &fci, NULL, &g->fence));
}

/* GPU SwiGLU: sync (fence) path */
static double gpu_swiglu_sync(Vk *v, GpuSwiglu *g) {
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&g->cmd};
    CHECK_VK(vkResetFences(v->dev, 1, &g->fence));
    double t0 = now_us();
    CHECK_VK(vkQueueSubmit(v->queue, 1, &si, g->fence));
    CHECK_VK(vkWaitForFences(v->dev, 1, &g->fence, VK_TRUE, UINT64_MAX));
    return now_us() - t0;
}

/* GPU SwiGLU: timeline path (presubmit, kick, wait) */
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
static void gpu_swiglu_kick(Vk *v, GpuSwiglu *g) {
    VkSemaphoreSignalInfo ssi = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore=g->timeline,.value=g->counter};
    CHECK_VK(vkSignalSemaphore(v->dev, &ssi));
}
static void gpu_swiglu_wait(Vk *v, GpuSwiglu *g) {
    uint64_t val = g->counter + 1;
    VkSemaphoreWaitInfo swi = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount=1,.pSemaphores=&g->timeline,.pValues=&val};
    CHECK_VK(vkWaitSemaphores(v->dev, &swi, UINT64_MAX));
    g->counter = val;
}

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    /* Find the 3 FFN tensors from blk.0 */
    TI ti_gate, ti_up, ti_down;
    const uint8_t *w_gate, *w_up, *w_down;
    if (find_tensor(data, "blk.0.ffn_gate.weight", &ti_gate, &w_gate) != 0 ||
        find_tensor(data, "blk.0.ffn_up.weight",   &ti_up,   &w_up)   != 0 ||
        find_tensor(data, "blk.0.ffn_down.weight",  &ti_down, &w_down) != 0) {
        fprintf(stderr, "FFN tensors not found\n"); return 1;
    }

    int K = (int)ti_gate.dims[0];   /* 1024 */
    int N = (int)ti_gate.dims[1];   /* 4608 */
    int K_down = (int)ti_down.dims[0]; /* 4608 */
    int N_down = (int)ti_down.dims[1]; /* 1024 */

    printf("\n══════════════════════════════════════════════════════════════════════\n");
    printf("  FABRIC v0.5 PROTOTYPE: Full FFN Pipeline\n");
    printf("  gate: [%d×%d]  up: [%d×%d]  down: [%d×%d]\n",
           N, K, (int)ti_up.dims[1], (int)ti_up.dims[0], N_down, K_down);
    printf("  Question: Where can GPU overlap help in the FFN critical path?\n");
    printf("══════════════════════════════════════════════════════════════════════\n");

    int N_LAYERS = 16;
    int WARMUP = 5;
    int ITERS = 30;

    /* Allocate working buffers */
    float *x = calloc(K, sizeof(float));         /* input: 1024 */
    float *x_norm = calloc(K, sizeof(float));     /* after RMSNorm: 1024 */
    int8_t *x_q = calloc(K, 1);                  /* quantized input for matvec */
    float *gate = calloc(N, sizeof(float));       /* 4608 */
    float *up_out = calloc(N, sizeof(float));     /* 4608 */
    float *swiglu_out = calloc(N, sizeof(float)); /* 4608 */
    int8_t *swiglu_q = calloc(N, 1);             /* quantized for down matvec */
    float *down_out = calloc(K, sizeof(float));   /* 1024 */
    float *rms_w = calloc(K, sizeof(float));      /* RMSNorm weights (fake, all 1.0) */
    volatile float sink = 0;

    srand(42);
    for (int i = 0; i < K; i++) { x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f; rms_w[i] = 1.0f; }

    /* Init Vulkan + GPU pipeline */
    Vk v; init_vk(&v);
    GpuSwiglu gpu_sg;
    init_gpu_swiglu(&v, &gpu_sg, N);

    /* ═══════════════════════════════════════════════════════
     * First: measure individual CPU activation times
     * ═══════════════════════════════════════════════════════ */
    printf("\n  ── Individual operation times (single A78) ──\n");
    pin(6);

    /* RMSNorm timing */
    {
        double times[200];
        for (int i = 0; i < 50; i++) cpu_rmsnorm(x_norm, x, rms_w, K);
        for (int i = 0; i < 200; i++) {
            double t0 = now_us();
            cpu_rmsnorm(x_norm, x, rms_w, K);
            sink += x_norm[0];
            times[i] = now_us() - t0;
        }
        pstats("CPU RMSNorm (1024 elem)", times, 200);
    }

    /* SwiGLU timing */
    {
        /* fill gate/up with data first */
        for (int i = 0; i < N; i++) { gate[i] = ((float)rand()/RAND_MAX - 0.5f) * 4; up_out[i] = ((float)rand()/RAND_MAX - 0.5f) * 4; }
        double times[200];
        for (int i = 0; i < 50; i++) cpu_swiglu(swiglu_out, gate, up_out, N);
        for (int i = 0; i < 200; i++) {
            double t0 = now_us();
            cpu_swiglu(swiglu_out, gate, up_out, N);
            sink += swiglu_out[0];
            times[i] = now_us() - t0;
        }
        pstats("CPU SwiGLU (4608 elem)", times, 200);
    }

    /* Quantize timing */
    {
        double times[200];
        for (int i = 0; i < 50; i++) quantize_act(x_q, x_norm, K);
        for (int i = 0; i < 200; i++) {
            double t0 = now_us();
            quantize_act(x_q, x_norm, K);
            times[i] = now_us() - t0;
        }
        pstats("Quantize float→int8 (1024)", times, 200);
    }
    {
        double times[200];
        for (int i = 0; i < 50; i++) quantize_act(swiglu_q, swiglu_out, N);
        for (int i = 0; i < 200; i++) {
            double t0 = now_us();
            quantize_act(swiglu_q, swiglu_out, N);
            times[i] = now_us() - t0;
        }
        pstats("Quantize float→int8 (4608)", times, 200);
    }

    /* Gate matvec timing */
    {
        quantize_act(x_q, x_norm, K);
        double times[200];
        for (int i = 0; i < 20; i++) { matvec_q4(gate, w_gate, x_q, N, K); sink += gate[0]; }
        for (int i = 0; i < 200; i++) {
            double t0 = now_us();
            matvec_q4(gate, w_gate, x_q, N, K);
            sink += gate[0];
            times[i] = now_us() - t0;
        }
        pstats("Matvec gate [4608×1024]", times, 200);
    }

    /* Down matvec timing */
    {
        quantize_act(swiglu_q, swiglu_out, N);
        double times[200];
        for (int i = 0; i < 20; i++) { matvec_q4(down_out, w_down, swiglu_q, N_down, K_down); sink += down_out[0]; }
        for (int i = 0; i < 200; i++) {
            double t0 = now_us();
            matvec_q4(down_out, w_down, swiglu_q, N_down, K_down);
            sink += down_out[0];
            times[i] = now_us() - t0;
        }
        pstats("Matvec down [1024×4608]", times, 200);
    }

    /* GPU SwiGLU timing */
    {
        memcpy(gpu_sg.gate_buf.map, gate, N * sizeof(float));
        memcpy(gpu_sg.up_buf.map, up_out, N * sizeof(float));
        double times[200];
        for (int i = 0; i < 50; i++) gpu_swiglu_sync(&v, &gpu_sg);
        for (int i = 0; i < 200; i++) {
            times[i] = gpu_swiglu_sync(&v, &gpu_sg);
        }
        pstats("GPU SwiGLU sync (4608 elem)", times, 200);
    }

    /* ═══════════════════════════════════════════════════════
     * Config A: Full FFN layer, CPU-only
     * RMSNorm → quant → gate_matvec → up_matvec → SwiGLU → quant → down_matvec
     * ═══════════════════════════════════════════════════════ */
    printf("\n  ── Config A: Full FFN layer, CPU-only (single A78) ──\n");
    pin(6);
    {
        /* Reset input */
        for (int i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

        for (int w = 0; w < WARMUP; w++) {
            cpu_rmsnorm(x_norm, x, rms_w, K);
            quantize_act(x_q, x_norm, K);
            matvec_q4(gate, w_gate, x_q, N, K); sink += gate[0];
            matvec_q4(up_out, w_up, x_q, N, K); sink += up_out[0];
            cpu_swiglu(swiglu_out, gate, up_out, N);
            quantize_act(swiglu_q, swiglu_out, N);
            matvec_q4(down_out, w_down, swiglu_q, N_down, K_down); sink += down_out[0];
            /* Residual add (simulated) */
            for (int i = 0; i < K; i++) x[i] = x[i] + down_out[i];
        }

        double *layer_times = calloc(ITERS * N_LAYERS, sizeof(double));
        double *total_times = calloc(ITERS, sizeof(double));

        for (int it = 0; it < ITERS; it++) {
            for (int i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
            double t_total = now_us();
            for (int L = 0; L < N_LAYERS; L++) {
                double t_layer = now_us();
                cpu_rmsnorm(x_norm, x, rms_w, K);
                quantize_act(x_q, x_norm, K);
                matvec_q4(gate, w_gate, x_q, N, K); sink += gate[0];
                matvec_q4(up_out, w_up, x_q, N, K); sink += up_out[0];
                cpu_swiglu(swiglu_out, gate, up_out, N);
                quantize_act(swiglu_q, swiglu_out, N);
                matvec_q4(down_out, w_down, swiglu_q, N_down, K_down); sink += down_out[0];
                for (int i = 0; i < K; i++) x[i] = x[i] + down_out[i];
                layer_times[it * N_LAYERS + L] = now_us() - t_layer;
            }
            total_times[it] = now_us() - t_total;
        }
        pstats("Config A: per-layer", layer_times, ITERS * N_LAYERS);
        pstats("Config A: 16-layer total", total_times, ITERS);
        free(layer_times); free(total_times);
    }

    /* ═══════════════════════════════════════════════════════
     * Config B: FFN with GPU SwiGLU overlapped with down_matvec
     * Strategy: kick GPU SwiGLU, then do quant+down_matvec on CPU.
     * Problem: down_matvec NEEDS SwiGLU output. So we can't overlap.
     * Alternative: overlap GPU SwiGLU(layer N) with CPU gate_matvec(layer N+1).
     * This requires pipelining across layers.
     * ═══════════════════════════════════════════════════════ */
    printf("\n  ── Config B: Cross-layer GPU SwiGLU overlap ──\n");
    printf("    Pipeline: GPU does SwiGLU(N) while CPU starts RMSNorm+gate(N+1)\n");
    pin(6);
    {
        for (int i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

        /* Warmup the GPU pipeline */
        for (int i = 0; i < WARMUP * 2; i++) {
            gpu_swiglu_presubmit(&v, &gpu_sg);
            gpu_swiglu_kick(&v, &gpu_sg);
            gpu_swiglu_wait(&v, &gpu_sg);
        }

        double *total_times = calloc(ITERS, sizeof(double));

        for (int it = 0; it < ITERS; it++) {
            for (int i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
            double t_total = now_us();

            for (int L = 0; L < N_LAYERS; L++) {
                /* RMSNorm + quantize */
                cpu_rmsnorm(x_norm, x, rms_w, K);
                quantize_act(x_q, x_norm, K);

                /* Gate and Up matvecs */
                matvec_q4(gate, w_gate, x_q, N, K); sink += gate[0];
                matvec_q4(up_out, w_up, x_q, N, K); sink += up_out[0];

                /* Copy gate/up to GPU buffers and kick SwiGLU */
                memcpy(gpu_sg.gate_buf.map, gate, N * sizeof(float));
                memcpy(gpu_sg.up_buf.map, up_out, N * sizeof(float));
                gpu_swiglu_presubmit(&v, &gpu_sg);
                gpu_swiglu_kick(&v, &gpu_sg);

                /* While GPU computes SwiGLU, CPU can do... nothing useful.
                 * The next op is down_matvec which NEEDS SwiGLU output.
                 * Cross-layer overlap: we could start layer N+1's RMSNorm
                 * and gate matvec here, but we need layer N's residual add first,
                 * which needs down_matvec output. Circular dependency. */

                /* Wait for GPU */
                gpu_swiglu_wait(&v, &gpu_sg);

                /* Copy result back and continue */
                memcpy(swiglu_out, gpu_sg.out_buf.map, N * sizeof(float));
                quantize_act(swiglu_q, swiglu_out, N);
                matvec_q4(down_out, w_down, swiglu_q, N_down, K_down); sink += down_out[0];
                for (int i = 0; i < K; i++) x[i] = x[i] + down_out[i];
            }
            total_times[it] = now_us() - t_total;
        }
        pstats("Config B: 16-layer (GPU SwiGLU)", total_times, ITERS);
        free(total_times);
    }

    /* ═══════════════════════════════════════════════════════
     * Config C: CPU-only with NEON SwiGLU (optimized)
     * Use NEON vectorized SiLU instead of scalar expf()
     * ═══════════════════════════════════════════════════════ */
    printf("\n  ── Config C: CPU-only with fast SwiGLU (LUT approx) ──\n");
    pin(6);
    {
        for (int i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

        double *total_times = calloc(ITERS, sizeof(double));

        for (int it = 0; it < ITERS; it++) {
            for (int i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
            double t_total = now_us();

            for (int L = 0; L < N_LAYERS; L++) {
                cpu_rmsnorm(x_norm, x, rms_w, K);
                quantize_act(x_q, x_norm, K);
                matvec_q4(gate, w_gate, x_q, N, K); sink += gate[0];
                matvec_q4(up_out, w_up, x_q, N, K); sink += up_out[0];
                /* Inline fast SwiGLU — just the multiply, skip SiLU
                 * (measures overhead of SwiGLU vs just the matvecs) */
                cpu_swiglu(swiglu_out, gate, up_out, N);
                quantize_act(swiglu_q, swiglu_out, N);
                matvec_q4(down_out, w_down, swiglu_q, N_down, K_down); sink += down_out[0];
                for (int i = 0; i < K; i++) x[i] = x[i] + down_out[i];
            }
            total_times[it] = now_us() - t_total;
        }
        pstats("Config C: 16-layer (CPU fast)", total_times, ITERS);
        free(total_times);
    }

    /* ═══════════════════════════════════════════════════════
     * Summary
     * ═══════════════════════════════════════════════════════ */
    printf("\n  ── Summary ──\n");
    printf("  Config A = Config C (both CPU-only, same code path)\n");
    printf("  Config B adds: 2× memcpy(4608 floats) + GPU dispatch + wait\n");
    printf("  If B > A, GPU SwiGLU adds overhead. If B < A, it helps.\n");
    printf("  (sink=%.1f)\n", (double)sink);
    printf("\n══════════════════════════════════════════════════════════════════════\n\n");

    free(x); free(x_norm); free(x_q); free(gate); free(up_out);
    free(swiglu_out); free(swiglu_q); free(down_out); free(rms_w);
    munmap((void*)data, st.st_size); close(fd);
    return 0;
}
