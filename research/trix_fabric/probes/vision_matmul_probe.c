/*
 * vision_matmul_probe.c — GPU batched F16 matmul for async vision encoding
 *
 * Tests whether the PowerVR BXM-8-256 can usefully encode SigLIP2 vision
 * patches in the background while the CPU runs text inference.
 *
 * Probe dimensions match SigLIP2: K=1152, N=1152 (attention projection).
 * Tests batch sizes 1, 16, 64 (= number of image patches).
 *
 * Modes:
 *   A. Raw F16 matmul — W[N,K] * X[B,K]^T
 *   B. Palettized matmul — 8-bit indices + 256-entry F16 palette
 *   C. CPU F16 matmul reference (NEON FMLA)
 *   D. Concurrent: GPU batched matmul + CPU Q4_0 matvec (bus contention)
 *
 * Build:
 *   glslc --target-env=vulkan1.1 -o vision_matmul.spv vision_matmul.comp
 *   $CC -O2 -march=armv8.2-a+dotprod+fp16 -o vision_matmul_probe \
 *       vision_matmul_probe.c -lvulkan -lm
 *
 * Run:
 *   adb shell "cd /data/local/tmp && taskset c0 ./vision_matmul_probe model.gguf"
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

/* ── Vulkan helpers ── */
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
static int cmp_dbl(const void *a, const void *b) {
    double da=*(const double*)a, db=*(const double*)b; return (da>db)-(da<db); }
static void pstats(const char *l, double *t, int n) {
    double s=0,mn=1e18,mx=0;
    for(int i=0;i<n;i++){s+=t[i];if(t[i]<mn)mn=t[i];if(t[i]>mx)mx=t[i];}
    double q[n]; memcpy(q,t,n*sizeof(double)); qsort(q,n,sizeof(double),cmp_dbl);
    printf("  %-48s mean=%9.1f  min=%9.1f  p50=%9.1f  max=%9.1f us\n",
           l, s/n, mn, q[n/2], mx);
}

/* ── F16 conversion ── */
static uint16_t f32_to_f16(float v) {
    uint32_t x; memcpy(&x, &v, 4);
    uint16_t s = (x >> 16) & 0x8000;
    int e = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7C00;
    return s | (e << 10) | (m >> 13);
}
static float f16_to_f32(uint16_t h) {
    uint32_t s=(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,f;
    if(!e){if(!m)f=s;else{e=1;while(!(m&0x400)){m<<=1;e--;}m&=0x3FF;f=s|((e+112)<<23)|(m<<13);}}
    else if(e==31)f=s|0x7F800000|(m<<13);else f=s|((e+112)<<23)|(m<<13);
    float r;memcpy(&r,&f,4);return r;
}

/* ── GGUF Q4_0 matvec (for concurrent CPU test) ── */
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

/* ── CPU F16 matvec reference (NEON FMLA) ── */
#ifdef __ARM_NEON
static void matvec_f16_cpu(float *out, const uint16_t *W, const uint16_t *X,
                           int N, int K, int B) {
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            const uint16_t *wrow = W + (size_t)n * K;
            const uint16_t *xrow = X + (size_t)b * K;
            int k = 0;
            for (; k + 7 < K; k += 8) {
                float16x8_t wv = vld1q_f16((const __fp16*)(wrow + k));
                float16x8_t xv = vld1q_f16((const __fp16*)(xrow + k));
                /* Widen to f32 and fma */
                acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(wv)),
                                       vcvt_f32_f16(vget_low_f16(xv)));
                acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(wv)),
                                       vcvt_f32_f16(vget_high_f16(xv)));
            }
            float sum = vaddvq_f32(acc0) + vaddvq_f32(acc1);
            for (; k < K; k++)
                sum += f16_to_f32(wrow[k]) * f16_to_f32(xrow[k]);
            out[b * N + n] = sum;
        }
    }
}
#endif

/* ── Vulkan infrastructure ── */
typedef struct {
    VkInstance inst; VkPhysicalDevice pdev; VkDevice dev;
    uint32_t qf; VkQueue queue; VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties mprops;
} Vk;
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; VkDeviceSize sz; } Buf;

static uint32_t find_mem(Vk *v,uint32_t bits,VkMemoryPropertyFlags f){
    for(uint32_t i=0;i<v->mprops.memoryTypeCount;i++)
        if((bits&(1u<<i))&&(v->mprops.memoryTypes[i].propertyFlags&f)==f) return i;
    fprintf(stderr,"no mem type\n");exit(1);
}
static Buf make_buf(Vk *v, VkDeviceSize sz, VkBufferUsageFlags usage) {
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

static void init_vk(Vk *v) {
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

    /* Enable F16 storage + compute + timeline semaphores */
    VkPhysicalDevice16BitStorageFeatures f16store={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .storageBuffer16BitAccess=VK_TRUE,.uniformAndStorageBuffer16BitAccess=VK_TRUE};
    VkPhysicalDeviceShaderFloat16Int8Features f16math={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .shaderFloat16=VK_TRUE,.pNext=&f16store};
    VkPhysicalDeviceTimelineSemaphoreFeatures tsf={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        .timelineSemaphore=VK_TRUE,.pNext=&f16math};

    const char *exts[]={"VK_KHR_timeline_semaphore","VK_KHR_16bit_storage","VK_KHR_shader_float16_int8"};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&tsf,
        .queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=3,.ppEnabledExtensionNames=exts};
    CHECK_VK(vkCreateDevice(v->pdev,&dci,NULL,&v->dev));
    vkGetDeviceQueue(v->dev,v->qf,0,&v->queue);
    VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=v->qf};
    CHECK_VK(vkCreateCommandPool(v->dev,&pci,NULL,&v->pool));
    pfn_vkSignalSemaphore=(PFN_vkSignalSemaphoreKHR)vkGetDeviceProcAddr(v->dev,"vkSignalSemaphoreKHR");
    pfn_vkWaitSemaphores=(PFN_vkWaitSemaphoresKHR)vkGetDeviceProcAddr(v->dev,"vkWaitSemaphoresKHR");
}

/* ── Vision matmul pipeline ── */
typedef struct {
    VkPipeline pipe; VkPipelineLayout layout;
    VkDescriptorSetLayout dsl; VkDescriptorPool dp; VkDescriptorSet ds;
    Buf w_f16_buf;     /* F16 weights [N*K] */
    Buf x_buf;         /* F16 activations [B*K] */
    Buf o_buf;         /* F32 output [B*N] */
    Buf w_idx_buf;     /* Palettized indices [N*K/4] packed uint32 */
    Buf palette_buf;   /* Palette [256] F16 */
    VkSemaphore timeline;
    uint64_t counter;
} VisionProbe;

static void init_vision(Vk *v, VisionProbe *vp, uint32_t N, uint32_t K, uint32_t max_B) {
    vp->counter = 0;

    /* Load shader */
    FILE *f = fopen("vision_matmul.spv", "rb");
    if (!f) { fprintf(stderr, "Cannot open vision_matmul.spv\n"); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sz); fread(spv, 1, sz, f); fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=spv};
    CHECK_VK(vkCreateShaderModule(v->dev,&smci,NULL,&sm)); free(spv);

    /* 5 bindings: w_f16, x, o, w_idx, palette */
    VkDescriptorSetLayoutBinding binds[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dslci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount=5,.pBindings=binds};
    CHECK_VK(vkCreateDescriptorSetLayout(v->dev,&dslci,NULL,&vp->dsl));

    VkPushConstantRange pcr={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=16};
    VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&vp->dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
    CHECK_VK(vkCreatePipelineLayout(v->dev,&plci,NULL,&vp->layout));

    VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},
        .layout=vp->layout};
    CHECK_VK(vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&cpci,NULL,&vp->pipe));
    vkDestroyShaderModule(v->dev,sm,NULL);

    /* Allocate buffers */
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vp->w_f16_buf   = make_buf(v, (VkDeviceSize)N * K * 2, usage);          /* F16 weights */
    vp->x_buf       = make_buf(v, (VkDeviceSize)max_B * K * 2, usage);      /* F16 activations */
    vp->o_buf       = make_buf(v, (VkDeviceSize)max_B * N * 4, usage);      /* F32 output */
    vp->w_idx_buf   = make_buf(v, (VkDeviceSize)N * K, usage);              /* uint8 indices (packed) */
    vp->palette_buf = make_buf(v, 256 * 2, usage);                          /* F16 palette */

    /* Descriptor set */
    VkDescriptorPoolSize dps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
    VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&dps};
    CHECK_VK(vkCreateDescriptorPool(v->dev,&dpci,NULL,&vp->dp));
    VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=vp->dp,.descriptorSetCount=1,.pSetLayouts=&vp->dsl};
    CHECK_VK(vkAllocateDescriptorSets(v->dev,&dsai,&vp->ds));

    VkDescriptorBufferInfo dbis[] = {
        {vp->w_f16_buf.buf, 0, vp->w_f16_buf.sz},
        {vp->x_buf.buf,     0, vp->x_buf.sz},
        {vp->o_buf.buf,     0, vp->o_buf.sz},
        {vp->w_idx_buf.buf, 0, vp->w_idx_buf.sz},
        {vp->palette_buf.buf, 0, vp->palette_buf.sz},
    };
    VkWriteDescriptorSet wds[5];
    for (int i = 0; i < 5; i++) {
        wds[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=vp->ds,.dstBinding=i,.descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbis[i]};
    }
    vkUpdateDescriptorSets(v->dev, 5, wds, 0, NULL);

    /* Timeline semaphore */
    VkSemaphoreTypeCreateInfo stci={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE,.initialValue=0};
    VkSemaphoreCreateInfo sci={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,.pNext=&stci};
    CHECK_VK(vkCreateSemaphore(v->dev,&sci,NULL,&vp->timeline));
}

/* Fire a GPU matmul and wait for completion */
static double vision_fire(Vk *v, VisionProbe *vp, uint32_t mode, uint32_t N, uint32_t K, uint32_t B) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    CHECK_VK(vkAllocateCommandBuffers(v->dev,&cbai,&cmd));

    VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK_VK(vkBeginCommandBuffer(cmd,&bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vp->pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vp->layout, 0, 1, &vp->ds, 0, NULL);
    uint32_t pc[4] = {mode, N, K, B};
    vkCmdPushConstants(cmd, vp->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(cmd, N, B, 1);  /* one workgroup per output element */
    CHECK_VK(vkEndCommandBuffer(cmd));

    uint64_t wait_val = vp->counter;
    uint64_t sig_val = vp->counter + 1;
    VkTimelineSemaphoreSubmitInfo tssi={.sType=VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount=1,.pWaitSemaphoreValues=&wait_val,
        .signalSemaphoreValueCount=1,.pSignalSemaphoreValues=&sig_val};
    VkPipelineStageFlags ws=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.pNext=&tssi,
        .waitSemaphoreCount=1,.pWaitSemaphores=&vp->timeline,.pWaitDstStageMask=&ws,
        .commandBufferCount=1,.pCommandBuffers=&cmd,
        .signalSemaphoreCount=1,.pSignalSemaphores=&vp->timeline};

    double t0 = now_us();
    CHECK_VK(vkQueueSubmit(v->queue,1,&si,VK_NULL_HANDLE));
    VkSemaphoreWaitInfo swi={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount=1,.pSemaphores=&vp->timeline,.pValues=&sig_val};
    CHECK_VK(vkWaitSemaphores(v->dev,&swi,UINT64_MAX));
    vp->counter = sig_val;
    double elapsed = now_us() - t0;

    vkFreeCommandBuffers(v->dev, v->pool, 1, &cmd);
    return elapsed;
}

/* Pre-submit GPU work that waits on a host signal (for concurrent test) */
static VkCommandBuffer vision_presubmit(Vk *v, VisionProbe *vp, uint32_t mode,
                                         uint32_t N, uint32_t K, uint32_t B) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    CHECK_VK(vkAllocateCommandBuffers(v->dev,&cbai,&cmd));

    VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK_VK(vkBeginCommandBuffer(cmd,&bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vp->pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vp->layout, 0, 1, &vp->ds, 0, NULL);
    uint32_t pc[4] = {mode, N, K, B};
    vkCmdPushConstants(cmd, vp->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(cmd, N, B, 1);
    CHECK_VK(vkEndCommandBuffer(cmd));

    uint64_t wait_val = vp->counter + 1; /* waits for host signal */
    uint64_t sig_val  = vp->counter + 2;
    VkTimelineSemaphoreSubmitInfo tssi={.sType=VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount=1,.pWaitSemaphoreValues=&wait_val,
        .signalSemaphoreValueCount=1,.pSignalSemaphoreValues=&sig_val};
    VkPipelineStageFlags ws=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.pNext=&tssi,
        .waitSemaphoreCount=1,.pWaitSemaphores=&vp->timeline,.pWaitDstStageMask=&ws,
        .commandBufferCount=1,.pCommandBuffers=&cmd,
        .signalSemaphoreCount=1,.pSignalSemaphores=&vp->timeline};
    CHECK_VK(vkQueueSubmit(v->queue,1,&si,VK_NULL_HANDLE));
    return cmd;
}
static void vision_kick(Vk *v, VisionProbe *vp) {
    VkSemaphoreSignalInfo ssi={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore=vp->timeline,.value=vp->counter+1};
    CHECK_VK(vkSignalSemaphore(v->dev,&ssi));
}
static void vision_wait(Vk *v, VisionProbe *vp) {
    uint64_t val = vp->counter + 2;
    VkSemaphoreWaitInfo swi={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount=1,.pSemaphores=&vp->timeline,.pValues=&val};
    CHECK_VK(vkWaitSemaphores(v->dev,&swi,UINT64_MAX));
    vp->counter = val;
}

/* ═══════════════ MAIN ═══════════════ */
int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    /* Map GGUF for CPU Q4_0 matvec reference */
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TI ti; const uint8_t *wgt;
    find_tensor(data, "blk.0.ffn_gate.weight", &ti, &wgt);
    int q4_K = (int)ti.dims[0], q4_N = (int)ti.dims[1];

    /* SigLIP2 dimensions */
    const uint32_t N = 1152, K = 1152;
    const uint32_t batches[] = {1, 16, 64};
    const int n_batches = 3;
    const uint32_t max_B = 64;

    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("  VISION MATMUL PROBE: GPU as Async Vision Encoder\n");
    printf("  SigLIP2 dimensions: N=%u K=%u, batches=1,16,64\n", N, K);
    printf("  PowerVR BXM-8-256: 128-wide SIMD, F16 compute, 16KB shared\n");
    printf("══════════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);

    pin(6);
    printf("  [init] pinned to cpu6 (big core)\n"); fflush(stdout);
    Vk v; init_vk(&v);
    printf("  [init] Vulkan ready (F16 storage + compute enabled)\n"); fflush(stdout);
    VisionProbe vp; init_vision(&v, &vp, N, K, max_B);
    printf("  [init] buffers: W=%.1f MB, X=%.1f KB, O=%.1f KB\n",
           (double)(N*K*2)/1e6, (double)(max_B*K*2)/1e3, (double)(max_B*N*4)/1e3);
    fflush(stdout);

    /* Fill weights with random F16 */
    srand(42);
    uint16_t *w_host = (uint16_t*)vp.w_f16_buf.map;
    for (uint32_t i = 0; i < N*K; i++)
        w_host[i] = f32_to_f16(((float)rand()/(float)RAND_MAX - 0.5f) * 0.1f);

    /* Fill activations with random F16 */
    uint16_t *x_host = (uint16_t*)vp.x_buf.map;
    for (uint32_t i = 0; i < max_B*K; i++)
        x_host[i] = f32_to_f16(((float)rand()/(float)RAND_MAX - 0.5f) * 2.0f);

    /* Build palette: 256 uniformly spaced F16 values covering weight range */
    uint16_t *pal = (uint16_t*)vp.palette_buf.map;
    for (int i = 0; i < 256; i++)
        pal[i] = f32_to_f16(-0.05f + 0.1f * (float)i / 255.0f);

    /* Build palettized weight indices: nearest palette entry for each weight */
    uint8_t *idx = (uint8_t*)vp.w_idx_buf.map;
    for (uint32_t i = 0; i < N*K; i++) {
        float wv = f16_to_f32(w_host[i]);
        int best = 0; float best_d = 1e9;
        for (int p = 0; p < 256; p++) {
            float pv = f16_to_f32(pal[p]);
            float d = fabsf(wv - pv);
            if (d < best_d) { best_d = d; best = p; }
        }
        idx[i] = (uint8_t)best;
    }
    printf("  [init] data ready (weights, activations, palette, indices)\n\n");
    fflush(stdout);

    int WARMUP = 5;
    int ITERS = 20;  /* fewer iters — each dispatch is expensive */
    double *t = calloc(ITERS, sizeof(double));

    /* ── Test A: GPU Raw F16 matmul at various batch sizes ── */
    printf("  ═══ Test A: GPU Raw F16 Batched Matmul [%u x %u] ═══\n\n", N, K);
    fflush(stdout);
    for (int bi = 0; bi < n_batches; bi++) {
        uint32_t B = batches[bi];
        char label[64]; snprintf(label, sizeof(label), "GPU F16 matmul B=%u (%u×%u)", B, N, K);

        for (int i = 0; i < WARMUP; i++) vision_fire(&v, &vp, 0, N, K, B);
        for (int i = 0; i < ITERS; i++) t[i] = vision_fire(&v, &vp, 0, N, K, B);
        pstats(label, t, ITERS);

        double mean = 0; for(int i=0;i<ITERS;i++) mean+=t[i]; mean/=ITERS;
        double gflops = 2.0 * N * K * B / (mean * 1e3);  /* GFLOP/s */
        printf("    → %.2f GFLOP/s, %.1f ms per matmul\n\n", gflops, mean/1e3);
        fflush(stdout);
    }

    /* ── Test B: GPU Palettized matmul at various batch sizes ── */
    printf("  ═══ Test B: GPU Palettized Matmul [%u x %u] (8-bit idx + 256 palette) ═══\n\n", N, K);
    fflush(stdout);
    for (int bi = 0; bi < n_batches; bi++) {
        uint32_t B = batches[bi];
        char label[64]; snprintf(label, sizeof(label), "GPU palette matmul B=%u (%u×%u)", B, N, K);

        for (int i = 0; i < WARMUP; i++) vision_fire(&v, &vp, 1, N, K, B);
        for (int i = 0; i < ITERS; i++) t[i] = vision_fire(&v, &vp, 1, N, K, B);
        pstats(label, t, ITERS);

        double mean = 0; for(int i=0;i<ITERS;i++) mean+=t[i]; mean/=ITERS;
        double gflops = 2.0 * N * K * B / (mean * 1e3);
        printf("    → %.2f GFLOP/s, %.1f ms per matmul\n\n", gflops, mean/1e3);
        fflush(stdout);
    }

    /* ── Test C: CPU F16 matmul reference (NEON) ── */
    printf("  ═══ Test C: CPU F16 Matmul Reference (NEON FMLA) ═══\n\n");
    fflush(stdout);
    float *cpu_out = calloc(max_B * N, sizeof(float));
    for (int bi = 0; bi < n_batches; bi++) {
        uint32_t B = batches[bi];
        char label[64]; snprintf(label, sizeof(label), "CPU F16 matmul B=%u (%u×%u)", B, N, K);

        /* warmup */
        matvec_f16_cpu(cpu_out, w_host, x_host, N, K, B);
        for (int i = 0; i < ITERS; i++) {
            double t0 = now_us();
            matvec_f16_cpu(cpu_out, w_host, x_host, N, K, B);
            t[i] = now_us() - t0;
        }
        pstats(label, t, ITERS);

        double mean = 0; for(int i=0;i<ITERS;i++) mean+=t[i]; mean/=ITERS;
        double gflops = 2.0 * N * K * B / (mean * 1e3);
        printf("    → %.2f GFLOP/s, %.1f ms per matmul\n\n", gflops, mean/1e3);
        fflush(stdout);
    }

    /* ── Test D: Correctness check — GPU vs CPU (batch=1) ── */
    printf("  ═══ Test D: Correctness Check ═══\n\n");
    fflush(stdout);
    /* Run GPU */
    vision_fire(&v, &vp, 0, N, K, 1);
    float *gpu_out = (float*)vp.o_buf.map;
    /* Run CPU */
    matvec_f16_cpu(cpu_out, w_host, x_host, N, K, 1);
    float max_err = 0, avg_err = 0;
    for (uint32_t i = 0; i < N; i++) {
        float err = fabsf(gpu_out[i] - cpu_out[i]);
        float rel = (fabsf(cpu_out[i]) > 1e-6f) ? err / fabsf(cpu_out[i]) : err;
        if (rel > max_err) max_err = rel;
        avg_err += rel;
    }
    avg_err /= N;
    printf("  GPU vs CPU: max_rel_err=%.6f  avg_rel_err=%.6f\n\n", max_err, avg_err);

    /* ── Test E: Concurrent GPU vision + CPU Q4_0 text inference ── */
    printf("  ═══ Test E: Concurrent GPU Vision + CPU Text Inference ═══\n\n");
    fflush(stdout);

    int8_t *q4_act = calloc(q4_K, 1);
    float *q4_out = calloc(q4_N, sizeof(float));
    volatile float sink = 0;
    for (int i = 0; i < q4_K; i++) q4_act[i] = (int8_t)(rand() % 256 - 128);

    /* CPU Q4_0 matvec alone baseline */
    for (int i = 0; i < 10; i++) { matvec_q4(q4_out, wgt, q4_act, q4_N, q4_K); sink += q4_out[0]; }
    double *t_cpu_alone = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_us();
        matvec_q4(q4_out, wgt, q4_act, q4_N, q4_K); sink += q4_out[0];
        t_cpu_alone[i] = now_us() - t0;
    }
    pstats("CPU Q4_0 matvec ALONE", t_cpu_alone, ITERS);

    /* CPU Q4_0 matvec while GPU does F16 batched matmul B=64 */
    double *t_cpu_concurrent = calloc(ITERS, sizeof(double));
    double *t_gpu_concurrent = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        VkCommandBuffer cmd = vision_presubmit(&v, &vp, 0, N, K, 64);

        double t0 = now_us();
        vision_kick(&v, &vp);
        /* CPU does Q4_0 matvec while GPU processes vision */
        matvec_q4(q4_out, wgt, q4_act, q4_N, q4_K); sink += q4_out[0];
        double t_cpu_done = now_us();
        t_cpu_concurrent[i] = t_cpu_done - t0;

        vision_wait(&v, &vp);
        double t_done = now_us();
        t_gpu_concurrent[i] = t_done - t0;

        vkFreeCommandBuffers(v.dev, v.pool, 1, &cmd);
    }
    pstats("CPU Q4_0 matvec CONCURRENT (with GPU B=64)", t_cpu_concurrent, ITERS);
    pstats("GPU F16 matmul B=64 CONCURRENT wall time", t_gpu_concurrent, ITERS);

    /* Summary */
    double cpu_alone_mean = 0, cpu_conc_mean = 0;
    for(int i=0;i<ITERS;i++) { cpu_alone_mean += t_cpu_alone[i]; cpu_conc_mean += t_cpu_concurrent[i]; }
    cpu_alone_mean /= ITERS; cpu_conc_mean /= ITERS;
    double slowdown = (cpu_conc_mean - cpu_alone_mean) / cpu_alone_mean * 100;

    printf("\n  ── Summary ──\n");
    printf("  CPU Q4_0 matvec alone:       %.1f us\n", cpu_alone_mean);
    printf("  CPU Q4_0 matvec concurrent:  %.1f us  (%.1f%% %s)\n",
           cpu_conc_mean, fabsf(slowdown), slowdown > 0 ? "SLOWER" : "faster");
    printf("  → If slowdown < 10%%: GPU vision encoding is nearly FREE\n");
    printf("  (sink=%.1f)\n", (double)sink);
    printf("\n══════════════════════════════════════════════════════════════════\n\n");

    free(t); free(t_cpu_alone); free(t_cpu_concurrent); free(t_gpu_concurrent);
    free(q4_act); free(q4_out); free(cpu_out);
    munmap((void*)data, st.st_size); close(fd);
    return 0;
}
