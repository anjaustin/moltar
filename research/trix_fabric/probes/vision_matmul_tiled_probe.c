/*
 * vision_matmul_tiled_probe.c — Tiled GPU matmul vs v1 head-to-head
 *
 * Loads BOTH shaders (v1 = vision_matmul.spv, v2 = vision_matmul_tiled.spv)
 * and benchmarks them with identical data.
 *
 * Tests:
 *   A. v1 (naive) GPU F16 matmul at B=1,16,64
 *   B. v2 (tiled) GPU F16 matmul at B=1,16,64
 *   C. Correctness: v2 vs CPU
 *   D. Concurrent: v2 tiled + CPU Q4_0 matvec (bus contention)
 *   E. Summary comparison
 *
 * Build:
 *   glslc --target-env=vulkan1.1 -o vision_matmul.spv vision_matmul.comp
 *   glslc --target-env=vulkan1.1 -o vision_matmul_tiled.spv vision_matmul_tiled.comp
 *   $CC -O2 -march=armv8.2-a+dotprod+fp16 -o vision_matmul_tiled_probe \
 *       vision_matmul_tiled_probe.c -lvulkan -lm
 *
 * Run:
 *   adb shell "cd /data/local/tmp && taskset c0 ./vision_matmul_tiled_probe LFM2-350M-Q4_0.gguf"
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
    printf("  %-52s mean=%9.1f  min=%9.1f  p50=%9.1f  max=%9.1f us\n",
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

/* ── CPU F16 matvec reference ── */
#ifdef __ARM_NEON
static void matvec_f16_cpu(float *out, const uint16_t *W, const uint16_t *X,
                           int N, int K, int B) {
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
            const uint16_t *wrow = W + (size_t)n * K;
            const uint16_t *xrow = X + (size_t)b * K;
            int k = 0;
            for (; k + 7 < K; k += 8) {
                float16x8_t wv = vld1q_f16((const __fp16*)(wrow + k));
                float16x8_t xv = vld1q_f16((const __fp16*)(xrow + k));
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

/* ── Pipeline creation from SPV file ── */
typedef struct {
    VkPipeline pipe; VkPipelineLayout layout;
    VkDescriptorSetLayout dsl; VkDescriptorPool dp; VkDescriptorSet ds;
    VkSemaphore timeline;
    uint64_t counter;
} Pipeline;

static VkShaderModule load_spv(Vk *v, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sz); fread(spv, 1, sz, f); fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=spv};
    CHECK_VK(vkCreateShaderModule(v->dev,&smci,NULL,&sm)); free(spv);
    return sm;
}

static Pipeline make_pipeline(Vk *v, const char *spv_path, Buf *bufs, int n_bufs) {
    Pipeline p = {.counter = 0};
    VkShaderModule sm = load_spv(v, spv_path);

    /* 5 bindings (same layout for both v1 and v2 for descriptor compat) */
    VkDescriptorSetLayoutBinding binds[5];
    for (int i = 0; i < 5; i++) {
        binds[i] = (VkDescriptorSetLayoutBinding){
            .binding=i,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
    }
    VkDescriptorSetLayoutCreateInfo dslci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount=5,.pBindings=binds};
    CHECK_VK(vkCreateDescriptorSetLayout(v->dev,&dslci,NULL,&p.dsl));

    VkPushConstantRange pcr={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=16};
    VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&p.dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
    CHECK_VK(vkCreatePipelineLayout(v->dev,&plci,NULL,&p.layout));

    VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},
        .layout=p.layout};
    CHECK_VK(vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&cpci,NULL,&p.pipe));
    vkDestroyShaderModule(v->dev,sm,NULL);

    VkDescriptorPoolSize dps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
    VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&dps};
    CHECK_VK(vkCreateDescriptorPool(v->dev,&dpci,NULL,&p.dp));
    VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=p.dp,.descriptorSetCount=1,.pSetLayouts=&p.dsl};
    CHECK_VK(vkAllocateDescriptorSets(v->dev,&dsai,&p.ds));

    VkWriteDescriptorSet wds[5];
    VkDescriptorBufferInfo dbis[5];
    for (int i = 0; i < 5; i++) {
        dbis[i] = (VkDescriptorBufferInfo){.buffer=bufs[i].buf, .offset=0, .range=bufs[i].sz};
        wds[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=p.ds,.dstBinding=i,.descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbis[i]};
    }
    vkUpdateDescriptorSets(v->dev, 5, wds, 0, NULL);

    VkSemaphoreTypeCreateInfo stci={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE,.initialValue=0};
    VkSemaphoreCreateInfo sci={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,.pNext=&stci};
    CHECK_VK(vkCreateSemaphore(v->dev,&sci,NULL,&p.timeline));

    return p;
}

/* Fire GPU dispatch and wait */
static double gpu_fire(Vk *v, Pipeline *p, uint32_t mode, uint32_t N, uint32_t K, uint32_t B,
                       uint32_t dispatch_x, uint32_t dispatch_y) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    CHECK_VK(vkAllocateCommandBuffers(v->dev,&cbai,&cmd));

    VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK_VK(vkBeginCommandBuffer(cmd,&bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &p->ds, 0, NULL);
    uint32_t pc[4] = {mode, N, K, B};
    vkCmdPushConstants(cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(cmd, dispatch_x, dispatch_y, 1);
    CHECK_VK(vkEndCommandBuffer(cmd));

    uint64_t wait_val = p->counter;
    uint64_t sig_val = p->counter + 1;
    VkTimelineSemaphoreSubmitInfo tssi={.sType=VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount=1,.pWaitSemaphoreValues=&wait_val,
        .signalSemaphoreValueCount=1,.pSignalSemaphoreValues=&sig_val};
    VkPipelineStageFlags ws=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.pNext=&tssi,
        .waitSemaphoreCount=1,.pWaitSemaphores=&p->timeline,.pWaitDstStageMask=&ws,
        .commandBufferCount=1,.pCommandBuffers=&cmd,
        .signalSemaphoreCount=1,.pSignalSemaphores=&p->timeline};

    double t0 = now_us();
    CHECK_VK(vkQueueSubmit(v->queue,1,&si,VK_NULL_HANDLE));
    VkSemaphoreWaitInfo swi={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount=1,.pSemaphores=&p->timeline,.pValues=&sig_val};
    CHECK_VK(vkWaitSemaphores(v->dev,&swi,UINT64_MAX));
    p->counter = sig_val;
    double elapsed = now_us() - t0;
    vkFreeCommandBuffers(v->dev, v->pool, 1, &cmd);
    return elapsed;
}

/* Pre-submit for concurrent test */
static VkCommandBuffer gpu_presubmit(Vk *v, Pipeline *p, uint32_t mode,
                                      uint32_t N, uint32_t K, uint32_t B,
                                      uint32_t dispatch_x, uint32_t dispatch_y) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    CHECK_VK(vkAllocateCommandBuffers(v->dev,&cbai,&cmd));

    VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK_VK(vkBeginCommandBuffer(cmd,&bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &p->ds, 0, NULL);
    uint32_t pc[4] = {mode, N, K, B};
    vkCmdPushConstants(cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(cmd, dispatch_x, dispatch_y, 1);
    CHECK_VK(vkEndCommandBuffer(cmd));

    uint64_t wait_val = p->counter + 1;
    uint64_t sig_val  = p->counter + 2;
    VkTimelineSemaphoreSubmitInfo tssi={.sType=VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount=1,.pWaitSemaphoreValues=&wait_val,
        .signalSemaphoreValueCount=1,.pSignalSemaphoreValues=&sig_val};
    VkPipelineStageFlags ws=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.pNext=&tssi,
        .waitSemaphoreCount=1,.pWaitSemaphores=&p->timeline,.pWaitDstStageMask=&ws,
        .commandBufferCount=1,.pCommandBuffers=&cmd,
        .signalSemaphoreCount=1,.pSignalSemaphores=&p->timeline};
    CHECK_VK(vkQueueSubmit(v->queue,1,&si,VK_NULL_HANDLE));
    return cmd;
}
static void gpu_kick(Vk *v, Pipeline *p) {
    VkSemaphoreSignalInfo ssi={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore=p->timeline,.value=p->counter+1};
    CHECK_VK(vkSignalSemaphore(v->dev,&ssi));
}
static void gpu_wait(Vk *v, Pipeline *p) {
    uint64_t val = p->counter + 2;
    VkSemaphoreWaitInfo swi={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount=1,.pSemaphores=&p->timeline,.pValues=&val};
    CHECK_VK(vkWaitSemaphores(v->dev,&swi,UINT64_MAX));
    p->counter = val;
}

/* ═══════════════ MAIN ═══════════════ */
int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    /* Map GGUF for CPU Q4_0 matvec */
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TI ti; const uint8_t *wgt;
    find_tensor(data, "blk.0.ffn_gate.weight", &ti, &wgt);
    int q4_K = (int)ti.dims[0], q4_N = (int)ti.dims[1];

    const uint32_t N = 1152, K = 1152;
    const uint32_t batches[] = {1, 16, 64};
    const int n_batches = 3;
    const uint32_t max_B = 64;

    printf("\n══════════════════════════════════════════════════════════════════════\n");
    printf("  TILED MATMUL PROBE: Shared Memory Tiling vs Naive\n");
    printf("  SigLIP2: N=%u K=%u, batches=1,16,64\n", N, K);
    printf("  v1: dispatch(N,B,1) = per-element, weights loaded B times\n");
    printf("  v2: dispatch(N,1,1) = per-row, weights loaded ONCE via shmem\n");
    printf("══════════════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);

    pin(6);
    printf("  [init] pinned to cpu6 (big core)\n"); fflush(stdout);
    Vk v; init_vk(&v);
    printf("  [init] Vulkan ready\n"); fflush(stdout);

    /* Shared buffers */
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    Buf bufs[5];
    bufs[0] = make_buf(&v, (VkDeviceSize)N * K * 2, usage);          /* W F16 */
    bufs[1] = make_buf(&v, (VkDeviceSize)max_B * K * 2, usage);      /* X F16 */
    bufs[2] = make_buf(&v, (VkDeviceSize)max_B * N * 4, usage);      /* O F32 */
    bufs[3] = make_buf(&v, (VkDeviceSize)N * K, usage);              /* idx (v1 compat) */
    bufs[4] = make_buf(&v, 256 * 2, usage);                          /* palette (v1 compat) */

    /* Create both pipelines sharing same buffers */
    Pipeline p_v1 = make_pipeline(&v, "vision_matmul.spv", bufs, 5);
    printf("  [init] v1 pipeline (naive) ready\n"); fflush(stdout);
    Pipeline p_v2 = make_pipeline(&v, "vision_matmul_tiled.spv", bufs, 5);
    printf("  [init] v2 pipeline (tiled) ready\n"); fflush(stdout);

    /* Fill data */
    srand(42);
    uint16_t *w_host = (uint16_t*)bufs[0].map;
    for (uint32_t i = 0; i < N*K; i++)
        w_host[i] = f32_to_f16(((float)rand()/(float)RAND_MAX - 0.5f) * 0.1f);
    uint16_t *x_host = (uint16_t*)bufs[1].map;
    for (uint32_t i = 0; i < max_B*K; i++)
        x_host[i] = f32_to_f16(((float)rand()/(float)RAND_MAX - 0.5f) * 2.0f);
    printf("  [init] data ready\n\n"); fflush(stdout);

    int WARMUP = 5, ITERS = 20;
    double *t = calloc(ITERS, sizeof(double));
    double v1_means[3], v2_means[3];

    /* ── Test A: v1 naive at B=1,16,64 ── */
    printf("  ═══ Test A: v1 Naive GPU F16 [%u x %u] ═══\n\n", N, K);
    fflush(stdout);
    for (int bi = 0; bi < n_batches; bi++) {
        uint32_t B = batches[bi];
        char label[80]; snprintf(label, sizeof(label), "v1 naive B=%u  dispatch(%u,%u,1)", B, N, B);
        for (int i = 0; i < WARMUP; i++) gpu_fire(&v, &p_v1, 0, N, K, B, N, B);
        for (int i = 0; i < ITERS; i++) t[i] = gpu_fire(&v, &p_v1, 0, N, K, B, N, B);
        pstats(label, t, ITERS);
        double mean = 0; for(int i=0;i<ITERS;i++) mean+=t[i]; mean/=ITERS;
        v1_means[bi] = mean;
        double gflops = 2.0 * N * K * B / (mean * 1e3);
        printf("    → %.2f GFLOP/s, %.1f ms\n\n", gflops, mean/1e3);
        fflush(stdout);
    }

    /* ── Test B: v2 tiled at B=1,16,64 ── */
    printf("  ═══ Test B: v2 Tiled GPU F16 [%u x %u] (shmem weight reuse) ═══\n\n", N, K);
    fflush(stdout);
    for (int bi = 0; bi < n_batches; bi++) {
        uint32_t B = batches[bi];
        char label[80]; snprintf(label, sizeof(label), "v2 tiled B=%u  dispatch(%u,1,1)", B, N);
        /* v2 uses mode=2 and dispatches (N, 1, 1) */
        for (int i = 0; i < WARMUP; i++) gpu_fire(&v, &p_v2, 2, N, K, B, N, 1);
        for (int i = 0; i < ITERS; i++) t[i] = gpu_fire(&v, &p_v2, 2, N, K, B, N, 1);
        pstats(label, t, ITERS);
        double mean = 0; for(int i=0;i<ITERS;i++) mean+=t[i]; mean/=ITERS;
        v2_means[bi] = mean;
        double gflops = 2.0 * N * K * B / (mean * 1e3);
        printf("    → %.2f GFLOP/s, %.1f ms\n\n", gflops, mean/1e3);
        fflush(stdout);
    }

    /* ── Test C: Correctness v2 vs CPU ── */
    printf("  ═══ Test C: Correctness (v2 tiled vs CPU) ═══\n\n");
    fflush(stdout);
    gpu_fire(&v, &p_v2, 2, N, K, 1, N, 1);
    float *gpu_out = (float*)bufs[2].map;
    float *cpu_out = calloc(max_B * N, sizeof(float));
    matvec_f16_cpu(cpu_out, w_host, x_host, N, K, 1);
    float max_err = 0, avg_err = 0;
    for (uint32_t i = 0; i < N; i++) {
        float err = fabsf(gpu_out[i] - cpu_out[i]);
        float rel = (fabsf(cpu_out[i]) > 1e-6f) ? err / fabsf(cpu_out[i]) : err;
        if (rel > max_err) max_err = rel;
        avg_err += rel;
    }
    avg_err /= N;
    printf("  v2 vs CPU: max_rel_err=%.6f  avg_rel_err=%.6f  %s\n\n",
           max_err, avg_err, max_err < 0.01f ? "PASS" : "FAIL");
    fflush(stdout);

    /* ── Test D: Concurrent v2 tiled B=64 + CPU Q4_0 ── */
    printf("  ═══ Test D: Concurrent v2 Tiled B=64 + CPU Q4_0 ═══\n\n");
    fflush(stdout);

    int8_t *q4_act = calloc(q4_K, 1);
    float *q4_out = calloc(q4_N, sizeof(float));
    volatile float sink = 0;
    for (int i = 0; i < q4_K; i++) q4_act[i] = (int8_t)(rand() % 256 - 128);

    /* CPU alone baseline */
    for (int i = 0; i < 10; i++) { matvec_q4(q4_out, wgt, q4_act, q4_N, q4_K); sink += q4_out[0]; }
    double *t_cpu_alone = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_us();
        matvec_q4(q4_out, wgt, q4_act, q4_N, q4_K); sink += q4_out[0];
        t_cpu_alone[i] = now_us() - t0;
    }
    pstats("CPU Q4_0 matvec ALONE", t_cpu_alone, ITERS);

    /* v1 naive concurrent (for comparison) */
    double *t_cpu_v1 = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        VkCommandBuffer cmd = gpu_presubmit(&v, &p_v1, 0, N, K, 64, N, 64);
        double t0 = now_us();
        gpu_kick(&v, &p_v1);
        matvec_q4(q4_out, wgt, q4_act, q4_N, q4_K); sink += q4_out[0];
        t_cpu_v1[i] = now_us() - t0;
        gpu_wait(&v, &p_v1);
        vkFreeCommandBuffers(v.dev, v.pool, 1, &cmd);
    }
    pstats("CPU Q4_0 CONCURRENT w/ v1 naive B=64", t_cpu_v1, ITERS);

    /* v2 tiled concurrent */
    double *t_cpu_v2 = calloc(ITERS, sizeof(double));
    double *t_gpu_v2 = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        VkCommandBuffer cmd = gpu_presubmit(&v, &p_v2, 2, N, K, 64, N, 1);
        double t0 = now_us();
        gpu_kick(&v, &p_v2);
        matvec_q4(q4_out, wgt, q4_act, q4_N, q4_K); sink += q4_out[0];
        double t_cpu_done = now_us();
        t_cpu_v2[i] = t_cpu_done - t0;
        gpu_wait(&v, &p_v2);
        t_gpu_v2[i] = now_us() - t0;
        vkFreeCommandBuffers(v.dev, v.pool, 1, &cmd);
    }
    pstats("CPU Q4_0 CONCURRENT w/ v2 tiled B=64", t_cpu_v2, ITERS);
    pstats("v2 tiled B=64 GPU wall time (concurrent)", t_gpu_v2, ITERS);

    /* ── Summary ── */
    double alone_mean=0, v1c_mean=0, v2c_mean=0;
    for(int i=0;i<ITERS;i++) { alone_mean+=t_cpu_alone[i]; v1c_mean+=t_cpu_v1[i]; v2c_mean+=t_cpu_v2[i]; }
    alone_mean/=ITERS; v1c_mean/=ITERS; v2c_mean/=ITERS;
    double v1_slow = (v1c_mean - alone_mean) / alone_mean * 100;
    double v2_slow = (v2c_mean - alone_mean) / alone_mean * 100;

    printf("\n  ═══════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("  ═══════════════════════════════════════════════════════════════\n\n");

    printf("  GPU throughput (B=64):\n");
    printf("    v1 naive:  %.1f ms  (%.2f GFLOP/s)\n", v1_means[2]/1e3, 2.0*N*K*64/(v1_means[2]*1e3));
    printf("    v2 tiled:  %.1f ms  (%.2f GFLOP/s)\n", v2_means[2]/1e3, 2.0*N*K*64/(v2_means[2]*1e3));
    double speedup = v1_means[2] / v2_means[2];
    printf("    → v2 is %.2fx %s than v1\n\n", speedup > 1.0 ? speedup : 1.0/speedup,
           speedup > 1.0 ? "FASTER" : "SLOWER");

    printf("  Speedup by batch size:\n");
    for (int bi = 0; bi < n_batches; bi++) {
        double sp = v1_means[bi] / v2_means[bi];
        printf("    B=%-3u  v1=%.1fms  v2=%.1fms  → %.2fx\n",
               batches[bi], v1_means[bi]/1e3, v2_means[bi]/1e3, sp);
    }

    printf("\n  Bus contention (CPU Q4_0 matvec slowdown):\n");
    printf("    CPU alone:            %.1f us\n", alone_mean);
    printf("    w/ v1 naive (B=64):   %.1f us  (%.1f%% %s)\n",
           v1c_mean, fabs(v1_slow), v1_slow > 0 ? "SLOWER" : "faster");
    printf("    w/ v2 tiled (B=64):   %.1f us  (%.1f%% %s)\n",
           v2c_mean, fabs(v2_slow), v2_slow > 0 ? "SLOWER" : "faster");
    printf("    → Tiling %s bus contention by %.1f pp\n",
           v2_slow < v1_slow ? "REDUCED" : "INCREASED", fabs(v1_slow - v2_slow));
    printf("  (sink=%.1f)\n", (double)sink);
    printf("\n══════════════════════════════════════════════════════════════════════\n\n");

    free(t); free(t_cpu_alone); free(t_cpu_v1); free(t_cpu_v2); free(t_gpu_v2);
    free(q4_act); free(q4_out); free(cpu_out);
    munmap((void*)data, st.st_size); close(fd);
    return 0;
}
