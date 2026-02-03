/*
 * coprocessor_v2_probe.c — PowerVR as Fabric Coprocessor (driver-friendly)
 *
 * Uses timeline semaphores for rapid redispatch (proven in Probe A).
 * Pre-records one command buffer per operation, resubmits via timeline.
 * No spin-wait, no driver watchdog issues.
 *
 * Tests:
 *   1. NOP round-trip (timeline signal→completion baseline)
 *   2. SiLU on 1024 elements (D_MODEL size)
 *   3. RMSNorm on 1024 elements
 *   4. SwiGLU on 1024 elements (gate + up → output)
 *   5. Softmax on 1024 elements
 *   6. CPU matvec alone (reference)
 *   7. CPU matvec + concurrent GPU SiLU (bus contention test)
 *   8. CPU matvec + concurrent GPU NOP (dispatch-only contention)
 *
 * Build:
 *   glslc --target-env=vulkan1.1 -o coprocessor_v2.spv coprocessor_v2.comp
 *   $CC -O2 -march=armv8.2-a+dotprod -o coprocessor_v2_probe coprocessor_v2_probe.c -lvulkan -lm
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
    printf("  %-44s mean=%7.1f  min=%7.1f  p50=%7.1f  p99=%7.1f  max=%7.1f us\n",
           l, s/n, mn, q[n/2], q[(int)(n*0.99)], mx);
}

/* ── GGUF + matvec (same as other probes) ── */
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

/* ── Vulkan ── */
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

/* ── Coprocessor pipeline ── */
typedef struct {
    VkPipeline pipe; VkPipelineLayout layout;
    VkDescriptorSetLayout dsl; VkDescriptorPool dp; VkDescriptorSet ds;
    Buf data_buf;
    VkCommandBuffer cmds[5]; /* one per op type */
    VkSemaphore timeline;
    uint64_t counter;
} Coproc;

static void init_coproc(Vk *v, Coproc *c, uint32_t n_elem) {
    c->counter = 0;

    FILE *f = fopen("coprocessor_v2.spv", "rb");
    if (!f) { fprintf(stderr, "Cannot open coprocessor_v2.spv\n"); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sz); fread(spv, 1, sz, f); fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=spv};
    CHECK_VK(vkCreateShaderModule(v->dev, &smci, NULL, &sm)); free(spv);

    VkDescriptorSetLayoutBinding bind = {.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dslci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=1,.pBindings=&bind};
    CHECK_VK(vkCreateDescriptorSetLayout(v->dev, &dslci, NULL, &c->dsl));

    VkPushConstantRange pcr = {.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=8}; /* op + n */
    VkPipelineLayoutCreateInfo plci = {.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&c->dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
    CHECK_VK(vkCreatePipelineLayout(v->dev, &plci, NULL, &c->layout));

    VkComputePipelineCreateInfo cpci = {.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},
        .layout=c->layout};
    CHECK_VK(vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &c->pipe));
    vkDestroyShaderModule(v->dev, sm, NULL);

    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&dps};
    CHECK_VK(vkCreateDescriptorPool(v->dev, &dpci, NULL, &c->dp));
    VkDescriptorSetAllocateInfo dsai = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=c->dp,.descriptorSetCount=1,.pSetLayouts=&c->dsl};
    CHECK_VK(vkAllocateDescriptorSets(v->dev, &dsai, &c->ds));

    /* Data buffer: 2×n_elem floats (data + weights for RMSNorm/SwiGLU) */
    c->data_buf = make_buf(v, 2 * n_elem * sizeof(float));
    VkDescriptorBufferInfo dbi = {.buffer=c->data_buf.buf,.offset=0,.range=2*n_elem*sizeof(float)};
    VkWriteDescriptorSet wds = {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet=c->ds,.dstBinding=0,.descriptorCount=1,
        .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbi};
    vkUpdateDescriptorSets(v->dev, 1, &wds, 0, NULL);

    /* Pre-record one command buffer per operation */
    VkCommandBufferAllocateInfo cbai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=5};
    CHECK_VK(vkAllocateCommandBuffers(v->dev, &cbai, c->cmds));

    for (uint32_t op = 0; op < 5; op++) {
        VkCommandBufferBeginInfo bi = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK_VK(vkBeginCommandBuffer(c->cmds[op], &bi));
        vkCmdBindPipeline(c->cmds[op], VK_PIPELINE_BIND_POINT_COMPUTE, c->pipe);
        vkCmdBindDescriptorSets(c->cmds[op], VK_PIPELINE_BIND_POINT_COMPUTE, c->layout, 0, 1, &c->ds, 0, NULL);
        uint32_t pc[2] = {op, n_elem};
        vkCmdPushConstants(c->cmds[op], c->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
        vkCmdDispatch(c->cmds[op], 1, 1, 1);
        CHECK_VK(vkEndCommandBuffer(c->cmds[op]));
    }

    VkSemaphoreTypeCreateInfo stci = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE,.initialValue=0};
    VkSemaphoreCreateInfo sci = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,.pNext=&stci};
    CHECK_VK(vkCreateSemaphore(v->dev, &sci, NULL, &c->timeline));
}

/* ── Simple fire-and-wait (for tests 1-5): no gate, just submit+wait ── */
static double coproc_fire(Vk *v, Coproc *c, uint32_t op) {
    uint64_t wait_val = c->counter;
    uint64_t sig_val = c->counter + 1;
    VkTimelineSemaphoreSubmitInfo tssi = {.sType=VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount=1,.pWaitSemaphoreValues=&wait_val,
        .signalSemaphoreValueCount=1,.pSignalSemaphoreValues=&sig_val};
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.pNext=&tssi,
        .waitSemaphoreCount=1,.pWaitSemaphores=&c->timeline,.pWaitDstStageMask=&ws,
        .commandBufferCount=1,.pCommandBuffers=&c->cmds[op],
        .signalSemaphoreCount=1,.pSignalSemaphores=&c->timeline};
    double t0 = now_us();
    CHECK_VK(vkQueueSubmit(v->queue, 1, &si, VK_NULL_HANDLE));
    VkSemaphoreWaitInfo swi = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount=1,.pSemaphores=&c->timeline,.pValues=&sig_val};
    CHECK_VK(vkWaitSemaphores(v->dev, &swi, UINT64_MAX));
    c->counter = sig_val;
    return now_us() - t0;
}

/* ── Gated dispatch for concurrent CPU+GPU tests (7-8) ── */
/* presubmit: queue work that waits on counter+1 (a future host signal), signals counter+2 */
static void coproc_presubmit(Vk *v, Coproc *c, uint32_t op) {
    uint64_t wait_val = c->counter + 1; /* GPU waits for host to signal this */
    uint64_t sig_val = c->counter + 2;  /* GPU signals this when done */
    VkTimelineSemaphoreSubmitInfo tssi = {.sType=VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount=1,.pWaitSemaphoreValues=&wait_val,
        .signalSemaphoreValueCount=1,.pSignalSemaphoreValues=&sig_val};
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.pNext=&tssi,
        .waitSemaphoreCount=1,.pWaitSemaphores=&c->timeline,.pWaitDstStageMask=&ws,
        .commandBufferCount=1,.pCommandBuffers=&c->cmds[op],
        .signalSemaphoreCount=1,.pSignalSemaphores=&c->timeline};
    CHECK_VK(vkQueueSubmit(v->queue, 1, &si, VK_NULL_HANDLE));
}
/* kick: host signals counter+1, releasing the GPU */
static void coproc_kick(Vk *v, Coproc *c) {
    VkSemaphoreSignalInfo ssi = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore=c->timeline,.value=c->counter + 1};
    CHECK_VK(vkSignalSemaphore(v->dev, &ssi));
}
/* wait: host waits on counter+2 (GPU completion), then advances counter by 2 */
static void coproc_wait(Vk *v, Coproc *c) {
    uint64_t val = c->counter + 2;
    VkSemaphoreWaitInfo swi = {.sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount=1,.pSemaphores=&c->timeline,.pValues=&val};
    CHECK_VK(vkWaitSemaphores(v->dev, &swi, UINT64_MAX));
    c->counter = val;
}
/* Full gated round-trip (for warmup of gated path) */
static double coproc_roundtrip(Vk *v, Coproc *c, uint32_t op) {
    coproc_presubmit(v, c, op);
    double t0 = now_us();
    coproc_kick(v, c);
    coproc_wait(v, c);
    return now_us() - t0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TI ti; const uint8_t *wgt;
    find_tensor(data, "blk.0.ffn_gate.weight", &ti, &wgt);
    int K = (int)ti.dims[0], N = (int)ti.dims[1];

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  COPROCESSOR V2: PowerVR as Fabric Chip\n");
    printf("  Timeline semaphore dispatch, pre-recorded command buffers\n");
    printf("  1024-element operations (D_MODEL), single workgroup\n");
    printf("══════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    pin(6);
    printf("  [init] pinned to cpu6\n"); fflush(stdout);
    Vk v; init_vk(&v);
    printf("  [init] Vulkan ready\n"); fflush(stdout);
    Coproc c; init_coproc(&v, &c, 1024);
    printf("  [init] coprocessor ready\n"); fflush(stdout);

    /* Fill data buffer */
    srand(42);
    float *dbuf = (float*)c.data_buf.map;
    for (int i = 0; i < 2048; i++) dbuf[i] = ((float)rand()/(float)0x7fffffff - 0.5f) * 4.0f;

    int WARMUP = 50;
    int ITERS = 500;
    double *t = calloc(ITERS, sizeof(double));

    /* Test 1: NOP */
    printf("\n  === Test 1: NOP (timeline round-trip only) ===\n"); fflush(stdout);
    for (int i = 0; i < WARMUP; i++) coproc_fire(&v, &c, 0);
    for (int i = 0; i < ITERS; i++) t[i] = coproc_fire(&v, &c, 0);
    pstats("NOP (timeline)", t, ITERS);

    /* Test 2: SiLU 1024 */
    printf("\n  === Test 2: SiLU (1024 elements) ===\n"); fflush(stdout);
    for (int i = 0; i < WARMUP; i++) coproc_fire(&v, &c, 1);
    for (int i = 0; i < ITERS; i++) t[i] = coproc_fire(&v, &c, 1);
    pstats("SiLU 1024", t, ITERS);

    /* Test 3: RMSNorm 1024 */
    printf("\n  === Test 3: RMSNorm (1024 elements) ===\n"); fflush(stdout);
    for (int i = 1024; i < 2048; i++) dbuf[i] = 1.0f; /* weights */
    for (int i = 0; i < WARMUP; i++) coproc_fire(&v, &c, 2);
    for (int i = 0; i < ITERS; i++) t[i] = coproc_fire(&v, &c, 2);
    pstats("RMSNorm 1024", t, ITERS);

    /* Test 4: SwiGLU 1024 */
    printf("\n  === Test 4: SwiGLU (1024 gate + 1024 up) ===\n"); fflush(stdout);
    for (int i = 0; i < WARMUP; i++) coproc_fire(&v, &c, 3);
    for (int i = 0; i < ITERS; i++) t[i] = coproc_fire(&v, &c, 3);
    pstats("SwiGLU 1024", t, ITERS);

    /* Test 5: Softmax 1024 */
    printf("\n  === Test 5: Softmax (1024 elements) ===\n"); fflush(stdout);
    for (int i = 0; i < WARMUP; i++) coproc_fire(&v, &c, 4);
    for (int i = 0; i < ITERS; i++) t[i] = coproc_fire(&v, &c, 4);
    pstats("Softmax 1024", t, ITERS);

    /* Test 6: CPU matvec baseline */
    printf("\n  === Test 6: CPU matvec alone [4608x1024] ===\n"); fflush(stdout);
    int8_t *act = calloc(K, 1);
    float *cpu_out = calloc(N, sizeof(float));
    volatile float sink = 0;
    for (int i = 0; i < K; i++) act[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < 20; i++) { matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0]; }
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_us();
        matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0];
        t[i] = now_us() - t0;
    }
    pstats("CPU matvec alone", t, ITERS);

    /* Test 7: CPU matvec + concurrent GPU SiLU */
    printf("\n  === Test 7: CPU matvec + GPU SiLU (concurrent) ===\n");
    printf("  KEY TEST: does GPU on-chip work steal DRAM bandwidth?\n"); fflush(stdout);
    for (int i = 0; i < 20; i++) coproc_roundtrip(&v, &c, 1);
    double *t_wall = calloc(ITERS, sizeof(double));
    double *t_extra = calloc(ITERS, sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        /* Pre-submit GPU work */
        coproc_presubmit(&v, &c, 1);
        double t0 = now_us();
        /* Kick GPU and start CPU matvec simultaneously */
        coproc_kick(&v, &c);
        matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0];
        double t_cpu = now_us();
        /* Wait for GPU */
        coproc_wait(&v, &c);
        double t_done = now_us();
        t_wall[i] = t_done - t0;
        t_extra[i] = t_done - t_cpu; /* wait time after CPU done */
    }
    pstats("Concurrent wall (CPU+GPU)", t_wall, ITERS);
    pstats("GPU extra wait after CPU", t_extra, ITERS);

    /* Test 8: CPU matvec + concurrent GPU NOP */
    printf("\n  === Test 8: CPU matvec + GPU NOP (dispatch contention only) ===\n"); fflush(stdout);
    for (int i = 0; i < 20; i++) coproc_roundtrip(&v, &c, 0);
    for (int i = 0; i < ITERS; i++) {
        coproc_presubmit(&v, &c, 0);
        double t0 = now_us();
        coproc_kick(&v, &c);
        matvec_q4(cpu_out, wgt, act, N, K); sink += cpu_out[0];
        double t_cpu = now_us();
        coproc_wait(&v, &c);
        double t_done = now_us();
        t_wall[i] = t_done - t0;
        t_extra[i] = t_done - t_cpu;
    }
    pstats("Concurrent wall (CPU+NOP)", t_wall, ITERS);
    pstats("NOP extra wait after CPU", t_extra, ITERS);

    printf("\n  ── Summary ──\n");
    printf("  If Test 7 wall ≈ Test 6: GPU SiLU is FREE (no bus contention)\n");
    printf("  If Test 7 wall > Test 6: GPU steals bandwidth even for on-chip work\n");
    printf("  (sink=%.1f)\n", (double)sink);
    printf("\n══════════════════════════════════════════════════════════════\n\n");

    free(t); free(t_wall); free(t_extra); free(act); free(cpu_out);
    munmap((void*)data, st.st_size); close(fd);
    return 0;
}
