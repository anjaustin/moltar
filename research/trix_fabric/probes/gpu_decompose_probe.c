/*
 * gpu_decompose_probe.c — Where is the GPU time going?
 *
 * Three shaders, same dispatch shape (N=1152, B=64):
 *   1. TILED (real):    shmem weight load + activation read + FMA
 *   2. READONLY:        shmem weight load + activation read + ADD (no multiply)
 *   3. NOP:             just write zeros
 *
 * If TILED ≈ NOP → dispatch overhead dominates (GPU isn't computing)
 * If TILED ≈ READONLY >> NOP → memory bandwidth dominates (GPU loads but barely computes)
 * If TILED >> READONLY >> NOP → GPU is actually doing real FMA work
 *
 * Build:
 *   glslc --target-env=vulkan1.1 -o vision_matmul_tiled.spv vision_matmul_tiled.comp
 *   glslc --target-env=vulkan1.1 -o vision_matmul_readonly.spv vision_matmul_readonly.comp
 *   glslc --target-env=vulkan1.1 -o vision_matmul_nop.spv vision_matmul_nop.comp
 *   $CC -O2 -march=armv8.2-a+dotprod+fp16 -o gpu_decompose_probe gpu_decompose_probe.c -lvulkan -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
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
static int cmp_dbl(const void *a, const void *b) {
    double da=*(const double*)a, db=*(const double*)b; return (da>db)-(da<db); }
static void pstats(const char *l, double *t, int n) {
    double s=0,mn=1e18,mx=0;
    for(int i=0;i<n;i++){s+=t[i];if(t[i]<mn)mn=t[i];if(t[i]>mx)mx=t[i];}
    double q[n]; memcpy(q,t,n*sizeof(double)); qsort(q,n,sizeof(double),cmp_dbl);
    printf("  %-40s mean=%9.1f  min=%9.1f  p50=%9.1f  max=%9.1f us\n",
           l, s/n, mn, q[n/2], mx);
}

static uint16_t f32_to_f16(float v) {
    uint32_t x; memcpy(&x, &v, 4);
    uint16_t s = (x >> 16) & 0x8000;
    int e = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7C00;
    return s | (e << 10) | (m >> 13);
}

/* ── Vulkan infra (same as other probes) ── */
typedef struct {
    VkInstance inst; VkPhysicalDevice pdev; VkDevice dev;
    uint32_t qf; VkQueue queue; VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties mprops;
} Vk;
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; VkDeviceSize sz; } Buf;

static uint32_t find_mem(Vk *v,uint32_t bits,VkMemoryPropertyFlags f){
    for(uint32_t i=0;i<v->mprops.memoryTypeCount;i++)
        if((bits&(1u<<i))&&(v->mprops.memoryTypes[i].propertyFlags&f)==f) return i;
    exit(1);
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
    const char *exts[]={"VK_KHR_16bit_storage","VK_KHR_shader_float16_int8"};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&f16math,
        .queueCreateInfoCount=1,.pQueueCreateInfos=&qci,.enabledExtensionCount=2,.ppEnabledExtensionNames=exts};
    CHECK_VK(vkCreateDevice(v->pdev,&dci,NULL,&v->dev));
    vkGetDeviceQueue(v->dev,v->qf,0,&v->queue);
    VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=v->qf};
    CHECK_VK(vkCreateCommandPool(v->dev,&pci,NULL,&v->pool));
}

typedef struct { VkPipeline pipe; VkPipelineLayout layout; VkDescriptorSet ds;
    VkDescriptorSetLayout dsl; VkDescriptorPool dp; } Pipe;

static Pipe make_pipe(Vk *v, const char *spv, Buf *bufs) {
    Pipe p;
    FILE *f=fopen(spv,"rb"); if(!f){fprintf(stderr,"No %s\n",spv);exit(1);}
    fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
    uint32_t *code=malloc(sz);fread(code,1,sz,f);fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=code};
    CHECK_VK(vkCreateShaderModule(v->dev,&smci,NULL,&sm));free(code);

    VkDescriptorSetLayoutBinding binds[5];
    for(int i=0;i<5;i++) binds[i]=(VkDescriptorSetLayoutBinding){i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,NULL};
    VkDescriptorSetLayoutCreateInfo dslci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=5,.pBindings=binds};
    CHECK_VK(vkCreateDescriptorSetLayout(v->dev,&dslci,NULL,&p.dsl));

    VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,16};
    VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&p.dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
    CHECK_VK(vkCreatePipelineLayout(v->dev,&plci,NULL,&p.layout));

    VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},.layout=p.layout};
    CHECK_VK(vkCreateComputePipelines(v->dev,VK_NULL_HANDLE,1,&cpci,NULL,&p.pipe));
    vkDestroyShaderModule(v->dev,sm,NULL);

    VkDescriptorPoolSize dps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,5};
    VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&dps};
    CHECK_VK(vkCreateDescriptorPool(v->dev,&dpci,NULL,&p.dp));
    VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=p.dp,.descriptorSetCount=1,.pSetLayouts=&p.dsl};
    CHECK_VK(vkAllocateDescriptorSets(v->dev,&dsai,&p.ds));

    VkWriteDescriptorSet wds[5]; VkDescriptorBufferInfo dbis[5];
    for(int i=0;i<5;i++){
        dbis[i]=(VkDescriptorBufferInfo){bufs[i].buf,0,bufs[i].sz};
        wds[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=p.ds,.dstBinding=i,
            .descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbis[i]};
    }
    vkUpdateDescriptorSets(v->dev,5,wds,0,NULL);
    return p;
}

static double fire(Vk *v, Pipe *p, uint32_t pc[4], uint32_t dx, uint32_t dy) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    CHECK_VK(vkAllocateCommandBuffers(v->dev,&cbai,&cmd));
    VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK_VK(vkBeginCommandBuffer(cmd,&bi));
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p->pipe);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p->layout,0,1,&p->ds,0,NULL);
    vkCmdPushConstants(cmd,p->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);
    vkCmdDispatch(cmd,dx,dy,1);
    CHECK_VK(vkEndCommandBuffer(cmd));

    VkFence fence;
    VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK_VK(vkCreateFence(v->dev,&fci,NULL,&fence));
    VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};

    double t0=now_us();
    CHECK_VK(vkQueueSubmit(v->queue,1,&si,fence));
    CHECK_VK(vkWaitForFences(v->dev,1,&fence,VK_TRUE,UINT64_MAX));
    double elapsed=now_us()-t0;

    vkDestroyFence(v->dev,fence,NULL);
    vkFreeCommandBuffers(v->dev,v->pool,1,&cmd);
    return elapsed;
}

int main(void) {
    const uint32_t N=1152, K=1152, B=64;

    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("  GPU TIME DECOMPOSITION: Where does the time go?\n");
    printf("  N=%u K=%u B=%u  dispatch(%u,1,1) = %u workgroups\n", N,K,B,N,N);
    printf("══════════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);

    pin(6);
    Vk v; init_vk(&v);
    printf("  [init] Vulkan ready\n"); fflush(stdout);

    VkBufferUsageFlags usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    Buf bufs[5];
    bufs[0]=make_buf(&v,(VkDeviceSize)N*K*2,usage);
    bufs[1]=make_buf(&v,(VkDeviceSize)B*K*2,usage);
    bufs[2]=make_buf(&v,(VkDeviceSize)B*N*4,usage);
    bufs[3]=make_buf(&v,N*K,usage);
    bufs[4]=make_buf(&v,256*2,usage);

    /* Fill with random data */
    srand(42);
    uint16_t *w=(uint16_t*)bufs[0].map;
    for(uint32_t i=0;i<N*K;i++) w[i]=f32_to_f16(((float)rand()/(float)RAND_MAX-0.5f)*0.1f);
    uint16_t *x=(uint16_t*)bufs[1].map;
    for(uint32_t i=0;i<B*K;i++) x[i]=f32_to_f16(((float)rand()/(float)RAND_MAX-0.5f)*2.0f);

    Pipe p_tiled=make_pipe(&v,"vision_matmul_tiled.spv",bufs);
    Pipe p_readonly=make_pipe(&v,"vision_matmul_readonly.spv",bufs);
    Pipe p_nop=make_pipe(&v,"vision_matmul_nop.spv",bufs);
    printf("  [init] 3 pipelines ready\n\n"); fflush(stdout);

    int WARMUP=5, ITERS=20;
    double *t=calloc(ITERS,sizeof(double));
    uint32_t pc[4]={2,N,K,B};

    /* Also test v1 naive for reference */
    Pipe p_v1=make_pipe(&v,"vision_matmul.spv",bufs);
    printf("  ═══ v1 Naive (dispatch %u,%u,1 = %u workgroups) ═══\n",N,B,N*B);
    uint32_t pc_v1[4]={0,N,K,B};
    for(int i=0;i<WARMUP;i++) fire(&v,&p_v1,pc_v1,N,B);
    for(int i=0;i<ITERS;i++) t[i]=fire(&v,&p_v1,pc_v1,N,B);
    pstats("v1 naive (full matmul)",t,ITERS);
    double v1_mean=0;for(int i=0;i<ITERS;i++)v1_mean+=t[i];v1_mean/=ITERS;
    printf("\n");fflush(stdout);

    printf("  ═══ All tiled variants (dispatch %u,1,1 = %u workgroups) ═══\n\n",N,N);
    fflush(stdout);

    for(int i=0;i<WARMUP;i++) fire(&v,&p_nop,pc,N,1);
    for(int i=0;i<ITERS;i++) t[i]=fire(&v,&p_nop,pc,N,1);
    pstats("NOP (write zeros only)",t,ITERS);
    double nop_mean=0;for(int i=0;i<ITERS;i++)nop_mean+=t[i];nop_mean/=ITERS;

    for(int i=0;i<WARMUP;i++) fire(&v,&p_readonly,pc,N,1);
    for(int i=0;i<ITERS;i++) t[i]=fire(&v,&p_readonly,pc,N,1);
    pstats("READONLY (load W+X, add only)",t,ITERS);
    double ro_mean=0;for(int i=0;i<ITERS;i++)ro_mean+=t[i];ro_mean/=ITERS;

    for(int i=0;i<WARMUP;i++) fire(&v,&p_tiled,pc,N,1);
    for(int i=0;i<ITERS;i++) t[i]=fire(&v,&p_tiled,pc,N,1);
    pstats("TILED (load W+X, FMA)",t,ITERS);
    double tiled_mean=0;for(int i=0;i<ITERS;i++)tiled_mean+=t[i];tiled_mean/=ITERS;

    /* Decomposition */
    double dispatch_pct = nop_mean / tiled_mean * 100;
    double load_pct = (ro_mean - nop_mean) / tiled_mean * 100;
    double compute_pct = (tiled_mean - ro_mean) / tiled_mean * 100;

    printf("\n  ═══ TIME DECOMPOSITION (B=%u) ═══\n\n", B);
    printf("  v1 naive:       %9.1f us  (%.1f ms)\n", v1_mean, v1_mean/1e3);
    printf("  NOP:            %9.1f us  (%.1f ms)  = dispatch + write zeros\n", nop_mean, nop_mean/1e3);
    printf("  READONLY:       %9.1f us  (%.1f ms)  = dispatch + memory loads\n", ro_mean, ro_mean/1e3);
    printf("  TILED (real):   %9.1f us  (%.1f ms)  = dispatch + loads + FMA\n", tiled_mean, tiled_mean/1e3);
    printf("\n");
    printf("  Dispatch overhead:  %5.1f%%  (%.1f ms)\n", dispatch_pct, nop_mean/1e3);
    printf("  Memory loads:       %5.1f%%  (%.1f ms)\n", load_pct, (ro_mean-nop_mean)/1e3);
    printf("  Actual compute:     %5.1f%%  (%.1f ms)\n", compute_pct, (tiled_mean-ro_mean)/1e3);
    printf("\n");
    if (compute_pct < 10.0)
        printf("  VERDICT: GPU is NOT meaningfully computing. %.0f%% dispatch + %.0f%% memory.\n",
               dispatch_pct, load_pct);
    else if (compute_pct < 30.0)
        printf("  VERDICT: GPU is lightly computing (%.0f%%). Mostly memory-bound (%.0f%%).\n",
               compute_pct, load_pct);
    else
        printf("  VERDICT: GPU IS computing (%.0f%% of time in FMA).\n", compute_pct);

    printf("\n══════════════════════════════════════════════════════════════════\n\n");

    free(t);
    return 0;
}
