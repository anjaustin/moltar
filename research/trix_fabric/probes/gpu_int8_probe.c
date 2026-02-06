/*
 * gpu_int8_probe.c — Test branchless INT8 ternary GEMM on GPU
 *
 * Compares:
 * 1. Original FP16 shader (gpu_ternary_gemm.spv)
 * 2. Branchless INT8 shader (gpu_ternary_branchless.spv)
 * 3. CPU NEON baseline
 *
 * Build:
 *   glslc --target-env=vulkan1.1 gpu_ternary_branchless.comp -o gpu_ternary_branchless.spv
 *   $NDK/.../aarch64-linux-android28-clang -O2 -march=armv8.2-a+dotprod \
 *       -o gpu_int8_probe gpu_int8_probe.c -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <sched.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* Minimal Vulkan types */
typedef uint32_t VkFlags, VkBool32;
typedef uint64_t VkDeviceSize;
typedef struct VkInstance_T* VkInstance;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
typedef struct VkDevice_T* VkDevice;
typedef struct VkQueue_T* VkQueue;
typedef struct VkBuffer_T* VkBuffer;
typedef struct VkDeviceMemory_T* VkDeviceMemory;
typedef struct VkShaderModule_T* VkShaderModule;
typedef struct VkPipelineLayout_T* VkPipelineLayout;
typedef struct VkPipeline_T* VkPipeline;
typedef struct VkDescriptorSetLayout_T* VkDescriptorSetLayout;
typedef struct VkDescriptorPool_T* VkDescriptorPool;
typedef struct VkDescriptorSet_T* VkDescriptorSet;
typedef struct VkCommandPool_T* VkCommandPool;
typedef struct VkCommandBuffer_T* VkCommandBuffer;
typedef struct VkFence_T* VkFence;

typedef enum { VK_SUCCESS = 0 } VkResult;
#define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO 2
#define VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO 3
#define VK_STRUCTURE_TYPE_SUBMIT_INFO 4
#define VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO 5
#define VK_STRUCTURE_TYPE_FENCE_CREATE_INFO 8
#define VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO 12
#define VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO 16
#define VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO 29
#define VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO 30
#define VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO 32
#define VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO 33
#define VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO 34
#define VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET 35
#define VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO 39
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO 40
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO 42
#define VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO 18
#define VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 0x20
#define VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 0x02
#define VK_MEMORY_PROPERTY_HOST_COHERENT_BIT 0x04
#define VK_DESCRIPTOR_TYPE_STORAGE_BUFFER 7
#define VK_SHADER_STAGE_COMPUTE_BIT 0x20
#define VK_PIPELINE_BIND_POINT_COMPUTE 1
#define VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT 1
#define VK_COMMAND_BUFFER_LEVEL_PRIMARY 0
#define VK_SHARING_MODE_EXCLUSIVE 0
#define VK_QUEUE_COMPUTE_BIT 2
#define VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT 2
#define VK_API_VERSION_1_1 ((1<<22)|(1<<12))
#define VK_NULL_HANDLE NULL

/* Structures - minimal versions */
typedef struct { uint32_t sType; const void *pNext; const char *pAppName; uint32_t appVer; const char *pEngName; uint32_t engVer; uint32_t apiVer; } VkApplicationInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; const VkApplicationInfo *pAppInfo; uint32_t layerCnt; const char *const *ppLayers; uint32_t extCnt; const char *const *ppExts; } VkInstanceCreateInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; uint32_t qfi; uint32_t qCnt; const float *pPrio; } VkDeviceQueueCreateInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; uint32_t qciCnt; const VkDeviceQueueCreateInfo *pQcis; uint32_t layerCnt; const char *const *ppLayers; uint32_t extCnt; const char *const *ppExts; const void *pFeats; } VkDeviceCreateInfo;
typedef struct { uint32_t qFlags; uint32_t qCnt; uint32_t tsBits; uint32_t gran[3]; } VkQueueFamilyProperties;
typedef struct { uint32_t propFlags; uint32_t heapIdx; } VkMemoryType;
typedef struct { uint64_t sz; uint32_t flags; } VkMemoryHeap;
typedef struct { uint32_t mtCnt; VkMemoryType mt[32]; uint32_t mhCnt; VkMemoryHeap mh[16]; } VkPhysicalDeviceMemoryProperties;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; uint64_t sz; uint32_t usage; uint32_t sharingMode; uint32_t qfiCnt; const uint32_t *pQfis; } VkBufferCreateInfo;
typedef struct { uint64_t sz; uint64_t align; uint32_t memBits; } VkMemoryRequirements;
typedef struct { uint32_t sType; const void *pNext; uint64_t allocSz; uint32_t mti; } VkMemoryAllocateInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; size_t codeSz; const uint32_t *pCode; } VkShaderModuleCreateInfo;
typedef struct { uint32_t binding; uint32_t descType; uint32_t descCnt; uint32_t stageFlags; const void *pSamplers; } VkDescriptorSetLayoutBinding;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; uint32_t bindCnt; const VkDescriptorSetLayoutBinding *pBinds; } VkDescriptorSetLayoutCreateInfo;
typedef struct { uint32_t stageFlags; uint32_t offset; uint32_t sz; } VkPushConstantRange;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; uint32_t setCnt; const VkDescriptorSetLayout *pSets; uint32_t pcrCnt; const VkPushConstantRange *pPcrs; } VkPipelineLayoutCreateInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; uint32_t stage; VkShaderModule mod; const char *pName; const void *pSpec; } VkPipelineShaderStageCreateInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; VkPipelineShaderStageCreateInfo stage; VkPipelineLayout layout; VkPipeline basePipe; int32_t baseIdx; } VkComputePipelineCreateInfo;
typedef struct { uint32_t type; uint32_t descCnt; } VkDescriptorPoolSize;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; uint32_t maxSets; uint32_t psCnt; const VkDescriptorPoolSize *pPs; } VkDescriptorPoolCreateInfo;
typedef struct { uint32_t sType; const void *pNext; VkDescriptorPool dp; uint32_t dsCnt; const VkDescriptorSetLayout *pSets; } VkDescriptorSetAllocateInfo;
typedef struct { VkBuffer buf; uint64_t off; uint64_t range; } VkDescriptorBufferInfo;
typedef struct { uint32_t sType; const void *pNext; VkDescriptorSet dstSet; uint32_t dstBind; uint32_t dstElem; uint32_t descCnt; uint32_t descType; const void *pImg; const VkDescriptorBufferInfo *pBuf; const void *pTex; } VkWriteDescriptorSet;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; uint32_t qfi; } VkCommandPoolCreateInfo;
typedef struct { uint32_t sType; const void *pNext; VkCommandPool cp; uint32_t level; uint32_t cbCnt; } VkCommandBufferAllocateInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; const void *pInh; } VkCommandBufferBeginInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t wsCnt; const void *pWs; const uint32_t *pWsMask; uint32_t cbCnt; const VkCommandBuffer *pCbs; uint32_t ssCnt; const void *pSs; } VkSubmitInfo;
typedef struct { uint32_t sType; const void *pNext; uint32_t flags; } VkFenceCreateInfo;

/* Function pointers */
static void *vk_lib;

typedef VkResult (*PFN_vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
typedef void (*PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
typedef void (*PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);
typedef VkResult (*PFN_vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
typedef void (*PFN_vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue*);
typedef VkResult (*PFN_vkCreateBuffer)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
typedef void (*PFN_vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements*);
typedef VkResult (*PFN_vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
typedef VkResult (*PFN_vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, uint64_t);
typedef VkResult (*PFN_vkMapMemory)(VkDevice, VkDeviceMemory, uint64_t, uint64_t, uint32_t, void**);
typedef VkResult (*PFN_vkCreateShaderModule)(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*);
typedef void (*PFN_vkDestroyShaderModule)(VkDevice, VkShaderModule, const void*);
typedef VkResult (*PFN_vkCreateDescriptorSetLayout)(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*);
typedef VkResult (*PFN_vkCreatePipelineLayout)(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*);
typedef VkResult (*PFN_vkCreateComputePipelines)(VkDevice, void*, uint32_t, const VkComputePipelineCreateInfo*, const void*, VkPipeline*);
typedef VkResult (*PFN_vkCreateDescriptorPool)(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*);
typedef VkResult (*PFN_vkAllocateDescriptorSets)(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
typedef void (*PFN_vkUpdateDescriptorSets)(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const void*);
typedef VkResult (*PFN_vkCreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*);
typedef VkResult (*PFN_vkAllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
typedef VkResult (*PFN_vkBeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo*);
typedef void (*PFN_vkCmdBindPipeline)(VkCommandBuffer, uint32_t, VkPipeline);
typedef void (*PFN_vkCmdBindDescriptorSets)(VkCommandBuffer, uint32_t, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*);
typedef void (*PFN_vkCmdPushConstants)(VkCommandBuffer, VkPipelineLayout, uint32_t, uint32_t, uint32_t, const void*);
typedef void (*PFN_vkCmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
typedef VkResult (*PFN_vkEndCommandBuffer)(VkCommandBuffer);
typedef VkResult (*PFN_vkCreateFence)(VkDevice, const VkFenceCreateInfo*, const void*, VkFence*);
typedef VkResult (*PFN_vkQueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
typedef VkResult (*PFN_vkWaitForFences)(VkDevice, uint32_t, const VkFence*, uint32_t, uint64_t);
typedef VkResult (*PFN_vkResetFences)(VkDevice, uint32_t, const VkFence*);
typedef void (*PFN_vkFreeCommandBuffers)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*);

static PFN_vkCreateInstance pvkCreateInstance;
static PFN_vkEnumeratePhysicalDevices pvkEnumeratePhysicalDevices;
static PFN_vkGetPhysicalDeviceMemoryProperties pvkGetPhysicalDeviceMemoryProperties;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties pvkGetPhysicalDeviceQueueFamilyProperties;
static PFN_vkCreateDevice pvkCreateDevice;
static PFN_vkGetDeviceQueue pvkGetDeviceQueue;
static PFN_vkCreateBuffer pvkCreateBuffer;
static PFN_vkGetBufferMemoryRequirements pvkGetBufferMemoryRequirements;
static PFN_vkAllocateMemory pvkAllocateMemory;
static PFN_vkBindBufferMemory pvkBindBufferMemory;
static PFN_vkMapMemory pvkMapMemory;
static PFN_vkCreateShaderModule pvkCreateShaderModule;
static PFN_vkDestroyShaderModule pvkDestroyShaderModule;
static PFN_vkCreateDescriptorSetLayout pvkCreateDescriptorSetLayout;
static PFN_vkCreatePipelineLayout pvkCreatePipelineLayout;
static PFN_vkCreateComputePipelines pvkCreateComputePipelines;
static PFN_vkCreateDescriptorPool pvkCreateDescriptorPool;
static PFN_vkAllocateDescriptorSets pvkAllocateDescriptorSets;
static PFN_vkUpdateDescriptorSets pvkUpdateDescriptorSets;
static PFN_vkCreateCommandPool pvkCreateCommandPool;
static PFN_vkAllocateCommandBuffers pvkAllocateCommandBuffers;
static PFN_vkBeginCommandBuffer pvkBeginCommandBuffer;
static PFN_vkCmdBindPipeline pvkCmdBindPipeline;
static PFN_vkCmdBindDescriptorSets pvkCmdBindDescriptorSets;
static PFN_vkCmdPushConstants pvkCmdPushConstants;
static PFN_vkCmdDispatch pvkCmdDispatch;
static PFN_vkEndCommandBuffer pvkEndCommandBuffer;
static PFN_vkCreateFence pvkCreateFence;
static PFN_vkQueueSubmit pvkQueueSubmit;
static PFN_vkWaitForFences pvkWaitForFences;
static PFN_vkResetFences pvkResetFences;
static PFN_vkFreeCommandBuffers pvkFreeCommandBuffers;

#define LOAD_VK(name) do { p##name = (PFN_##name)dlsym(vk_lib, #name); if (!p##name) { fprintf(stderr, "Missing: %s\n", #name); return -1; } } while(0)

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

#ifdef __linux__
static void pin_cpu(int cpu) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}
#else
static void pin_cpu(int cpu) { (void)cpu; }
#endif

/* Vulkan state */
typedef struct {
    VkInstance inst;
    VkPhysicalDevice pdev;
    VkDevice dev;
    uint32_t qf;
    VkQueue queue;
    VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties mprops;
} Vk;

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; uint64_t sz; } Buf;

static int load_vulkan(void) {
    vk_lib = dlopen("libvulkan.so", RTLD_NOW);
    if (!vk_lib) vk_lib = dlopen("libvulkan.so.1", RTLD_NOW);
    if (!vk_lib) return -1;
    LOAD_VK(vkCreateInstance); LOAD_VK(vkEnumeratePhysicalDevices);
    LOAD_VK(vkGetPhysicalDeviceMemoryProperties); LOAD_VK(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_VK(vkCreateDevice); LOAD_VK(vkGetDeviceQueue);
    LOAD_VK(vkCreateBuffer); LOAD_VK(vkGetBufferMemoryRequirements);
    LOAD_VK(vkAllocateMemory); LOAD_VK(vkBindBufferMemory); LOAD_VK(vkMapMemory);
    LOAD_VK(vkCreateShaderModule); LOAD_VK(vkDestroyShaderModule);
    LOAD_VK(vkCreateDescriptorSetLayout); LOAD_VK(vkCreatePipelineLayout);
    LOAD_VK(vkCreateComputePipelines); LOAD_VK(vkCreateDescriptorPool);
    LOAD_VK(vkAllocateDescriptorSets); LOAD_VK(vkUpdateDescriptorSets);
    LOAD_VK(vkCreateCommandPool); LOAD_VK(vkAllocateCommandBuffers);
    LOAD_VK(vkBeginCommandBuffer); LOAD_VK(vkCmdBindPipeline);
    LOAD_VK(vkCmdBindDescriptorSets); LOAD_VK(vkCmdPushConstants);
    LOAD_VK(vkCmdDispatch); LOAD_VK(vkEndCommandBuffer);
    LOAD_VK(vkCreateFence); LOAD_VK(vkQueueSubmit);
    LOAD_VK(vkWaitForFences); LOAD_VK(vkResetFences); LOAD_VK(vkFreeCommandBuffers);
    return 0;
}

static uint32_t find_mem(Vk *v, uint32_t bits, uint32_t flags) {
    for (uint32_t i = 0; i < v->mprops.mtCnt; i++)
        if ((bits & (1u << i)) && (v->mprops.mt[i].propFlags & flags) == flags) return i;
    return 0;
}

static int make_buf(Vk *v, Buf *b, uint64_t sz) {
    b->sz = sz;
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, 0, 0};
    if (pvkCreateBuffer(v->dev, &ci, 0, &b->buf) != VK_SUCCESS) return -1;
    VkMemoryRequirements req;
    pvkGetBufferMemoryRequirements(v->dev, b->buf, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, req.sz,
        find_mem(v, req.memBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    if (pvkAllocateMemory(v->dev, &ai, 0, &b->mem) != VK_SUCCESS) return -1;
    pvkBindBufferMemory(v->dev, b->buf, b->mem, 0);
    pvkMapMemory(v->dev, b->mem, 0, sz, 0, &b->map);
    return 0;
}

static int init_vk(Vk *v) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "probe", 1, "probe", 1, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &app, 0, 0, 0, 0};
    if (pvkCreateInstance(&ici, 0, &v->inst) != VK_SUCCESS) return -1;
    uint32_t n = 1;
    pvkEnumeratePhysicalDevices(v->inst, &n, &v->pdev);
    pvkGetPhysicalDeviceMemoryProperties(v->pdev, &v->mprops);
    uint32_t qfc = 0;
    pvkGetPhysicalDeviceQueueFamilyProperties(v->pdev, &qfc, 0);
    VkQueueFamilyProperties *qfp = malloc(qfc * sizeof(*qfp));
    pvkGetPhysicalDeviceQueueFamilyProperties(v->pdev, &qfc, qfp);
    v->qf = 0;
    for (uint32_t i = 0; i < qfc; i++) if (qfp[i].qFlags & VK_QUEUE_COMPUTE_BIT) { v->qf = i; break; }
    free(qfp);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, v->qf, 1, &prio};
    
    /* Enable 8-bit storage */
    const char *exts[] = {"VK_KHR_8bit_storage", "VK_KHR_shader_float16_int8"};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0, 0, 1, &qci, 0, 0, 2, exts, 0};
    if (pvkCreateDevice(v->pdev, &dci, 0, &v->dev) != VK_SUCCESS) return -1;
    pvkGetDeviceQueue(v->dev, v->qf, 0, &v->queue);
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, v->qf};
    pvkCreateCommandPool(v->dev, &pci, 0, &v->pool);
    return 0;
}

/* Pipeline for branchless INT8 shader */
typedef struct {
    VkPipeline pipe;
    VkPipelineLayout layout;
    VkDescriptorSetLayout dsl;
    VkDescriptorPool dp;
    VkDescriptorSet ds;
    Buf weights, activations, scales, output;
    VkFence fence;
} Pipeline;

static int init_pipeline(Vk *v, Pipeline *p, const char *spv_file, uint32_t N, uint32_t K) {
    FILE *f = fopen(spv_file, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", spv_file); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sz); fread(spv, 1, sz, f); fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, sz, spv};
    if (pvkCreateShaderModule(v->dev, &smci, 0, &sm) != VK_SUCCESS) { free(spv); return -1; }
    free(spv);
    
    /* 4 bindings: weights, activations, scales, output */
    VkDescriptorSetLayoutBinding binds[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 4, binds};
    pvkCreateDescriptorSetLayout(v->dev, &dslci, 0, &p->dsl);
    
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &p->dsl, 1, &pcr};
    pvkCreatePipelineLayout(v->dev, &plci, 0, &p->layout);
    
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", 0},
        p->layout, 0, 0};
    pvkCreateComputePipelines(v->dev, 0, 1, &cpci, 0, &p->pipe);
    pvkDestroyShaderModule(v->dev, sm, 0);
    
    /* Buffers */
    make_buf(v, &p->weights, (uint64_t)N * K / 4);      /* packed trits */
    make_buf(v, &p->activations, K);                    /* int8 */
    make_buf(v, &p->scales, N * sizeof(float));         /* float per row */
    make_buf(v, &p->output, N * sizeof(float));         /* float output */
    
    /* Descriptor set */
    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, 1, 1, &dps};
    pvkCreateDescriptorPool(v->dev, &dpci, 0, &p->dp);
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, p->dp, 1, &p->dsl};
    pvkAllocateDescriptorSets(v->dev, &dsai, &p->ds);
    
    VkDescriptorBufferInfo dbis[4] = {
        {p->weights.buf, 0, p->weights.sz},
        {p->activations.buf, 0, p->activations.sz},
        {p->scales.buf, 0, p->scales.sz},
        {p->output.buf, 0, p->output.sz},
    };
    VkWriteDescriptorSet wds[4];
    for (int i = 0; i < 4; i++) {
        wds[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, p->ds, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &dbis[i], 0};
    }
    pvkUpdateDescriptorSets(v->dev, 4, wds, 0, 0);
    
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, 0, 0};
    pvkCreateFence(v->dev, &fci, 0, &p->fence);
    return 0;
}

static double dispatch(Vk *v, Pipeline *p, uint32_t N, uint32_t K) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, v->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    pvkAllocateCommandBuffers(v->dev, &cbai, &cmd);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0};
    pvkBeginCommandBuffer(cmd, &bi);
    pvkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    pvkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &p->ds, 0, 0);
    uint32_t pc[2] = {N, K};
    pvkCmdPushConstants(cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    pvkCmdDispatch(cmd, N, 1, 1);
    pvkEndCommandBuffer(cmd);
    
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cmd, 0, 0};
    double t0 = now_us();
    pvkQueueSubmit(v->queue, 1, &si, p->fence);
    pvkWaitForFences(v->dev, 1, &p->fence, 1, UINT64_MAX);
    double elapsed = now_us() - t0;
    pvkResetFences(v->dev, 1, &p->fence);
    pvkFreeCommandBuffers(v->dev, v->pool, 1, &cmd);
    return elapsed;
}

/* CPU reference */
#ifdef __ARM_NEON
static const int8_t TRIT_LUT[16] __attribute__((aligned(16))) = {0,1,-1,0, 0,1,-1,0, 0,1,-1,0, 0,1,-1,0};

static void cpu_ternary_gemv(float *out, const uint8_t *wgt, const int8_t *act, const float *scales, int N, int K) {
    int8x16_t lut = vld1q_s8(TRIT_LUT);
    uint8x16_t mask = vdupq_n_u8(0x03);
    
    for (int n = 0; n < N; n++) {
        int32_t sum = 0;
        const uint8_t *w = wgt + n * (K / 4);
        
        for (int k = 0; k < K; k += 64) {
            int8x16x4_t a = vld4q_s8(act + k);
            uint8x16_t wp = vld1q_u8(w + k / 4);
            
            int32x4_t acc0 = vdotq_s32(vdupq_n_s32(0), vqtbl1q_s8(lut, vandq_u8(wp, mask)), a.val[0]);
            int32x4_t acc1 = vdotq_s32(vdupq_n_s32(0), vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(wp, 2), mask)), a.val[1]);
            int32x4_t acc2 = vdotq_s32(vdupq_n_s32(0), vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(wp, 4), mask)), a.val[2]);
            int32x4_t acc3 = vdotq_s32(vdupq_n_s32(0), vqtbl1q_s8(lut, vshrq_n_u8(wp, 6)), a.val[3]);
            
            sum += vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
        }
        out[n] = (float)sum * scales[n];
    }
}
#endif

int main(int argc, char *argv[]) {
    uint32_t N = 4096, K = 1024;
    int iters = 30, warmup = 10;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    K = (K / 64) * 64;  /* Align to 64 */
    
    printf("\n========================================\n");
    printf("  GPU INT8 Branchless Ternary GEMM\n");
    printf("  N=%u, K=%u, iters=%d\n", N, K, iters);
    printf("========================================\n\n");
    
    /* Allocate data */
    uint8_t *weights = malloc(N * K / 4);
    int8_t *activations = malloc(K);
    float *scales = malloc(N * sizeof(float));
    float *out_cpu = malloc(N * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < N * K / 4; i++) weights[i] = rand() & 0xFF;
    for (int i = 0; i < (int)K; i++) activations[i] = (rand() % 256) - 128;
    for (int i = 0; i < (int)N; i++) scales[i] = 0.001f;
    
    pin_cpu(6);
    printf("  Pinned to CPU 6\n\n");
    
#ifdef __ARM_NEON
    printf("  CPU Baseline (NEON TBL+DOT):\n");
    volatile float sink = 0;
    for (int i = 0; i < warmup; i++) { cpu_ternary_gemv(out_cpu, weights, activations, scales, N, K); sink += out_cpu[0]; }
    
    double cpu_sum = 0;
    for (int i = 0; i < iters; i++) {
        double t0 = now_us();
        cpu_ternary_gemv(out_cpu, weights, activations, scales, N, K);
        sink += out_cpu[0];
        cpu_sum += now_us() - t0;
    }
    double cpu_mean = cpu_sum / iters;
    printf("    Mean: %.1f us\n", cpu_mean);
    printf("    (sink=%.2f)\n\n", (double)sink);
#endif
    
    /* GPU */
    if (load_vulkan() < 0) { printf("  No Vulkan\n"); return 1; }
    printf("  Vulkan loaded\n");
    
    Vk v;
    if (init_vk(&v) < 0) { printf("  Vulkan init failed\n"); return 1; }
    printf("  Vulkan ready (8-bit storage enabled)\n\n");
    
    Pipeline p;
    if (init_pipeline(&v, &p, "gpu_ternary_branchless.spv", N, K) < 0) {
        printf("  Pipeline init failed\n");
        return 1;
    }
    
    /* Upload data */
    memcpy(p.weights.map, weights, N * K / 4);
    memcpy(p.activations.map, activations, K);
    memcpy(p.scales.map, scales, N * sizeof(float));
    
    printf("  GPU Branchless INT8:\n");
    for (int i = 0; i < warmup; i++) dispatch(&v, &p, N, K);
    
    double gpu_sum = 0;
    for (int i = 0; i < iters; i++) gpu_sum += dispatch(&v, &p, N, K);
    double gpu_mean = gpu_sum / iters;
    printf("    Mean: %.1f us\n\n", gpu_mean);
    
    /* Compare */
    printf("========================================\n");
#ifdef __ARM_NEON
    printf("  CPU: %.1f us\n", cpu_mean);
#endif
    printf("  GPU: %.1f us\n", gpu_mean);
#ifdef __ARM_NEON
    printf("  Ratio: %.2fx %s\n", 
           gpu_mean > cpu_mean ? gpu_mean / cpu_mean : cpu_mean / gpu_mean,
           gpu_mean < cpu_mean ? "GPU faster" : "CPU faster");
#endif
    printf("========================================\n");
    
    free(weights); free(activations); free(scales); free(out_cpu);
    return 0;
}
