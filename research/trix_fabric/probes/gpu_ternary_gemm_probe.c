/*
 * gpu_ternary_gemm_probe.c — GPU Ternary GEMM + Spline Activation Benchmark
 *
 * Tests the hypothesis: GPU with 8 GB/s dedicated DRAM path can outperform
 * CPU for ternary GEMM when weights stream through GPU's buffer path.
 *
 * Architecture:
 *   1. 2-bit ternary weights packed (4 trits per byte)
 *   2. Per-block scales (32 weights per scale)
 *   3. Spline activation via buffer LUT with linear interpolation
 *   4. Compare to CPU baseline (NEON ternary kernels)
 *
 * Build:
 *   glslc gpu_ternary_gemm.comp -o gpu_ternary_gemm.spv
 *   $NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang \
 *       -O2 -march=armv8.2-a+dotprod -o gpu_ternary_gemm_probe \
 *       gpu_ternary_gemm_probe.c -lm
 *
 * Run:
 *   adb push gpu_ternary_gemm_probe gpu_ternary_gemm.spv /data/local/tmp/
 *   adb shell "cd /data/local/tmp && ./gpu_ternary_gemm_probe"
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

/* ══════════════════════════════════════════════════════════════════
 *  Vulkan Types (minimal definitions to avoid header dependency)
 * ══════════════════════════════════════════════════════════════════ */

typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
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

typedef enum VkResult {
    VK_SUCCESS = 0,
    VK_NOT_READY = 1,
    VK_TIMEOUT = 2,
} VkResult;

typedef enum VkStructureType {
    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
    VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
} VkStructureType;

typedef enum VkBufferUsageFlagBits {
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020,
} VkBufferUsageFlagBits;

typedef enum VkMemoryPropertyFlagBits {
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004,
} VkMemoryPropertyFlagBits;

typedef enum VkDescriptorType {
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
} VkDescriptorType;

typedef enum VkShaderStageFlagBits {
    VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020,
} VkShaderStageFlagBits;

typedef enum VkPipelineBindPoint {
    VK_PIPELINE_BIND_POINT_COMPUTE = 1,
} VkPipelineBindPoint;

typedef enum VkCommandBufferUsageFlagBits {
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001,
} VkCommandBufferUsageFlagBits;

typedef enum VkCommandBufferLevel {
    VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0,
} VkCommandBufferLevel;

typedef enum VkSharingMode {
    VK_SHARING_MODE_EXCLUSIVE = 0,
} VkSharingMode;

typedef enum VkQueueFlagBits {
    VK_QUEUE_COMPUTE_BIT = 0x00000002,
} VkQueueFlagBits;

typedef enum VkCommandPoolCreateFlagBits {
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002,
} VkCommandPoolCreateFlagBits;

#define VK_API_VERSION_1_0 ((1 << 22) | (0 << 12) | 0)
#define VK_NULL_HANDLE NULL

/* Structures */
typedef struct {
    VkStructureType sType;
    const void *pNext;
    const char *pApplicationName;
    uint32_t applicationVersion;
    const char *pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    const VkApplicationInfo *pApplicationInfo;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float *pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo *pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
    const void *pEnabledFeatures;
} VkDeviceCreateInfo;

typedef struct {
    VkFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    uint32_t minImageTransferGranularity[3];
} VkQueueFamilyProperties;

typedef struct {
    VkFlags propertyFlags;
    uint32_t heapIndex;
} VkMemoryType;

typedef struct {
    VkDeviceSize size;
    VkFlags flags;
} VkMemoryHeap;

typedef struct {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkFlags usage;
    VkSharingMode sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t *pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
} VkMemoryRequirements;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
} VkMemoryAllocateInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    size_t codeSize;
    const uint32_t *pCode;
} VkShaderModuleCreateInfo;

typedef struct {
    uint32_t binding;
    VkDescriptorType descriptorType;
    uint32_t descriptorCount;
    VkFlags stageFlags;
    const void *pImmutableSamplers;
} VkDescriptorSetLayoutBinding;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding *pBindings;
} VkDescriptorSetLayoutCreateInfo;

typedef struct {
    VkFlags stageFlags;
    uint32_t offset;
    uint32_t size;
} VkPushConstantRange;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout *pSetLayouts;
    uint32_t pushConstantRangeCount;
    const VkPushConstantRange *pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    VkShaderStageFlagBits stage;
    VkShaderModule module;
    const char *pName;
    const void *pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;

typedef struct {
    VkDescriptorType type;
    uint32_t descriptorCount;
} VkDescriptorPoolSize;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize *pPoolSizes;
} VkDescriptorPoolCreateInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout *pSetLayouts;
} VkDescriptorSetAllocateInfo;

typedef struct {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
} VkDescriptorBufferInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    VkDescriptorType descriptorType;
    const void *pImageInfo;
    const VkDescriptorBufferInfo *pBufferInfo;
    const void *pTexelBufferView;
} VkWriteDescriptorSet;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
} VkCommandPoolCreateInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkCommandPool commandPool;
    VkCommandBufferLevel level;
    uint32_t commandBufferCount;
} VkCommandBufferAllocateInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    const void *pInheritanceInfo;
} VkCommandBufferBeginInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    uint32_t waitSemaphoreCount;
    const void *pWaitSemaphores;
    const VkFlags *pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer *pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void *pSignalSemaphores;
} VkSubmitInfo;

typedef struct {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
} VkFenceCreateInfo;

/* Function pointer types */
typedef VkResult (*PFN_vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
typedef void (*PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
typedef void (*PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);
typedef VkResult (*PFN_vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
typedef void (*PFN_vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue*);
typedef VkResult (*PFN_vkCreateBuffer)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
typedef void (*PFN_vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements*);
typedef VkResult (*PFN_vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
typedef VkResult (*PFN_vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
typedef VkResult (*PFN_vkMapMemory)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
typedef void (*PFN_vkUnmapMemory)(VkDevice, VkDeviceMemory);
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
typedef void (*PFN_vkCmdBindPipeline)(VkCommandBuffer, VkPipelineBindPoint, VkPipeline);
typedef void (*PFN_vkCmdBindDescriptorSets)(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*);
typedef void (*PFN_vkCmdPushConstants)(VkCommandBuffer, VkPipelineLayout, VkFlags, uint32_t, uint32_t, const void*);
typedef void (*PFN_vkCmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
typedef VkResult (*PFN_vkEndCommandBuffer)(VkCommandBuffer);
typedef VkResult (*PFN_vkCreateFence)(VkDevice, const VkFenceCreateInfo*, const void*, VkFence*);
typedef VkResult (*PFN_vkQueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
typedef VkResult (*PFN_vkWaitForFences)(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t);
typedef VkResult (*PFN_vkResetFences)(VkDevice, uint32_t, const VkFence*);
typedef void (*PFN_vkFreeCommandBuffers)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*);
typedef void (*PFN_vkDestroyFence)(VkDevice, VkFence, const void*);
typedef void (*PFN_vkDestroyBuffer)(VkDevice, VkBuffer, const void*);
typedef void (*PFN_vkFreeMemory)(VkDevice, VkDeviceMemory, const void*);
typedef void (*PFN_vkDestroyPipeline)(VkDevice, VkPipeline, const void*);
typedef void (*PFN_vkDestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void*);
typedef void (*PFN_vkDestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void*);
typedef void (*PFN_vkDestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void*);
typedef void (*PFN_vkDestroyCommandPool)(VkDevice, VkCommandPool, const void*);
typedef void (*PFN_vkDestroyDevice)(VkDevice, const void*);
typedef void (*PFN_vkDestroyInstance)(VkInstance, const void*);

/* Global function pointers */
static void *vk_lib = NULL;
static PFN_vkCreateInstance vkCreateInstance;
static PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
static PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
static PFN_vkCreateDevice vkCreateDevice;
static PFN_vkGetDeviceQueue vkGetDeviceQueue;
static PFN_vkCreateBuffer vkCreateBuffer;
static PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
static PFN_vkAllocateMemory vkAllocateMemory;
static PFN_vkBindBufferMemory vkBindBufferMemory;
static PFN_vkMapMemory vkMapMemory;
static PFN_vkUnmapMemory vkUnmapMemory;
static PFN_vkCreateShaderModule vkCreateShaderModule;
static PFN_vkDestroyShaderModule vkDestroyShaderModule;
static PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout;
static PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
static PFN_vkCreateComputePipelines vkCreateComputePipelines;
static PFN_vkCreateDescriptorPool vkCreateDescriptorPool;
static PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets;
static PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets;
static PFN_vkCreateCommandPool vkCreateCommandPool;
static PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
static PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
static PFN_vkCmdBindPipeline vkCmdBindPipeline;
static PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
static PFN_vkCmdPushConstants vkCmdPushConstants;
static PFN_vkCmdDispatch vkCmdDispatch;
static PFN_vkEndCommandBuffer vkEndCommandBuffer;
static PFN_vkCreateFence vkCreateFence;
static PFN_vkQueueSubmit vkQueueSubmit;
static PFN_vkWaitForFences vkWaitForFences;
static PFN_vkResetFences vkResetFences;
static PFN_vkFreeCommandBuffers vkFreeCommandBuffers;
static PFN_vkDestroyFence vkDestroyFence;

#define LOAD_VK(name) do { \
    name = (PFN_##name)dlsym(vk_lib, #name); \
    if (!name) { fprintf(stderr, "Failed to load %s\n", #name); return -1; } \
} while(0)

#define CHECK_VK(call) do { \
    VkResult _r = (call); \
    if (_r != VK_SUCCESS) { fprintf(stderr, "VK error %d at %s:%d\n", _r, __FILE__, __LINE__); return -1; } \
} while(0)

/* ══════════════════════════════════════════════════════════════════
 *  Timing & Utilities
 * ══════════════════════════════════════════════════════════════════ */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

#ifdef __linux__
static void pin_cpu(int cpu) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(cpu, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}
#else
static void pin_cpu(int cpu) { (void)cpu; }
#endif

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
    uint32_t s = (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t f;
    if (!e) {
        if (!m) f = s;
        else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; f = s | ((e + 112) << 23) | (m << 13); }
    } else if (e == 31) f = s | 0x7F800000 | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}

/* ══════════════════════════════════════════════════════════════════
 *  CPU Reference: Ternary GEMM (from matvec_shootout)
 * ══════════════════════════════════════════════════════════════════ */

#ifdef __ARM_NEON

static const int8_t TRIT_DECODE_TABLE[16] __attribute__((aligned(16))) = {
    0, 1, -1, 0,
    0, 1, -1, 0,
    0, 1, -1, 0,
    0, 1, -1, 0
};

/* CPU ternary matvec with spline activation */
static void cpu_ternary_matvec_spline(
    float * __restrict__ out,
    const uint8_t * __restrict__ packed_wgt,  /* 2-bit packed */
    const int8_t * __restrict__ act_i8,
    const float * __restrict__ scales,        /* per-block */
    const float * __restrict__ spline_lut,    /* 256 entries */
    int N, int K
) {
    const int K_packed = K / 4;
    const int n_blocks_per_row = K / 32;

    int8x16_t lut = vld1q_s8(TRIT_DECODE_TABLE);
    uint8x16_t mask_03 = vdupq_n_u8(0x03);

    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        const uint8_t *w_ptr = packed_wgt + n * K_packed;
        const int8_t *a_ptr = act_i8;
        const float *s_ptr = scales + n * n_blocks_per_row;

        for (int k = 0; k < K; k += 64) {
            int8x16x4_t a_streams = vld4q_s8(a_ptr);
            a_ptr += 64;

            uint8x16_t w_packed = vld1q_u8(w_ptr);
            w_ptr += 16;

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            acc0 = vdotq_s32(acc0, vqtbl1q_s8(lut, vandq_u8(w_packed, mask_03)), a_streams.val[0]);
            acc1 = vdotq_s32(acc1, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w_packed, 2), mask_03)), a_streams.val[1]);
            acc2 = vdotq_s32(acc2, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(w_packed, 4), mask_03)), a_streams.val[2]);
            acc3 = vdotq_s32(acc3, vqtbl1q_s8(lut, vshrq_n_u8(w_packed, 6)), a_streams.val[3]);

            int32x4_t acc = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
            int32_t dot = vaddvq_s32(acc);

            int block_idx = k / 32;
            float avg_scale = (s_ptr[block_idx] + s_ptr[block_idx + 1]) * 0.5f;
            sum += avg_scale * (float)dot / 64.0f;
        }

        /* Spline activation: quantize to 8-bit index, lookup */
        int idx = (int)((sum + 4.0f) * 32.0f);  /* Map [-4, 4] to [0, 256] */
        if (idx < 0) idx = 0;
        if (idx > 255) idx = 255;
        out[n] = spline_lut[idx];
    }
}
#endif

/* ══════════════════════════════════════════════════════════════════
 *  Generate Test Data
 * ══════════════════════════════════════════════════════════════════ */

static void generate_ternary_weights(uint8_t *packed, float *scales, int N, int K) {
    srand(42);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t byte = 0;
            for (int i = 0; i < 4; i++) {
                int r = rand() % 3;
                uint8_t trit = (r == 0) ? 0 : (r == 1) ? 1 : 2;
                byte |= (trit << (i * 2));
            }
            packed[n * (K / 4) + k / 4] = byte;
        }
        for (int b = 0; b < K / 32; b++) {
            scales[n * (K / 32) + b] = ((float)rand() / RAND_MAX) * 0.1f;
        }
    }
}

static void generate_activations_i8(int8_t *act, int K) {
    srand(123);
    for (int k = 0; k < K; k++) {
        act[k] = (int8_t)((rand() % 256) - 128);
    }
}

static void generate_activations_f16(uint16_t *act_f16, const int8_t *act_i8, int K) {
    for (int k = 0; k < K; k++) {
        act_f16[k] = f32_to_f16((float)act_i8[k] / 127.0f);
    }
}

static void generate_spline_lut_f32(float *lut, int size) {
    /* SwiGLU: x * sigmoid(x) */
    for (int i = 0; i < size; i++) {
        float x = (i - size / 2) / (float)(size / 8);
        float sigmoid = 1.0f / (1.0f + expf(-x));
        lut[i] = x * sigmoid;
    }
}

static void generate_spline_lut_f16(uint16_t *lut_f16, const float *lut_f32, int size) {
    for (int i = 0; i < size; i++) {
        lut_f16[i] = f32_to_f16(lut_f32[i]);
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  Vulkan Infrastructure
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    VkInstance inst;
    VkPhysicalDevice pdev;
    VkDevice dev;
    uint32_t qf;
    VkQueue queue;
    VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties mprops;
} Vk;

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    void *map;
    VkDeviceSize sz;
} Buf;

static int load_vulkan(void) {
    vk_lib = dlopen("libvulkan.so", RTLD_NOW);
    if (!vk_lib) vk_lib = dlopen("libvulkan.so.1", RTLD_NOW);
    if (!vk_lib) {
        fprintf(stderr, "Could not load libvulkan.so\n");
        return -1;
    }
    
    LOAD_VK(vkCreateInstance);
    LOAD_VK(vkEnumeratePhysicalDevices);
    LOAD_VK(vkGetPhysicalDeviceMemoryProperties);
    LOAD_VK(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_VK(vkCreateDevice);
    LOAD_VK(vkGetDeviceQueue);
    LOAD_VK(vkCreateBuffer);
    LOAD_VK(vkGetBufferMemoryRequirements);
    LOAD_VK(vkAllocateMemory);
    LOAD_VK(vkBindBufferMemory);
    LOAD_VK(vkMapMemory);
    LOAD_VK(vkUnmapMemory);
    LOAD_VK(vkCreateShaderModule);
    LOAD_VK(vkDestroyShaderModule);
    LOAD_VK(vkCreateDescriptorSetLayout);
    LOAD_VK(vkCreatePipelineLayout);
    LOAD_VK(vkCreateComputePipelines);
    LOAD_VK(vkCreateDescriptorPool);
    LOAD_VK(vkAllocateDescriptorSets);
    LOAD_VK(vkUpdateDescriptorSets);
    LOAD_VK(vkCreateCommandPool);
    LOAD_VK(vkAllocateCommandBuffers);
    LOAD_VK(vkBeginCommandBuffer);
    LOAD_VK(vkCmdBindPipeline);
    LOAD_VK(vkCmdBindDescriptorSets);
    LOAD_VK(vkCmdPushConstants);
    LOAD_VK(vkCmdDispatch);
    LOAD_VK(vkEndCommandBuffer);
    LOAD_VK(vkCreateFence);
    LOAD_VK(vkQueueSubmit);
    LOAD_VK(vkWaitForFences);
    LOAD_VK(vkResetFences);
    LOAD_VK(vkFreeCommandBuffers);
    LOAD_VK(vkDestroyFence);
    
    return 0;
}

static uint32_t find_mem(Vk *v, uint32_t bits, VkFlags flags) {
    for (uint32_t i = 0; i < v->mprops.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (v->mprops.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    fprintf(stderr, "No suitable memory type\n");
    return 0;
}

static int make_buf(Vk *v, Buf *b, VkDeviceSize sz, VkFlags usage) {
    b->sz = sz;
    VkBufferCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sz,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    CHECK_VK(vkCreateBuffer(v->dev, &ci, NULL, &b->buf));
    
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(v->dev, b->buf, &req);
    
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = find_mem(v, req.memoryTypeBits, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    };
    CHECK_VK(vkAllocateMemory(v->dev, &ai, NULL, &b->mem));
    CHECK_VK(vkBindBufferMemory(v->dev, b->buf, b->mem, 0));
    CHECK_VK(vkMapMemory(v->dev, b->mem, 0, sz, 0, &b->map));
    
    return 0;
}

static int init_vk(Vk *v) {
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_0
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app
    };
    CHECK_VK(vkCreateInstance(&ici, NULL, &v->inst));
    
    uint32_t n = 1;
    CHECK_VK(vkEnumeratePhysicalDevices(v->inst, &n, &v->pdev));
    vkGetPhysicalDeviceMemoryProperties(v->pdev, &v->mprops);
    
    uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(v->pdev, &qfc, NULL);
    VkQueueFamilyProperties *qfp = malloc(qfc * sizeof(*qfp));
    vkGetPhysicalDeviceQueueFamilyProperties(v->pdev, &qfc, qfp);
    v->qf = UINT32_MAX;
    for (uint32_t i = 0; i < qfc; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            v->qf = i;
            break;
        }
    }
    free(qfp);
    
    if (v->qf == UINT32_MAX) {
        fprintf(stderr, "No compute queue found\n");
        return -1;
    }
    
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = v->qf,
        .queueCount = 1,
        .pQueuePriorities = &prio
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci
    };
    CHECK_VK(vkCreateDevice(v->pdev, &dci, NULL, &v->dev));
    vkGetDeviceQueue(v->dev, v->qf, 0, &v->queue);
    
    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = v->qf
    };
    CHECK_VK(vkCreateCommandPool(v->dev, &pci, NULL, &v->pool));
    
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 *  GPU Ternary GEMM Pipeline
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    VkPipeline pipe;
    VkPipelineLayout layout;
    VkDescriptorSetLayout dsl;
    VkDescriptorPool dp;
    VkDescriptorSet ds;
    Buf weights_buf;    /* Packed ternary weights */
    Buf input_buf;      /* FP16 input activations */
    Buf scales_buf;     /* FP16 per-block scales */
    Buf output_buf;     /* FP16 output */
    Buf spline_buf;     /* FP16 spline LUT */
    VkFence fence;
} TernaryPipeline;

static int init_ternary_pipeline(Vk *v, TernaryPipeline *tp, uint32_t N, uint32_t K) {
    /* Load shader */
    FILE *f = fopen("gpu_ternary_gemm.spv", "rb");
    if (!f) {
        fprintf(stderr, "Cannot open gpu_ternary_gemm.spv\n");
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sz);
    fread(spv, 1, sz, f);
    fclose(f);
    
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sz,
        .pCode = spv
    };
    CHECK_VK(vkCreateShaderModule(v->dev, &smci, NULL, &sm));
    free(spv);
    
    /* 5 bindings: weights, input, scales, output, spline_lut */
    VkDescriptorSetLayoutBinding binds[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 5,
        .pBindings = binds
    };
    CHECK_VK(vkCreateDescriptorSetLayout(v->dev, &dslci, NULL, &tp->dsl));
    
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 8  /* N, K as uint32 */
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &tp->dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr
    };
    CHECK_VK(vkCreatePipelineLayout(v->dev, &plci, NULL, &tp->layout));
    
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm,
            .pName = "main"
        },
        .layout = tp->layout
    };
    CHECK_VK(vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &tp->pipe));
    vkDestroyShaderModule(v->dev, sm, NULL);
    
    /* Allocate buffers */
    VkDeviceSize weights_sz = (VkDeviceSize)N * K / 4;         /* 2-bit packed: K/4 bytes per row */
    VkDeviceSize input_sz = K * 2;                             /* FP16: K elements */
    VkDeviceSize scales_sz = (VkDeviceSize)N * (K / 32) * 2;   /* FP16: K/32 per row */
    VkDeviceSize output_sz = (VkDeviceSize)(N + 1) / 2 * 4;    /* FP16 packed pairs */
    VkDeviceSize spline_sz = 256 * 2;                          /* FP16: 256 entries */
    
    if (make_buf(v, &tp->weights_buf, weights_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) < 0) return -1;
    if (make_buf(v, &tp->input_buf, input_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) < 0) return -1;
    if (make_buf(v, &tp->scales_buf, scales_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) < 0) return -1;
    if (make_buf(v, &tp->output_buf, output_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) < 0) return -1;
    if (make_buf(v, &tp->spline_buf, spline_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) < 0) return -1;
    
    /* Descriptor pool and set */
    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &dps
    };
    CHECK_VK(vkCreateDescriptorPool(v->dev, &dpci, NULL, &tp->dp));
    
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = tp->dp,
        .descriptorSetCount = 1,
        .pSetLayouts = &tp->dsl
    };
    CHECK_VK(vkAllocateDescriptorSets(v->dev, &dsai, &tp->ds));
    
    VkDescriptorBufferInfo dbis[] = {
        {tp->weights_buf.buf, 0, tp->weights_buf.sz},
        {tp->input_buf.buf, 0, tp->input_buf.sz},
        {tp->scales_buf.buf, 0, tp->scales_buf.sz},
        {tp->output_buf.buf, 0, tp->output_buf.sz},
        {tp->spline_buf.buf, 0, tp->spline_buf.sz},
    };
    VkWriteDescriptorSet wds[5];
    for (int i = 0; i < 5; i++) {
        wds[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = tp->ds,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbis[i]
        };
    }
    vkUpdateDescriptorSets(v->dev, 5, wds, 0, NULL);
    
    /* Fence for synchronization */
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK_VK(vkCreateFence(v->dev, &fci, NULL, &tp->fence));
    
    return 0;
}

static double ternary_dispatch(Vk *v, TernaryPipeline *tp, uint32_t N, uint32_t K) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = v->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(v->dev, &cbai, &cmd);
    
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tp->pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tp->layout, 0, 1, &tp->ds, 0, NULL);
    uint32_t pc[2] = {N, K};
    vkCmdPushConstants(cmd, tp->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    vkCmdDispatch(cmd, N, 1, 1);  /* One workgroup per row */
    vkEndCommandBuffer(cmd);
    
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd
    };
    
    double t0 = now_us();
    vkQueueSubmit(v->queue, 1, &si, tp->fence);
    vkWaitForFences(v->dev, 1, &tp->fence, 1, UINT64_MAX);
    double elapsed = now_us() - t0;
    
    vkResetFences(v->dev, 1, &tp->fence);
    vkFreeCommandBuffers(v->dev, v->pool, 1, &cmd);
    
    return elapsed;
}

/* ══════════════════════════════════════════════════════════════════
 *  Main Benchmark
 * ══════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {
    uint32_t N = 4096;  /* FFN output dim */
    uint32_t K = 1024;  /* FFN input dim */
    int iters = 50;
    int warmup = 10;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);

    /* Ensure K is multiple of 64 for NEON kernel */
    K = (K / 64) * 64;
    if (K == 0) K = 64;

    printf("\n");
    printf("================================================================\n");
    printf("  GPU TERNARY GEMM + SPLINE ACTIVATION PROBE\n");
    printf("  N=%u (output), K=%u (input), iters=%d\n", N, K, iters);
    printf("================================================================\n\n");

    /* Allocate test data */
    size_t weights_size = (size_t)N * (K / 4);
    size_t scales_size = (size_t)N * (K / 32) * sizeof(float);
    size_t act_i8_size = K;
    size_t act_f16_size = K * sizeof(uint16_t);
    size_t out_size = N * sizeof(float);
    size_t spline_f32_size = 256 * sizeof(float);
    size_t spline_f16_size = 256 * sizeof(uint16_t);

    /* Use malloc - Android's aligned_alloc requires size % alignment == 0 */
    uint8_t *weights = malloc(weights_size);
    float *scales = malloc(scales_size);
    int8_t *act_i8 = malloc(act_i8_size);
    uint16_t *act_f16 = malloc(act_f16_size);
    float *output_cpu = malloc(out_size);
    float *spline_f32 = malloc(spline_f32_size);
    uint16_t *spline_f16 = malloc(spline_f16_size);

    printf("  Generating test data...\n"); fflush(stdout);
    generate_ternary_weights(weights, scales, N, K);
    printf("    weights done\n"); fflush(stdout);
    generate_activations_i8(act_i8, K);
    printf("    act_i8 done\n"); fflush(stdout);
    generate_activations_f16(act_f16, act_i8, K);
    printf("    act_f16 done\n"); fflush(stdout);
    generate_spline_lut_f32(spline_f32, 256);
    printf("    spline_f32 done\n"); fflush(stdout);
    generate_spline_lut_f16(spline_f16, spline_f32, 256);
    printf("    spline_f16 done\n"); fflush(stdout);

    printf("  Weights: %.2f KB (ternary packed)\n", weights_size / 1024.0);
    printf("  Scales: %.2f KB\n", scales_size / 1024.0);
    printf("  Spline LUT: %zu bytes\n", spline_f16_size);
    printf("\n"); fflush(stdout);

#ifdef __linux__
    pin_cpu(6);  /* Big core on MT6855 */
    printf("  Pinned to CPU 6 (big core)\n\n");
#endif

    /* ═══════════════════════════════════════════════════════════════
     *  CPU Baseline
     * ═══════════════════════════════════════════════════════════════ */

#ifdef __ARM_NEON
    printf("  CPU Baseline (NEON ternary + spline):\n"); fflush(stdout);

    /* Warmup */
    printf("    warmup...\n"); fflush(stdout);
    for (int i = 0; i < warmup; i++) {
        cpu_ternary_matvec_spline(output_cpu, weights, act_i8, scales, spline_f32, N, K);
    }
    printf("    warmup done\n"); fflush(stdout);

    volatile float sink = 0;
    double *cpu_times = malloc(iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = now_us();
        cpu_ternary_matvec_spline(output_cpu, weights, act_i8, scales, spline_f32, N, K);
        sink += output_cpu[0];  /* Prevent optimization */
        cpu_times[i] = now_us() - t0;
    }
    printf("    sample times: %.1f, %.1f, %.1f (sink=%.2f)\n", 
           cpu_times[0], cpu_times[1], cpu_times[2], (double)sink);
    fflush(stdout);

    double cpu_sum = 0, cpu_min = 1e18, cpu_max = 0;
    for (int i = 0; i < iters; i++) {
        cpu_sum += cpu_times[i];
        if (cpu_times[i] < cpu_min) cpu_min = cpu_times[i];
        if (cpu_times[i] > cpu_max) cpu_max = cpu_times[i];
    }
    double cpu_mean = cpu_sum / iters;
    free(cpu_times);

    double cpu_ops = 2.0 * N * K;
    double cpu_gops = cpu_ops / (cpu_mean * 1e3);
    double cpu_bytes = weights_size + act_i8_size + scales_size + out_size;
    double cpu_gbps = cpu_bytes / (cpu_mean * 1e3);

    printf("    Mean: %.1f us  (min=%.1f, max=%.1f)\n", cpu_mean, cpu_min, cpu_max);
    printf("    Throughput: %.2f GOP/s\n", cpu_gops);
    printf("    Bandwidth: %.2f GB/s (effective)\n", cpu_gbps);
    printf("\n");
#else
    printf("  CPU Baseline: NEON not available\n\n");
    double cpu_mean = 0;
#endif

    /* ═══════════════════════════════════════════════════════════════
     *  GPU Path
     * ═══════════════════════════════════════════════════════════════ */

    printf("  GPU Path (Vulkan ternary GEMM + spline):\n");

    if (load_vulkan() < 0) {
        printf("    ERROR: Could not load Vulkan\n");
        goto cleanup;
    }
    printf("    Vulkan library loaded\n");

    Vk v;
    if (init_vk(&v) < 0) {
        printf("    ERROR: Could not initialize Vulkan\n");
        goto cleanup;
    }
    printf("    Vulkan device ready\n");

    TernaryPipeline tp;
    if (init_ternary_pipeline(&v, &tp, N, K) < 0) {
        printf("    ERROR: Could not create pipeline\n");
        goto cleanup;
    }
    printf("    Pipeline created\n");

    /* Upload data to GPU buffers */
    memcpy(tp.weights_buf.map, weights, weights_size);
    memcpy(tp.input_buf.map, act_f16, act_f16_size);
    
    /* Convert scales to FP16 for GPU */
    uint16_t *scales_f16_gpu = (uint16_t*)tp.scales_buf.map;
    for (size_t i = 0; i < (size_t)N * (K / 32); i++) {
        scales_f16_gpu[i] = f32_to_f16(scales[i]);
    }
    
    memcpy(tp.spline_buf.map, spline_f16, spline_f16_size);
    printf("    Data uploaded\n\n");

    /* Warmup */
    printf("    Warming up (%d dispatches)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        ternary_dispatch(&v, &tp, N, K);
    }

    /* Benchmark */
    printf("    Benchmarking (%d dispatches)...\n", iters);
    double *gpu_times = malloc(iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        gpu_times[i] = ternary_dispatch(&v, &tp, N, K);
    }

    double gpu_sum = 0, gpu_min = 1e18, gpu_max = 0;
    for (int i = 0; i < iters; i++) {
        gpu_sum += gpu_times[i];
        if (gpu_times[i] < gpu_min) gpu_min = gpu_times[i];
        if (gpu_times[i] > gpu_max) gpu_max = gpu_times[i];
    }
    double gpu_mean = gpu_sum / iters;
    free(gpu_times);

    double gpu_ops = 2.0 * N * K;
    double gpu_gops = gpu_ops / (gpu_mean * 1e3);
    double gpu_bytes = weights_size + K * 2 + (size_t)N * (K / 32) * 2 + N * 2 + 256 * 2;
    double gpu_gbps = gpu_bytes / (gpu_mean * 1e3);

    printf("\n");
    printf("    Mean: %.1f us  (min=%.1f, max=%.1f)\n", gpu_mean, gpu_min, gpu_max);
    printf("    Throughput: %.2f GOP/s\n", gpu_gops);
    printf("    Bandwidth: %.2f GB/s (effective)\n", gpu_gbps);
    printf("\n");

    /* ═══════════════════════════════════════════════════════════════
     *  Comparison
     * ═══════════════════════════════════════════════════════════════ */

    printf("================================================================\n");
    printf("  COMPARISON:\n");
#ifdef __ARM_NEON
    printf("    CPU: %.1f us\n", cpu_mean);
#endif
    printf("    GPU: %.1f us\n", gpu_mean);
#ifdef __ARM_NEON
    if (gpu_mean < cpu_mean) {
        printf("    GPU is %.2fx FASTER\n", cpu_mean / gpu_mean);
    } else {
        printf("    CPU is %.2fx faster\n", gpu_mean / cpu_mean);
    }
#endif
    printf("================================================================\n\n");

    /* Analysis for batching */
    printf("  What if we batch multiple layers?\n");
    printf("  (Dispatch overhead amortized across layers)\n\n");
    
    double dispatch_overhead = 50.0;  /* us, measured previously */
    for (int layers = 1; layers <= 32; layers *= 2) {
        double batched_gpu = gpu_mean * layers;
        double amortized = batched_gpu / layers;
#ifdef __ARM_NEON
        double batched_cpu = cpu_mean * layers;
        printf("    %2d layers: GPU=%.1f us (%.1f/layer), CPU=%.1f us → %.2fx %s\n",
               layers, batched_gpu, amortized, batched_cpu,
               batched_cpu / batched_gpu,
               batched_gpu < batched_cpu ? "GPU" : "CPU");
#else
        printf("    %2d layers: GPU=%.1f us (%.1f us/layer)\n",
               layers, batched_gpu, amortized);
#endif
    }
    printf("\n");

cleanup:
    free(weights);
    free(scales);
    free(act_i8);
    free(act_f16);
    free(output_cpu);
    free(spline_f32);
    free(spline_f16);
    if (vk_lib) dlclose(vk_lib);

    return 0;
}
