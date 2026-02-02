/*
 * vk_tmu_probe — Benchmark sigmoid activation strategies on PowerVR BXM-8-256
 *
 * Measures three GPU sigmoid implementations + dispatch overhead:
 *   0. NOP dispatch (latency measurement)
 *   A. ALU:       1.0 / (1.0 + exp(-x))  — uses SFU
 *   B. Buffer LUT: 256-entry SSBO + manual lerp in ALU
 *   C. TMU LUT:   256-entry R16_SFLOAT 1D texture + VK_FILTER_LINEAR
 *
 * Also measures:
 *   - Dispatch latency (vkQueueSubmit → fence signal for NOP shader)
 *   - HOST_COHERENT write-to-read latency
 *   - Accuracy (max error vs libm sigmoid)
 *
 * Build (Android ARM64):
 *   NDK=~/Library/Android/sdk/ndk/28.2.13676358
 *   $NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang \
 *       -O2 -I../shaders -o vk_tmu_probe tests/vk_tmu_probe.c -lvulkan -lm
 *   adb push vk_tmu_probe /data/local/tmp/
 *   adb shell /data/local/tmp/vk_tmu_probe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vulkan/vulkan.h>

/* Embedded SPIR-V bytecode */
#include "dispatch_nop_spv.h"
#include "sigmoid_alu_spv.h"
#include "sigmoid_lut_buf_spv.h"
#include "sigmoid_tmu_spv.h"

/* ── Configuration ─────────────────────────────────────────────── */

#define N_ELEMENTS   1024    /* D_MODEL — the activation vector size */
#define N_WARMUP     50      /* warmup dispatches (fill caches, JIT) */
#define N_ITERS      500     /* timed dispatches per strategy */
#define LUT_SIZE     256     /* entries in sigmoid table */
#define X_MIN       (-8.0f)
#define X_MAX       ( 8.0f)

/* ── Helpers ───────────────────────────────────────────────────── */

#define CHECK_VK(call) do { \
    VkResult _r = (call); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "FATAL Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static float sigmoid_ref(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* FP32 → FP16 (IEEE 754) */
static uint16_t f32_to_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = ((u >> 23) & 0xFF) - 127;
    uint32_t frac = u & 0x7FFFFF;

    if (exp > 15) return sign | 0x7C00;           /* inf */
    if (exp < -14) {                               /* denorm or zero */
        frac = (frac | 0x800000) >> (1 - 14 - exp);
        return sign | (frac >> 13);
    }
    return sign | ((exp + 15) << 10) | (frac >> 13);
}

/* ── Vulkan context ────────────────────────────────────────────── */

typedef struct {
    VkInstance       instance;
    VkPhysicalDevice pdev;
    VkDevice         device;
    uint32_t         queue_family;
    VkQueue          queue;
    VkCommandPool    cmd_pool;
    VkCommandBuffer  cmd;
    VkFence          fence;
    VkDescriptorPool desc_pool;

    /* Memory properties for allocation */
    VkPhysicalDeviceMemoryProperties mem_props;
} VkCtx;

static uint32_t find_memory_type(VkCtx *ctx, uint32_t type_bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < ctx->mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (ctx->mem_props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    fprintf(stderr, "FATAL: no suitable memory type\n");
    exit(1);
}

static void init_vulkan(VkCtx *ctx) {
    /* Instance */
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vk_tmu_probe",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo inst_ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    CHECK_VK(vkCreateInstance(&inst_ci, NULL, &ctx->instance));

    /* Physical device */
    uint32_t n = 1;
    CHECK_VK(vkEnumeratePhysicalDevices(ctx->instance, &n, &ctx->pdev));

    vkGetPhysicalDeviceMemoryProperties(ctx->pdev, &ctx->mem_props);

    /* Find compute queue */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->pdev, &qf_count, NULL);
    VkQueueFamilyProperties *qf = malloc(qf_count * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->pdev, &qf_count, qf);
    ctx->queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx->queue_family = i;
            break;
        }
    }
    free(qf);
    if (ctx->queue_family == UINT32_MAX) {
        fprintf(stderr, "FATAL: no compute queue\n"); exit(1);
    }

    /* Logical device — enable FP16/INT8, 16-bit storage, 8-bit storage */
    float priority = 1.0f;
    VkDeviceQueueCreateInfo q_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    /* Chain feature structs */
    VkPhysicalDevice8BitStorageFeatures f8 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
        .storageBuffer8BitAccess = VK_TRUE,
    };
    VkPhysicalDevice16BitStorageFeatures f16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .pNext = &f8,
        .storageBuffer16BitAccess = VK_TRUE,
    };
    VkPhysicalDeviceShaderFloat16Int8Features fp16i8 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
        .pNext = &f16,
        .shaderFloat16 = VK_TRUE,
        .shaderInt8 = VK_TRUE,
    };

    const char *exts[] = {
        "VK_KHR_shader_float16_int8",
        "VK_KHR_16bit_storage",
        "VK_KHR_8bit_storage",
    };

    VkDeviceCreateInfo dev_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &fp16i8,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &q_ci,
        .enabledExtensionCount = 3,
        .ppEnabledExtensionNames = exts,
    };
    CHECK_VK(vkCreateDevice(ctx->pdev, &dev_ci, NULL, &ctx->device));
    vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->queue);

    /* Command pool + buffer */
    VkCommandPoolCreateInfo pool_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->queue_family,
    };
    CHECK_VK(vkCreateCommandPool(ctx->device, &pool_ci, NULL, &ctx->cmd_pool));

    VkCommandBufferAllocateInfo alloc_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    CHECK_VK(vkAllocateCommandBuffers(ctx->device, &alloc_ci, &ctx->cmd));

    /* Fence */
    VkFenceCreateInfo fence_ci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    CHECK_VK(vkCreateFence(ctx->device, &fence_ci, NULL, &ctx->fence));

    /* Descriptor pool — enough for all pipelines */
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
    };
    VkDescriptorPoolCreateInfo dp_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 8,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };
    CHECK_VK(vkCreateDescriptorPool(ctx->device, &dp_ci, NULL, &ctx->desc_pool));
}

static void destroy_vulkan(VkCtx *ctx) {
    vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
    vkDestroyFence(ctx->device, ctx->fence, NULL);
    vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
    vkDestroyDevice(ctx->device, NULL);
    vkDestroyInstance(ctx->instance, NULL);
}

/* ── Buffer helpers ────────────────────────────────────────────── */

typedef struct {
    VkBuffer     buf;
    VkDeviceMemory mem;
    void        *mapped;
    VkDeviceSize size;
} GpuBuf;

static GpuBuf create_buffer(VkCtx *ctx, VkDeviceSize size, VkBufferUsageFlags usage) {
    GpuBuf b = { .size = size };

    VkBufferCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    CHECK_VK(vkCreateBuffer(ctx->device, &ci, NULL, &b.buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx->device, b.buf, &req);

    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = find_memory_type(ctx, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    CHECK_VK(vkAllocateMemory(ctx->device, &ai, NULL, &b.mem));
    CHECK_VK(vkBindBufferMemory(ctx->device, b.buf, b.mem, 0));
    CHECK_VK(vkMapMemory(ctx->device, b.mem, 0, size, 0, &b.mapped));
    return b;
}

static void destroy_buffer(VkCtx *ctx, GpuBuf *b) {
    vkUnmapMemory(ctx->device, b->mem);
    vkDestroyBuffer(ctx->device, b->buf, NULL);
    vkFreeMemory(ctx->device, b->mem, NULL);
}

/* ── Pipeline helpers ──────────────────────────────────────────── */

static VkShaderModule create_shader(VkCtx *ctx, const uint32_t *code, size_t size) {
    VkShaderModuleCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = code,
    };
    VkShaderModule sm;
    CHECK_VK(vkCreateShaderModule(ctx->device, &ci, NULL, &sm));
    return sm;
}

/* ── Dispatch + timing ─────────────────────────────────────────── */

static double dispatch_and_time(VkCtx *ctx, VkPipeline pipeline,
                                 VkPipelineLayout layout,
                                 VkDescriptorSet desc_set,
                                 uint32_t group_count_x,
                                 const void *push_data, uint32_t push_size) {
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    CHECK_VK(vkResetCommandBuffer(ctx->cmd, 0));
    CHECK_VK(vkBeginCommandBuffer(ctx->cmd, &begin));

    vkCmdBindPipeline(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout, 0, 1, &desc_set, 0, NULL);
    if (push_data && push_size > 0) {
        vkCmdPushConstants(ctx->cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, push_size, push_data);
    }
    vkCmdDispatch(ctx->cmd, group_count_x, 1, 1);

    CHECK_VK(vkEndCommandBuffer(ctx->cmd));

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd,
    };

    CHECK_VK(vkResetFences(ctx->device, 1, &ctx->fence));
    double t0 = now_us();
    CHECK_VK(vkQueueSubmit(ctx->queue, 1, &submit, ctx->fence));
    CHECK_VK(vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX));
    double t1 = now_us();

    return t1 - t0;
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 0: Dispatch Latency (NOP shader)
 * ══════════════════════════════════════════════════════════════════ */

static void bench_dispatch_latency(VkCtx *ctx) {
    printf("\n=== Test 0: Dispatch Latency (NOP shader) ===\n");

    /* Buffer (just needs 1 float for the NOP to write) */
    GpuBuf out = create_buffer(ctx, sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    /* Descriptor set layout: 1 storage buffer */
    VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo dsl_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VkDescriptorSetLayout dsl;
    CHECK_VK(vkCreateDescriptorSetLayout(ctx->device, &dsl_ci, NULL, &dsl));

    /* Pipeline layout (no push constants) */
    VkPipelineLayoutCreateInfo pl_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
    };
    VkPipelineLayout pl;
    CHECK_VK(vkCreatePipelineLayout(ctx->device, &pl_ci, NULL, &pl));

    /* Shader + pipeline */
    VkShaderModule sm = create_shader(ctx,
        (const uint32_t *)dispatch_nop_spv, dispatch_nop_spv_len);

    VkComputePipelineCreateInfo pipe_ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm,
            .pName = "main",
        },
        .layout = pl,
    };
    VkPipeline pipeline;
    CHECK_VK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1,
                                       &pipe_ci, NULL, &pipeline));

    /* Descriptor set */
    VkDescriptorSetAllocateInfo ds_ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &dsl,
    };
    VkDescriptorSet ds;
    CHECK_VK(vkAllocateDescriptorSets(ctx->device, &ds_ai, &ds));

    VkDescriptorBufferInfo buf_info = { .buffer = out.buf, .offset = 0, .range = sizeof(float) };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_info,
    };
    vkUpdateDescriptorSets(ctx->device, 1, &write, 0, NULL);

    /* Warmup */
    for (int i = 0; i < N_WARMUP; i++) {
        dispatch_and_time(ctx, pipeline, pl, ds, 1, NULL, 0);
    }

    /* Timed */
    double total = 0, min_t = 1e9, max_t = 0;
    for (int i = 0; i < N_ITERS; i++) {
        double t = dispatch_and_time(ctx, pipeline, pl, ds, 1, NULL, 0);
        total += t;
        if (t < min_t) min_t = t;
        if (t > max_t) max_t = t;
    }

    printf("  Iterations: %d\n", N_ITERS);
    printf("  Mean:  %.1f us\n", total / N_ITERS);
    printf("  Min:   %.1f us\n", min_t);
    printf("  Max:   %.1f us\n", max_t);
    printf("  Result: %.1f (should be 1.0)\n", ((float *)out.mapped)[0]);

    /* Cleanup */
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyShaderModule(ctx->device, sm, NULL);
    vkDestroyPipelineLayout(ctx->device, pl, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, dsl, NULL);
    destroy_buffer(ctx, &out);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test A: ALU Sigmoid
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t count;
} PushALU;

static void bench_sigmoid_alu(VkCtx *ctx, GpuBuf *input, GpuBuf *output) {
    printf("\n=== Test A: ALU Sigmoid (SFU exp) ===\n");

    /* Descriptor set layout: 2 storage buffers */
    VkDescriptorSetLayoutBinding bindings[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };
    VkDescriptorSetLayoutCreateInfo dsl_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = bindings,
    };
    VkDescriptorSetLayout dsl;
    CHECK_VK(vkCreateDescriptorSetLayout(ctx->device, &dsl_ci, NULL, &dsl));

    VkPushConstantRange push = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0, .size = sizeof(PushALU),
    };
    VkPipelineLayoutCreateInfo pl_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push,
    };
    VkPipelineLayout pl;
    CHECK_VK(vkCreatePipelineLayout(ctx->device, &pl_ci, NULL, &pl));

    VkShaderModule sm = create_shader(ctx,
        (const uint32_t *)sigmoid_alu_spv, sigmoid_alu_spv_len);

    VkComputePipelineCreateInfo pipe_ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm, .pName = "main",
        },
        .layout = pl,
    };
    VkPipeline pipeline;
    CHECK_VK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1,
                                       &pipe_ci, NULL, &pipeline));

    /* Descriptor set */
    VkDescriptorSetAllocateInfo ds_ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = 1, .pSetLayouts = &dsl,
    };
    VkDescriptorSet ds;
    CHECK_VK(vkAllocateDescriptorSets(ctx->device, &ds_ai, &ds));

    VkDescriptorBufferInfo buf_infos[] = {
        { .buffer = input->buf,  .offset = 0, .range = input->size },
        { .buffer = output->buf, .offset = 0, .range = output->size },
    };
    VkWriteDescriptorSet writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 1, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[1] },
    };
    vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);

    PushALU pc = { .count = N_ELEMENTS };
    uint32_t groups = (N_ELEMENTS + 127) / 128;

    /* Warmup */
    for (int i = 0; i < N_WARMUP; i++) {
        dispatch_and_time(ctx, pipeline, pl, ds, groups, &pc, sizeof(pc));
    }

    /* Timed */
    double total = 0, min_t = 1e9, max_t = 0;
    for (int i = 0; i < N_ITERS; i++) {
        double t = dispatch_and_time(ctx, pipeline, pl, ds, groups, &pc, sizeof(pc));
        total += t;
        if (t < min_t) min_t = t;
        if (t > max_t) max_t = t;
    }

    /* Accuracy */
    float *in  = (float *)input->mapped;
    float *out = (float *)output->mapped;
    float max_err = 0;
    for (int i = 0; i < N_ELEMENTS; i++) {
        float ref = sigmoid_ref(in[i]);
        float err = fabsf(out[i] - ref);
        if (err > max_err) max_err = err;
    }

    printf("  Iterations: %d\n", N_ITERS);
    printf("  Mean:  %.1f us  (%.1f us/element)\n",
           total / N_ITERS, total / N_ITERS / N_ELEMENTS * 1000);
    printf("  Min:   %.1f us\n", min_t);
    printf("  Max:   %.1f us\n", max_t);
    printf("  Max error: %.2e\n", max_err);
    printf("  Sample: sigmoid(%.3f) = %.6f (ref %.6f)\n",
           in[N_ELEMENTS/2], out[N_ELEMENTS/2], sigmoid_ref(in[N_ELEMENTS/2]));

    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyShaderModule(ctx->device, sm, NULL);
    vkDestroyPipelineLayout(ctx->device, pl, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, dsl, NULL);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test B: Buffer LUT + Manual Lerp
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t count;
    float x_min;
    float x_max;
    float inv_step;
} PushLUT;

static void bench_sigmoid_lut_buf(VkCtx *ctx, GpuBuf *input, GpuBuf *output) {
    printf("\n=== Test B: Buffer LUT + Manual Lerp ===\n");

    /* Create LUT buffer */
    GpuBuf lut = create_buffer(ctx, LUT_SIZE * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    float *lut_data = (float *)lut.mapped;
    for (int i = 0; i < LUT_SIZE; i++) {
        float x = X_MIN + (X_MAX - X_MIN) * i / (float)(LUT_SIZE - 1);
        lut_data[i] = sigmoid_ref(x);
    }

    /* Descriptor set layout: 3 storage buffers */
    VkDescriptorSetLayoutBinding bindings[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };
    VkDescriptorSetLayoutCreateInfo dsl_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bindings,
    };
    VkDescriptorSetLayout dsl;
    CHECK_VK(vkCreateDescriptorSetLayout(ctx->device, &dsl_ci, NULL, &dsl));

    VkPushConstantRange push = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0, .size = sizeof(PushLUT),
    };
    VkPipelineLayoutCreateInfo pl_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push,
    };
    VkPipelineLayout pl;
    CHECK_VK(vkCreatePipelineLayout(ctx->device, &pl_ci, NULL, &pl));

    VkShaderModule sm = create_shader(ctx,
        (const uint32_t *)sigmoid_lut_buf_spv, sigmoid_lut_buf_spv_len);

    VkComputePipelineCreateInfo pipe_ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm, .pName = "main",
        },
        .layout = pl,
    };
    VkPipeline pipeline;
    CHECK_VK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1,
                                       &pipe_ci, NULL, &pipeline));

    /* Descriptor set */
    VkDescriptorSetAllocateInfo ds_ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = 1, .pSetLayouts = &dsl,
    };
    VkDescriptorSet ds;
    CHECK_VK(vkAllocateDescriptorSets(ctx->device, &ds_ai, &ds));

    VkDescriptorBufferInfo buf_infos[] = {
        { .buffer = input->buf,  .offset = 0, .range = input->size },
        { .buffer = output->buf, .offset = 0, .range = output->size },
        { .buffer = lut.buf,     .offset = 0, .range = lut.size },
    };
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++) {
        w[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buf_infos[i],
        };
    }
    vkUpdateDescriptorSets(ctx->device, 3, w, 0, NULL);

    PushLUT pc = {
        .count = N_ELEMENTS,
        .x_min = X_MIN,
        .x_max = X_MAX,
        .inv_step = (float)(LUT_SIZE - 1) / (X_MAX - X_MIN),
    };
    uint32_t groups = (N_ELEMENTS + 127) / 128;

    /* Warmup */
    for (int i = 0; i < N_WARMUP; i++) {
        dispatch_and_time(ctx, pipeline, pl, ds, groups, &pc, sizeof(pc));
    }

    /* Timed */
    double total = 0, min_t = 1e9, max_t = 0;
    for (int i = 0; i < N_ITERS; i++) {
        double t = dispatch_and_time(ctx, pipeline, pl, ds, groups, &pc, sizeof(pc));
        total += t;
        if (t < min_t) min_t = t;
        if (t > max_t) max_t = t;
    }

    /* Accuracy */
    float *in  = (float *)input->mapped;
    float *out = (float *)output->mapped;
    float max_err = 0;
    for (int i = 0; i < N_ELEMENTS; i++) {
        float ref = sigmoid_ref(in[i]);
        float err = fabsf(out[i] - ref);
        if (err > max_err) max_err = err;
    }

    printf("  Iterations: %d\n", N_ITERS);
    printf("  Mean:  %.1f us  (%.1f us/element)\n",
           total / N_ITERS, total / N_ITERS / N_ELEMENTS * 1000);
    printf("  Min:   %.1f us\n", min_t);
    printf("  Max:   %.1f us\n", max_t);
    printf("  Max error: %.2e\n", max_err);
    printf("  Sample: sigmoid(%.3f) = %.6f (ref %.6f)\n",
           in[N_ELEMENTS/2], out[N_ELEMENTS/2], sigmoid_ref(in[N_ELEMENTS/2]));

    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyShaderModule(ctx->device, sm, NULL);
    vkDestroyPipelineLayout(ctx->device, pl, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, dsl, NULL);
    destroy_buffer(ctx, &lut);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test C: TMU Texture LUT (R16_SFLOAT + VK_FILTER_LINEAR)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t count;
    float x_min;
    float x_range;
} PushTMU;

static void bench_sigmoid_tmu(VkCtx *ctx, GpuBuf *input, GpuBuf *output) {
    printf("\n=== Test C: TMU Texture LUT (R16_SFLOAT + LINEAR) ===\n");

    /* ── Create 1D R16_SFLOAT image ── */
    VkImageCreateInfo img_ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_1D,
        .format = VK_FORMAT_R16_SFLOAT,
        .extent = { .width = LUT_SIZE, .height = 1, .depth = 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage image;
    CHECK_VK(vkCreateImage(ctx->device, &img_ci, NULL, &image));

    VkMemoryRequirements img_req;
    vkGetImageMemoryRequirements(ctx->device, image, &img_req);
    VkMemoryAllocateInfo img_ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = img_req.size,
        .memoryTypeIndex = find_memory_type(ctx, img_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    VkDeviceMemory img_mem;
    CHECK_VK(vkAllocateMemory(ctx->device, &img_ai, NULL, &img_mem));
    CHECK_VK(vkBindImageMemory(ctx->device, image, img_mem, 0));

    /* Image view */
    VkImageViewCreateInfo iv_ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_1D,
        .format = VK_FORMAT_R16_SFLOAT,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1, .layerCount = 1,
        },
    };
    VkImageView img_view;
    CHECK_VK(vkCreateImageView(ctx->device, &iv_ci, NULL, &img_view));

    /* Sampler with LINEAR filter */
    VkSamplerCreateInfo samp_ci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .unnormalizedCoordinates = VK_FALSE,
    };
    VkSampler sampler;
    CHECK_VK(vkCreateSampler(ctx->device, &samp_ci, NULL, &sampler));

    /* ── Upload sigmoid LUT to image via staging buffer ── */
    GpuBuf staging = create_buffer(ctx, LUT_SIZE * sizeof(uint16_t),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    uint16_t *fp16_data = (uint16_t *)staging.mapped;
    for (int i = 0; i < LUT_SIZE; i++) {
        float x = X_MIN + (X_MAX - X_MIN) * i / (float)(LUT_SIZE - 1);
        fp16_data[i] = f32_to_f16(sigmoid_ref(x));
    }

    /* Record upload commands */
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    CHECK_VK(vkResetCommandBuffer(ctx->cmd, 0));
    CHECK_VK(vkBeginCommandBuffer(ctx->cmd, &begin));

    /* Transition: UNDEFINED → TRANSFER_DST */
    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1, .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(ctx->cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);

    /* Copy buffer → image */
    VkBufferImageCopy region = {
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .layerCount = 1,
        },
        .imageExtent = { .width = LUT_SIZE, .height = 1, .depth = 1 },
    };
    vkCmdCopyBufferToImage(ctx->cmd, staging.buf, image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    /* Transition: TRANSFER_DST → SHADER_READ_ONLY */
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);

    CHECK_VK(vkEndCommandBuffer(ctx->cmd));
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &ctx->cmd,
    };
    CHECK_VK(vkResetFences(ctx->device, 1, &ctx->fence));
    CHECK_VK(vkQueueSubmit(ctx->queue, 1, &submit, ctx->fence));
    CHECK_VK(vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX));

    destroy_buffer(ctx, &staging);

    /* ── Pipeline ── */
    VkDescriptorSetLayoutBinding bindings[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };
    VkDescriptorSetLayoutCreateInfo dsl_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bindings,
    };
    VkDescriptorSetLayout dsl;
    CHECK_VK(vkCreateDescriptorSetLayout(ctx->device, &dsl_ci, NULL, &dsl));

    VkPushConstantRange push = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0, .size = sizeof(PushTMU),
    };
    VkPipelineLayoutCreateInfo pl_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push,
    };
    VkPipelineLayout pl;
    CHECK_VK(vkCreatePipelineLayout(ctx->device, &pl_ci, NULL, &pl));

    VkShaderModule sm = create_shader(ctx,
        (const uint32_t *)sigmoid_tmu_spv, sigmoid_tmu_spv_len);

    VkComputePipelineCreateInfo pipe_ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm, .pName = "main",
        },
        .layout = pl,
    };
    VkPipeline pipeline;
    CHECK_VK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1,
                                       &pipe_ci, NULL, &pipeline));

    /* Descriptor set */
    VkDescriptorSetAllocateInfo ds_ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = 1, .pSetLayouts = &dsl,
    };
    VkDescriptorSet ds;
    CHECK_VK(vkAllocateDescriptorSets(ctx->device, &ds_ai, &ds));

    VkDescriptorBufferInfo buf_infos[] = {
        { .buffer = input->buf,  .offset = 0, .range = input->size },
        { .buffer = output->buf, .offset = 0, .range = output->size },
    };
    VkDescriptorImageInfo img_info = {
        .sampler = sampler,
        .imageView = img_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkWriteDescriptorSet writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 1, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 2, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &img_info },
    };
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

    PushTMU pc = {
        .count = N_ELEMENTS,
        .x_min = X_MIN,
        .x_range = X_MAX - X_MIN,
    };
    uint32_t groups = (N_ELEMENTS + 127) / 128;

    /* Warmup */
    for (int i = 0; i < N_WARMUP; i++) {
        dispatch_and_time(ctx, pipeline, pl, ds, groups, &pc, sizeof(pc));
    }

    /* Timed */
    double total = 0, min_t = 1e9, max_t = 0;
    for (int i = 0; i < N_ITERS; i++) {
        double t = dispatch_and_time(ctx, pipeline, pl, ds, groups, &pc, sizeof(pc));
        total += t;
        if (t < min_t) min_t = t;
        if (t > max_t) max_t = t;
    }

    /* Accuracy */
    float *in  = (float *)input->mapped;
    float *out = (float *)output->mapped;
    float max_err = 0;
    for (int i = 0; i < N_ELEMENTS; i++) {
        float ref = sigmoid_ref(in[i]);
        float err = fabsf(out[i] - ref);
        if (err > max_err) max_err = err;
    }

    printf("  Iterations: %d\n", N_ITERS);
    printf("  Mean:  %.1f us  (%.1f us/element)\n",
           total / N_ITERS, total / N_ITERS / N_ELEMENTS * 1000);
    printf("  Min:   %.1f us\n", min_t);
    printf("  Max:   %.1f us\n", max_t);
    printf("  Max error: %.2e (includes FP16 quantization)\n", max_err);
    printf("  Sample: sigmoid(%.3f) = %.6f (ref %.6f)\n",
           in[N_ELEMENTS/2], out[N_ELEMENTS/2], sigmoid_ref(in[N_ELEMENTS/2]));

    /* Cleanup */
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyShaderModule(ctx->device, sm, NULL);
    vkDestroyPipelineLayout(ctx->device, pl, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, dsl, NULL);
    vkDestroySampler(ctx->device, sampler, NULL);
    vkDestroyImageView(ctx->device, img_view, NULL);
    vkDestroyImage(ctx->device, image, NULL);
    vkFreeMemory(ctx->device, img_mem, NULL);
}

/* ══════════════════════════════════════════════════════════════════
 *  Main
 * ══════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  TriX Vulkan TMU Activation Probe                      ║\n");
    printf("║  PowerVR BXM-8-256 / Motorola moto g power 5G          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\nConfig: %d elements, %d warmup, %d iterations, %d-entry LUT\n",
           N_ELEMENTS, N_WARMUP, N_ITERS, LUT_SIZE);

    VkCtx ctx;
    init_vulkan(&ctx);

    /* Print device name for confirmation */
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.pdev, &props);
    printf("Device: %s\n", props.deviceName);

    /* Shared input/output buffers */
    GpuBuf input = create_buffer(&ctx, N_ELEMENTS * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    GpuBuf output = create_buffer(&ctx, N_ELEMENTS * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    /* Fill input: uniform distribution over [-10, 10] to test clamping too */
    float *in_data = (float *)input.mapped;
    for (int i = 0; i < N_ELEMENTS; i++) {
        in_data[i] = -10.0f + 20.0f * i / (float)(N_ELEMENTS - 1);
    }

    /* Run all benchmarks */
    bench_dispatch_latency(&ctx);
    bench_sigmoid_alu(&ctx, &input, &output);
    bench_sigmoid_lut_buf(&ctx, &input, &output);
    bench_sigmoid_tmu(&ctx, &input, &output);

    /* ── Summary ── */
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  DECISION DATA                                          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\nKey question: Is dispatch overhead < activation compute time?\n");
    printf("If dispatch > 100us, GPU offload only viable for batched ops.\n");
    printf("If dispatch < 50us, CPU+GPU pipeline is feasible per-layer.\n");
    printf("\nFor LFM2 context:\n");
    printf("  - SiLU on 1024 elements = 1 dispatch per ShortConv/Attn layer\n");
    printf("  - Softmax on seq_len elements = 1 dispatch per Attn layer\n");
    printf("  - Total per token: ~32 activations across 16 layers\n");
    printf("  - If dispatch = Xus, overhead = 32*X us per token\n");
    printf("  - At 40 tok/s target = 25ms/tok budget, overhead must be < 5ms\n");
    printf("  - => dispatch must be < 156us for overhead < 20%% of budget\n");

    /* Cleanup */
    destroy_buffer(&ctx, &input);
    destroy_buffer(&ctx, &output);
    destroy_vulkan(&ctx);

    printf("\n=== Done ===\n");
    return 0;
}
