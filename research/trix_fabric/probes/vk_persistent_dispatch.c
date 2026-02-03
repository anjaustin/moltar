/*
 * vk_persistent_dispatch.c — Probe A: Persistent Dispatch Latency
 *
 * Measures the REAL floor of GPU dispatch on PowerVR BXM-8-256:
 *
 * Test 1: Cold dispatch (record + submit + wait) — the 386us baseline
 * Test 2: Pre-recorded resubmit (reset fence + submit same cmdbuf + wait)
 * Test 3: Timeline semaphore signal→completion (CPU signal, GPU starts, CPU reads)
 * Test 4: Double-buffered pre-recorded (2 cmdbufs alternating, overlap submit)
 *
 * The key question: how much of 386us is recording overhead vs driver/HW?
 *
 * Build:
 *   NDK=~/Library/Android/sdk/ndk/28.2.13676358
 *   CC=$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang
 *   $CC -O2 -o vk_persistent_dispatch vk_persistent_dispatch.c -lvulkan -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vulkan/vulkan.h>

/* Timeline semaphore functions (KHR extension on Vulkan 1.1) */
static PFN_vkSignalSemaphoreKHR pfn_vkSignalSemaphore;
static PFN_vkWaitSemaphoresKHR pfn_vkWaitSemaphores;
#define vkSignalSemaphore pfn_vkSignalSemaphore
#define vkWaitSemaphores pfn_vkWaitSemaphores

/* ── Embedded NOP SPIR-V ──
 * Compiled from: #version 450; layout(local_size_x=1) in;
 * layout(set=0,binding=0) buffer O { float d[]; } o; void main() { o.d[0]=1.0; }
 *
 * We inline the SPIR-V bytes to avoid needing the shader headers.
 */
static const uint32_t nop_spirv[] = {
    /* Magic, version, generator, bound, schema */
    0x07230203, 0x00010000, 0x000d000a, 0x00000014, 0x00000000,
    /* OpCapability Shader */
    0x00020011, 0x00000001,
    /* OpMemoryModel Logical GLSL450 */
    0x0003000e, 0x00000000, 0x00000001,
    /* OpEntryPoint GLCompute %main "main" */
    0x0005000f, 0x00000005, 0x00000002, 0x6e69616d, 0x00000000,
    /* OpExecutionMode %main LocalSize 1 1 1 */
    0x00060010, 0x00000002, 0x00000011, 0x00000001, 0x00000001, 0x00000001,
    /* OpDecorate %_runtimearr_float ArrayStride 4 */
    0x00040047, 0x00000003, 0x00000006, 0x00000004,
    /* OpMemberDecorate %Output 0 Offset 0 */
    0x00050048, 0x00000004, 0x00000000, 0x00000023, 0x00000000,
    /* OpDecorate %Output Block */
    0x00030047, 0x00000004, 0x00000002,
    /* OpDecorate %output_buf DescriptorSet 0 */
    0x00040047, 0x00000005, 0x00000022, 0x00000000,
    /* OpDecorate %output_buf Binding 0 */
    0x00040047, 0x00000005, 0x00000021, 0x00000000,
    /* Types */
    0x00020013, 0x00000006,  /* OpTypeVoid */
    0x00030021, 0x00000007, 0x00000006,  /* OpTypeFunction void */
    0x00030016, 0x00000008, 0x00000020,  /* OpTypeFloat 32 */
    0x0003001d, 0x00000003, 0x00000008,  /* OpTypeRuntimeArray float */
    0x0003001e, 0x00000004, 0x00000003,  /* OpTypeStruct { float[] } */
    0x00040020, 0x00000009, 0x00000002, 0x00000004,  /* OpTypePointer StorageBuffer %Output */
    0x00040020, 0x0000000a, 0x00000002, 0x00000008,  /* OpTypePointer StorageBuffer float */
    0x00040015, 0x0000000b, 0x00000020, 0x00000001,  /* OpTypeInt 32 signed */
    0x0004002b, 0x0000000b, 0x0000000c, 0x00000000,  /* OpConstant int 0 */
    0x0004002b, 0x00000008, 0x0000000d, 0x3f800000,  /* OpConstant float 1.0 */
    /* Variable */
    0x0004003b, 0x00000009, 0x00000005, 0x00000002,  /* %output_buf = OpVariable StorageBuffer */
    /* Function */
    0x00050036, 0x00000006, 0x00000002, 0x00000000, 0x00000007,  /* OpFunction */
    0x000200f8, 0x0000000e,  /* OpLabel */
    0x00050041, 0x0000000a, 0x0000000f, 0x00000005, 0x0000000c,  /* OpAccessChain */
    0x00050041, 0x0000000a, 0x00000010, 0x0000000f, 0x0000000c,  /* OpAccessChain (into array) */
    0x0003003e, 0x00000010, 0x0000000d,  /* OpStore */
    0x000100fd,  /* OpReturn */
    0x00010038,  /* OpFunctionEnd */
};

/* Actually, let's not inline SPIR-V since getting it exactly right is fragile.
 * Instead, compile a minimal NOP shader at build time and embed it.
 * For robustness, let's just use the same approach as vk_tmu_probe:
 * compile the shader separately and embed the .spv file.
 *
 * BUT — since we want this to be standalone, let's just create the
 * simplest possible compute shader at runtime using the driver's
 * shader module creation. We'll use glslc on the host to pre-compile.
 *
 * SIMPLEST approach: reuse the dispatch_nop_spv.h from the existing probe.
 * We'll just compile with -I pointing to the shaders dir.
 */

/* If building standalone without the header, define a fallback */
#ifndef shaders_dispatch_nop_spv_len
/* We'll use a runtime-created trivial shader instead */
#define USE_RUNTIME_SHADER 1
#endif

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

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static void print_stats(const char *label, double *t, int n) {
    double sum = 0, mn = 1e18, mx = 0;
    for (int i = 0; i < n; i++) { sum += t[i]; if (t[i] < mn) mn = t[i]; if (t[i] > mx) mx = t[i]; }
    double s[n]; memcpy(s, t, n * sizeof(double));
    qsort(s, n, sizeof(double), cmp_double);
    printf("  %-44s mean=%7.1f  min=%7.1f  p50=%7.1f  p99=%7.1f  max=%7.1f us\n",
           label, sum / n, mn, s[n / 2], s[(int)(n * 0.99)], mx);
}

/* ── Vulkan context ── */
typedef struct {
    VkInstance       instance;
    VkPhysicalDevice pdev;
    VkDevice         device;
    uint32_t         queue_family;
    VkQueue          queue;
    VkCommandPool    cmd_pool;
    VkPhysicalDeviceMemoryProperties mem_props;
} Ctx;

static uint32_t find_mem(Ctx *c, uint32_t bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (c->mem_props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    fprintf(stderr, "no memory type\n"); exit(1);
}

static void init_ctx(Ctx *c) {
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "persistent_dispatch", .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    CHECK_VK(vkCreateInstance(&ici, NULL, &c->instance));

    uint32_t n = 1;
    CHECK_VK(vkEnumeratePhysicalDevices(c->instance, &n, &c->pdev));
    vkGetPhysicalDeviceMemoryProperties(c->pdev, &c->mem_props);

    uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->pdev, &qfc, NULL);
    VkQueueFamilyProperties *qf = malloc(qfc * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(c->pdev, &qfc, qf);
    c->queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qfc; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c->queue_family = i; break; }
    free(qf);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = c->queue_family, .queueCount = 1, .pQueuePriorities = &prio };

    /* Enable timeline semaphores */
    VkPhysicalDeviceTimelineSemaphoreFeatures ts_feat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        .timelineSemaphore = VK_TRUE,
    };

    const char *exts[] = { "VK_KHR_timeline_semaphore" };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &ts_feat,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts };
    CHECK_VK(vkCreateDevice(c->pdev, &dci, NULL, &c->device));
    vkGetDeviceQueue(c->device, c->queue_family, 0, &c->queue);

    VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = c->queue_family };
    CHECK_VK(vkCreateCommandPool(c->device, &pci, NULL, &c->cmd_pool));

    /* Load timeline semaphore function pointers */
    pfn_vkSignalSemaphore = (PFN_vkSignalSemaphoreKHR)vkGetDeviceProcAddr(c->device, "vkSignalSemaphoreKHR");
    pfn_vkWaitSemaphores = (PFN_vkWaitSemaphoresKHR)vkGetDeviceProcAddr(c->device, "vkWaitSemaphoresKHR");
    if (!pfn_vkSignalSemaphore || !pfn_vkWaitSemaphores) {
        fprintf(stderr, "WARNING: timeline semaphore functions not available, Test 3 will be skipped\n");
    }
}

/* Create a simple storage buffer */
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; } Buf;

static Buf make_buf(Ctx *c, VkDeviceSize sz) {
    Buf b;
    VkBufferCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sz, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
    CHECK_VK(vkCreateBuffer(c->device, &ci, NULL, &b.buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(c->device, b.buf, &req);
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = find_mem(c, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
    CHECK_VK(vkAllocateMemory(c->device, &ai, NULL, &b.mem));
    CHECK_VK(vkBindBufferMemory(c->device, b.buf, b.mem, 0));
    CHECK_VK(vkMapMemory(c->device, b.mem, 0, sz, 0, &b.map));
    return b;
}

int main(void) {
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  PROBE A: Persistent Dispatch Latency\n");
    printf("  GPU: PowerVR BXM-8-256 (Vulkan 1.1)\n");
    printf("  Question: How much of 386us is recording vs driver/HW?\n");
    printf("════════════════════════════════════════════════════════════\n");

    Ctx c;
    init_ctx(&c);

    /* ── Create NOP pipeline ── */
    /* Compile NOP shader from embedded SPIR-V.
     * This is the simplest possible compute shader:
     *   layout(local_size_x=1) in;
     *   layout(set=0,binding=0) buffer O { float d[]; } o;
     *   void main() { o.d[0] = 1.0; }
     *
     * We need the pre-compiled SPIR-V. Let's read it from a file on device.
     */

    /* Actually, let's embed the SPIR-V properly. The NOP shader SPIR-V is small. */
    /* We'll compile the shader externally and use xxd. For now, compile with -I. */

    /* For a truly standalone probe, create the shader module from a file */
    FILE *spvf = fopen("dispatch_nop.spv", "rb");
    if (!spvf) {
        fprintf(stderr, "Cannot open dispatch_nop.spv — compile with:\n");
        fprintf(stderr, "  glslc --target-env=vulkan1.1 -o dispatch_nop.spv dispatch_nop.comp\n");
        fprintf(stderr, "Or put dispatch_nop.spv in /data/local/tmp/\n");
        return 1;
    }
    fseek(spvf, 0, SEEK_END);
    long spv_sz = ftell(spvf);
    fseek(spvf, 0, SEEK_SET);
    uint32_t *spv = malloc(spv_sz);
    fread(spv, 1, spv_sz, spvf);
    fclose(spvf);

    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_sz, .pCode = spv };
    VkShaderModule shader;
    CHECK_VK(vkCreateShaderModule(c.device, &smci, NULL, &shader));
    free(spv);

    /* Descriptor set layout */
    VkDescriptorSetLayoutBinding bind = { .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &bind };
    VkDescriptorSetLayout dsl;
    CHECK_VK(vkCreateDescriptorSetLayout(c.device, &dslci, NULL, &dsl));

    /* Pipeline layout */
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl };
    VkPipelineLayout pl;
    CHECK_VK(vkCreatePipelineLayout(c.device, &plci, NULL, &pl));

    /* Compute pipeline */
    VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main" },
        .layout = pl };
    VkPipeline pipe;
    CHECK_VK(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

    /* Descriptor pool + set */
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 4, .poolSizeCount = 1, .pPoolSizes = &dps };
    VkDescriptorPool dp;
    CHECK_VK(vkCreateDescriptorPool(c.device, &dpci, NULL, &dp));

    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    VkDescriptorSet ds;
    CHECK_VK(vkAllocateDescriptorSets(c.device, &dsai, &ds));

    /* Buffer + bind to descriptor */
    Buf out = make_buf(&c, 4096);  /* 1 page, more than enough */
    VkDescriptorBufferInfo dbi = { .buffer = out.buf, .offset = 0, .range = 4096 };
    VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi };
    vkUpdateDescriptorSets(c.device, 1, &wds, 0, NULL);

    int WARMUP = 50;
    int ITERS = 500;

    /* ═══════════════════════════════════════════════════════════
     * TEST 1: Cold dispatch (re-record every time) — 386us baseline
     * ═══════════════════════════════════════════════════════════ */
    printf("\n  === Test 1: Cold dispatch (re-record + submit + wait) ===\n");
    {
        VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = c.cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VkCommandBuffer cmd;
        CHECK_VK(vkAllocateCommandBuffers(c.device, &cbai, &cmd));

        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkFence fence;
        CHECK_VK(vkCreateFence(c.device, &fci, NULL, &fence));

        /* Warmup */
        for (int i = 0; i < WARMUP; i++) {
            CHECK_VK(vkResetCommandBuffer(cmd, 0));
            VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
            CHECK_VK(vkBeginCommandBuffer(cmd, &bi));
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
            vkCmdDispatch(cmd, 1, 1, 1);
            CHECK_VK(vkEndCommandBuffer(cmd));
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
            CHECK_VK(vkResetFences(c.device, 1, &fence));
            CHECK_VK(vkQueueSubmit(c.queue, 1, &si, fence));
            CHECK_VK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
        }

        /* Measure */
        double *times = calloc(ITERS, sizeof(double));
        for (int i = 0; i < ITERS; i++) {
            CHECK_VK(vkResetCommandBuffer(cmd, 0));
            VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
            double t0 = now_us();
            CHECK_VK(vkBeginCommandBuffer(cmd, &bi));
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
            vkCmdDispatch(cmd, 1, 1, 1);
            CHECK_VK(vkEndCommandBuffer(cmd));
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
            CHECK_VK(vkResetFences(c.device, 1, &fence));
            CHECK_VK(vkQueueSubmit(c.queue, 1, &si, fence));
            CHECK_VK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
            times[i] = now_us() - t0;
        }
        print_stats("Cold (record+submit+wait)", times, ITERS);
        free(times);
        vkDestroyFence(c.device, fence, NULL);
        vkFreeCommandBuffers(c.device, c.cmd_pool, 1, &cmd);
    }

    /* ═══════════════════════════════════════════════════════════
     * TEST 2: Pre-recorded resubmit (record once, submit many)
     * ═══════════════════════════════════════════════════════════ */
    printf("\n  === Test 2: Pre-recorded resubmit (submit same cmdbuf) ===\n");
    {
        VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = c.cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VkCommandBuffer cmd;
        CHECK_VK(vkAllocateCommandBuffers(c.device, &cbai, &cmd));

        /* Record ONCE — no ONE_TIME_SUBMIT flag */
        VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        CHECK_VK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdDispatch(cmd, 1, 1, 1);
        CHECK_VK(vkEndCommandBuffer(cmd));

        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkFence fence;
        CHECK_VK(vkCreateFence(c.device, &fci, NULL, &fence));

        /* Warmup */
        for (int i = 0; i < WARMUP; i++) {
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
            CHECK_VK(vkResetFences(c.device, 1, &fence));
            CHECK_VK(vkQueueSubmit(c.queue, 1, &si, fence));
            CHECK_VK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
        }

        /* Measure */
        double *times = calloc(ITERS, sizeof(double));
        for (int i = 0; i < ITERS; i++) {
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
            CHECK_VK(vkResetFences(c.device, 1, &fence));
            double t0 = now_us();
            CHECK_VK(vkQueueSubmit(c.queue, 1, &si, fence));
            CHECK_VK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
            times[i] = now_us() - t0;
        }
        print_stats("Pre-recorded (submit+wait only)", times, ITERS);
        free(times);
        vkDestroyFence(c.device, fence, NULL);
        vkFreeCommandBuffers(c.device, c.cmd_pool, 1, &cmd);
    }

    /* ═══════════════════════════════════════════════════════════
     * TEST 3: Timeline semaphore (signal from CPU, GPU waits+runs)
     * ═══════════════════════════════════════════════════════════ */
    printf("\n  === Test 3: Timeline semaphore (CPU signal → GPU work → CPU wait) ===\n");
    {
        /* Create timeline semaphore */
        VkSemaphoreTypeCreateInfo stci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0 };
        VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &stci };
        VkSemaphore timeline;
        CHECK_VK(vkCreateSemaphore(c.device, &sci, NULL, &timeline));

        /* Pre-record command buffer */
        VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = c.cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VkCommandBuffer cmd;
        CHECK_VK(vkAllocateCommandBuffers(c.device, &cbai, &cmd));

        VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        CHECK_VK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdDispatch(cmd, 1, 1, 1);
        CHECK_VK(vkEndCommandBuffer(cmd));

        /* Submit with timeline: wait on value N, signal value N+1 */
        /* The GPU submission waits for timeline=N, then executes and signals timeline=N+1 */
        /* CPU signals timeline=N to kick it off, then waits for timeline=N+1 */

        uint64_t counter = 0;

        /* Warmup */
        for (int i = 0; i < WARMUP; i++) {
            uint64_t wait_val = counter;
            uint64_t signal_val = counter + 1;
            VkTimelineSemaphoreSubmitInfo tssi = { .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &wait_val,
                .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &signal_val };
            VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &tssi,
                .waitSemaphoreCount = 1, .pWaitSemaphores = &timeline, .pWaitDstStageMask = &wait_stage,
                .commandBufferCount = 1, .pCommandBuffers = &cmd,
                .signalSemaphoreCount = 1, .pSignalSemaphores = &timeline };

            /* CPU signals the wait value so GPU can start */
            VkSemaphoreSignalInfo ssi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                .semaphore = timeline, .value = wait_val };
            CHECK_VK(vkSignalSemaphore(c.device, &ssi));

            CHECK_VK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));

            VkSemaphoreWaitInfo swi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .semaphoreCount = 1, .pSemaphores = &timeline, .pValues = &signal_val };
            CHECK_VK(vkWaitSemaphores(c.device, &swi, UINT64_MAX));

            counter = signal_val;
        }

        /* Measure: submit first (GPU waits for signal), then CPU signals, then CPU waits for completion */
        double *times = calloc(ITERS, sizeof(double));
        for (int i = 0; i < ITERS; i++) {
            uint64_t wait_val = counter;
            uint64_t signal_val = counter + 1;
            VkTimelineSemaphoreSubmitInfo tssi = { .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &wait_val,
                .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &signal_val };
            VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &tssi,
                .waitSemaphoreCount = 1, .pWaitSemaphores = &timeline, .pWaitDstStageMask = &wait_stage,
                .commandBufferCount = 1, .pCommandBuffers = &cmd,
                .signalSemaphoreCount = 1, .pSignalSemaphores = &timeline };

            /* Pre-submit the work (GPU will stall waiting for timeline=wait_val) */
            CHECK_VK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));

            /* NOW measure: signal → completion */
            VkSemaphoreSignalInfo ssi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                .semaphore = timeline, .value = wait_val };

            double t0 = now_us();
            CHECK_VK(vkSignalSemaphore(c.device, &ssi));  /* kick GPU */

            VkSemaphoreWaitInfo swi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .semaphoreCount = 1, .pSemaphores = &timeline, .pValues = &signal_val };
            CHECK_VK(vkWaitSemaphores(c.device, &swi, UINT64_MAX));
            times[i] = now_us() - t0;

            counter = signal_val;
        }
        print_stats("Timeline (signal→completion)", times, ITERS);
        free(times);
        vkDestroySemaphore(c.device, timeline, NULL);
        vkFreeCommandBuffers(c.device, c.cmd_pool, 1, &cmd);
    }

    /* ═══════════════════════════════════════════════════════════
     * TEST 4: Submit-only latency (don't wait, just measure submit cost)
     * ═══════════════════════════════════════════════════════════ */
    printf("\n  === Test 4: Submit-only (vkQueueSubmit cost, no wait) ===\n");
    {
        VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = c.cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VkCommandBuffer cmd;
        CHECK_VK(vkAllocateCommandBuffers(c.device, &cbai, &cmd));

        VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        CHECK_VK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdDispatch(cmd, 1, 1, 1);
        CHECK_VK(vkEndCommandBuffer(cmd));

        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkFence fence;
        CHECK_VK(vkCreateFence(c.device, &fci, NULL, &fence));

        /* Warmup — must wait between each to not overflow queue */
        for (int i = 0; i < WARMUP; i++) {
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
            CHECK_VK(vkResetFences(c.device, 1, &fence));
            CHECK_VK(vkQueueSubmit(c.queue, 1, &si, fence));
            CHECK_VK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
        }

        /* Measure just the submit cost (still need to drain between iterations) */
        double *times_submit = calloc(ITERS, sizeof(double));
        double *times_wait = calloc(ITERS, sizeof(double));
        for (int i = 0; i < ITERS; i++) {
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
            CHECK_VK(vkResetFences(c.device, 1, &fence));

            double t0 = now_us();
            CHECK_VK(vkQueueSubmit(c.queue, 1, &si, fence));
            double t1 = now_us();
            CHECK_VK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
            double t2 = now_us();

            times_submit[i] = t1 - t0;
            times_wait[i] = t2 - t1;
        }
        print_stats("Submit only (vkQueueSubmit)", times_submit, ITERS);
        print_stats("Wait only (vkWaitForFences)", times_wait, ITERS);
        free(times_submit); free(times_wait);
        vkDestroyFence(c.device, fence, NULL);
        vkFreeCommandBuffers(c.device, c.cmd_pool, 1, &cmd);
    }

    /* ═══════════════════════════════════════════════════════════
     * TEST 5: Larger workload — 1024 elements (D_MODEL sized)
     * ═══════════════════════════════════════════════════════════ */
    printf("\n  === Test 5: Pre-recorded, 1024-element NOP (bigger dispatch) ===\n");
    {
        /* Same NOP shader but dispatch with 1024 invocations to simulate real work size */
        VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = c.cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VkCommandBuffer cmd;
        CHECK_VK(vkAllocateCommandBuffers(c.device, &cbai, &cmd));

        VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        CHECK_VK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdDispatch(cmd, 1024, 1, 1);  /* 1024 workgroups of 1 = 1024 invocations */
        CHECK_VK(vkEndCommandBuffer(cmd));

        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkFence fence;
        CHECK_VK(vkCreateFence(c.device, &fci, NULL, &fence));

        for (int i = 0; i < WARMUP; i++) {
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
            CHECK_VK(vkResetFences(c.device, 1, &fence));
            CHECK_VK(vkQueueSubmit(c.queue, 1, &si, fence));
            CHECK_VK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
        }

        double *times = calloc(ITERS, sizeof(double));
        for (int i = 0; i < ITERS; i++) {
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
            CHECK_VK(vkResetFences(c.device, 1, &fence));
            double t0 = now_us();
            CHECK_VK(vkQueueSubmit(c.queue, 1, &si, fence));
            CHECK_VK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
            times[i] = now_us() - t0;
        }
        print_stats("Pre-recorded 1024-dispatch", times, ITERS);
        free(times);
        vkDestroyFence(c.device, fence, NULL);
        vkFreeCommandBuffers(c.device, c.cmd_pool, 1, &cmd);
    }

    printf("\n════════════════════════════════════════════════════════════\n\n");

    /* Cleanup */
    vkDestroyPipeline(c.device, pipe, NULL);
    vkDestroyPipelineLayout(c.device, pl, NULL);
    vkDestroyDescriptorSetLayout(c.device, dsl, NULL);
    vkDestroyDescriptorPool(c.device, dp, NULL);
    vkDestroyShaderModule(c.device, shader, NULL);
    vkDestroyCommandPool(c.device, c.cmd_pool, NULL);
    vkDestroyDevice(c.device, NULL);
    vkDestroyInstance(c.instance, NULL);

    return 0;
}
