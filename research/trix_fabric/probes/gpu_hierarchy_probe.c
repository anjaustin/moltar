/*
 * gpu_hierarchy_probe.c — Map the GPU's memory hierarchy
 *
 * THE QUESTION: What caches exist between the GPU and DRAM?
 * If an SLC (System-Level Cache) exists on the NoC, we should see
 * a bandwidth cliff when the working set exceeds SLC capacity.
 *
 * TECHNIQUE:
 *   For each buffer size (4KB → 16MB):
 *     1. GPU reads the buffer N times (enough to see caching effects)
 *     2. Measure effective bandwidth for first read vs steady-state
 *     3. Cache tiers show as bandwidth plateaus with cliffs between them
 *
 *   Expected hierarchy:
 *     GPU internal (SLC slice?) → SLC → DRAM
 *     Each transition = bandwidth cliff
 *
 *   We also test each Vulkan memory type to see if any pins to SLC.
 *
 * Build:
 *   glslc --target-env=vulkan1.1 -o ace_lite_read.spv ace_lite_read.comp
 *   $CC -O2 -march=armv8.2-a+dotprod+fp16 -o gpu_hierarchy_probe gpu_hierarchy_probe.c -lvulkan -lm
 *
 * Run:
 *   adb shell "cd /data/local/tmp && taskset c0 ./gpu_hierarchy_probe"
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
static uint16_t f32_to_f16(float v) {
    uint32_t x; memcpy(&x, &v, 4);
    uint16_t s = (x >> 16) & 0x8000;
    int e = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7C00;
    return s | (e << 10) | (m >> 13);
}

/* ── Vulkan infra ── */
typedef struct {
    VkInstance inst; VkPhysicalDevice pdev; VkDevice dev;
    uint32_t qf; VkQueue queue; VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties mprops;
} Vk;

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; VkDeviceSize sz; } Buf;

static uint32_t find_mem(Vk *v, uint32_t bits, VkMemoryPropertyFlags f) {
    for(uint32_t i=0; i<v->mprops.memoryTypeCount; i++)
        if((bits&(1u<<i)) && (v->mprops.memoryTypes[i].propertyFlags&f)==f) return i;
    return UINT32_MAX;
}

static Buf make_buf_memtype(Vk *v, VkDeviceSize sz, VkBufferUsageFlags usage, uint32_t memtype) {
    Buf b = {.sz=sz};
    VkBufferCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size=sz, .usage=usage};
    CHECK_VK(vkCreateBuffer(v->dev, &ci, NULL, &b.buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(v->dev, b.buf, &req);

    /* Verify requested type is compatible */
    if (!(req.memoryTypeBits & (1u << memtype))) {
        fprintf(stderr, "  [warn] memtype %u not compatible (bits=0x%x)\n", memtype, req.memoryTypeBits);
        vkDestroyBuffer(v->dev, b.buf, NULL);
        b.buf = VK_NULL_HANDLE;
        return b;
    }

    VkMemoryAllocateInfo ai = {.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize=req.size, .memoryTypeIndex=memtype};
    VkResult r = vkAllocateMemory(v->dev, &ai, NULL, &b.mem);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  [warn] alloc failed for memtype %u (err=%d)\n", memtype, r);
        vkDestroyBuffer(v->dev, b.buf, NULL);
        b.buf = VK_NULL_HANDLE;
        return b;
    }
    CHECK_VK(vkBindBufferMemory(v->dev, b.buf, b.mem, 0));
    r = vkMapMemory(v->dev, b.mem, 0, sz, 0, &b.map);
    if (r != VK_SUCCESS) {
        b.map = NULL; /* device-local only, can't map */
    }
    return b;
}

static Buf make_buf(Vk *v, VkDeviceSize sz, VkBufferUsageFlags usage) {
    uint32_t mt = find_mem(v, UINT32_MAX,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    return make_buf_memtype(v, sz, usage, mt);
}

static void free_buf(Vk *v, Buf *b) {
    if (b->map) vkUnmapMemory(v->dev, b->mem);
    if (b->buf) vkDestroyBuffer(v->dev, b->buf, NULL);
    vkFreeMemory(v->dev, b->mem, NULL);
}

static void init_vk(Vk *v) {
    VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion=VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app};
    CHECK_VK(vkCreateInstance(&ici, NULL, &v->inst));
    uint32_t n=1; CHECK_VK(vkEnumeratePhysicalDevices(v->inst, &n, &v->pdev));
    vkGetPhysicalDeviceMemoryProperties(v->pdev, &v->mprops);
    uint32_t qfc=0; vkGetPhysicalDeviceQueueFamilyProperties(v->pdev, &qfc, NULL);
    VkQueueFamilyProperties *qfp = malloc(qfc * sizeof(*qfp));
    vkGetPhysicalDeviceQueueFamilyProperties(v->pdev, &qfc, qfp);
    v->qf = UINT32_MAX;
    for(uint32_t i=0; i<qfc; i++) if(qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { v->qf=i; break; }
    free(qfp);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=v->qf, .queueCount=1, .pQueuePriorities=&prio};
    VkPhysicalDevice16BitStorageFeatures f16store = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .storageBuffer16BitAccess=VK_TRUE, .uniformAndStorageBuffer16BitAccess=VK_TRUE};
    VkPhysicalDeviceShaderFloat16Int8Features f16math = {
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .shaderFloat16=VK_TRUE, .pNext=&f16store};
    const char *exts[] = {"VK_KHR_16bit_storage", "VK_KHR_shader_float16_int8"};
    VkDeviceCreateInfo dci = {.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext=&f16math,
        .queueCreateInfoCount=1, .pQueueCreateInfos=&qci,
        .enabledExtensionCount=2, .ppEnabledExtensionNames=exts};
    CHECK_VK(vkCreateDevice(v->pdev, &dci, NULL, &v->dev));
    vkGetDeviceQueue(v->dev, v->qf, 0, &v->queue);
    VkCommandPoolCreateInfo pci = {.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex=v->qf};
    CHECK_VK(vkCreateCommandPool(v->dev, &pci, NULL, &v->pool));
}

/* ── Pipeline (reuses ace_lite_read.spv — same simple reader) ── */
typedef struct {
    VkPipeline pipe; VkPipelineLayout layout;
    VkDescriptorSet ds; VkDescriptorSetLayout dsl; VkDescriptorPool dp;
} Pipe;

static Pipe make_pipe(Vk *v, const char *spv) {
    Pipe p;
    FILE *f = fopen(spv, "rb");
    if(!f) { fprintf(stderr, "Cannot open %s\n", spv); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc(sz); fread(code, 1, sz, f); fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize=sz, .pCode=code};
    CHECK_VK(vkCreateShaderModule(v->dev, &smci, NULL, &sm)); free(code);

    VkDescriptorSetLayoutBinding binds[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}
    };
    VkDescriptorSetLayoutCreateInfo dslci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount=2, .pBindings=binds};
    CHECK_VK(vkCreateDescriptorSetLayout(v->dev, &dslci, NULL, &p.dsl));

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo plci = {.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1, .pSetLayouts=&p.dsl,
        .pushConstantRangeCount=1, .pPushConstantRanges=&pcr};
    CHECK_VK(vkCreatePipelineLayout(v->dev, &plci, NULL, &p.layout));

    VkComputePipelineCreateInfo cpci = {.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sm, .pName="main"},
        .layout=p.layout};
    CHECK_VK(vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &p.pipe));
    vkDestroyShaderModule(v->dev, sm, NULL);

    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets=1, .poolSizeCount=1, .pPoolSizes=&dps};
    CHECK_VK(vkCreateDescriptorPool(v->dev, &dpci, NULL, &p.dp));
    VkDescriptorSetAllocateInfo dsai = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=p.dp, .descriptorSetCount=1, .pSetLayouts=&p.dsl};
    CHECK_VK(vkAllocateDescriptorSets(v->dev, &dsai, &p.ds));
    return p;
}

static void bind_bufs(Vk *v, Pipe *p, Buf *in, Buf *out) {
    VkDescriptorBufferInfo dbis[2] = { {in->buf, 0, in->sz}, {out->buf, 0, out->sz} };
    VkWriteDescriptorSet wds[2];
    for(int i=0; i<2; i++)
        wds[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=p->ds, .dstBinding=i, .descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo=&dbis[i]};
    vkUpdateDescriptorSets(v->dev, 2, wds, 0, NULL);
}

static double fire_gpu(Vk *v, Pipe *p, uint32_t pc[4], uint32_t num_wgs) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool, .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount=1};
    CHECK_VK(vkAllocateCommandBuffers(v->dev, &cbai, &cmd));
    VkCommandBufferBeginInfo bi = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK_VK(vkBeginCommandBuffer(cmd, &bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &p->ds, 0, NULL);
    vkCmdPushConstants(cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(cmd, num_wgs, 1, 1);
    CHECK_VK(vkEndCommandBuffer(cmd));

    VkFence fence;
    VkFenceCreateInfo fci = {.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK_VK(vkCreateFence(v->dev, &fci, NULL, &fence));
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&cmd};

    double t0 = now_us();
    CHECK_VK(vkQueueSubmit(v->queue, 1, &si, fence));
    CHECK_VK(vkWaitForFences(v->dev, 1, &fence, VK_TRUE, UINT64_MAX));
    double elapsed = now_us() - t0;

    vkDestroyFence(v->dev, fence, NULL);
    vkFreeCommandBuffers(v->dev, v->pool, 1, &cmd);
    return elapsed;
}

/* ── Multi-read dispatch: N reads in one command buffer ── */
static double fire_gpu_multi(Vk *v, Pipe *p, uint32_t pc[4], uint32_t num_wgs, int repeats) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=v->pool, .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount=1};
    CHECK_VK(vkAllocateCommandBuffers(v->dev, &cbai, &cmd));
    VkCommandBufferBeginInfo bi = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK_VK(vkBeginCommandBuffer(cmd, &bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &p->ds, 0, NULL);
    vkCmdPushConstants(cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);

    /* Record N dispatches + barriers in one command buffer */
    for (int r = 0; r < repeats; r++) {
        vkCmdDispatch(cmd, num_wgs, 1, 1);
        if (r < repeats - 1) {
            VkMemoryBarrier mb = {.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        }
    }
    CHECK_VK(vkEndCommandBuffer(cmd));

    VkFence fence;
    VkFenceCreateInfo fci = {.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK_VK(vkCreateFence(v->dev, &fci, NULL, &fence));
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount=1, .pCommandBuffers=&cmd};

    double t0 = now_us();
    CHECK_VK(vkQueueSubmit(v->queue, 1, &si, fence));
    CHECK_VK(vkWaitForFences(v->dev, 1, &fence, VK_TRUE, UINT64_MAX));
    double elapsed = now_us() - t0;

    vkDestroyFence(v->dev, fence, NULL);
    vkFreeCommandBuffers(v->dev, v->pool, 1, &cmd);
    return elapsed;
}

/* ══════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  GPU Memory Hierarchy Probe\n");
    printf("  Map cache tiers: GPU internal → SLC → DRAM\n");
    printf("══════════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);

    pin(6);
    Vk v; init_vk(&v);
    printf("  [init] Vulkan ready\n"); fflush(stdout);

    /* ── Dump all memory types ── */
    printf("\n  ═══ VULKAN MEMORY TYPES ═══\n\n");
    for (uint32_t i = 0; i < v.mprops.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = v.mprops.memoryTypes[i].propertyFlags;
        uint32_t heap = v.mprops.memoryTypes[i].heapIndex;
        printf("    type[%u]: heap=%u  flags=0x%03x", i, heap, f);
        if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)     printf(" DEVICE_LOCAL");
        if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)     printf(" HOST_VISIBLE");
        if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)    printf(" HOST_COHERENT");
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)      printf(" HOST_CACHED");
        if (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) printf(" LAZY");
        if (f & VK_MEMORY_PROPERTY_PROTECTED_BIT)        printf(" PROTECTED");
        printf("\n");
    }
    printf("\n  ═══ VULKAN MEMORY HEAPS ═══\n\n");
    for (uint32_t i = 0; i < v.mprops.memoryHeapCount; i++) {
        VkMemoryHeap h = v.mprops.memoryHeaps[i];
        printf("    heap[%u]: %llu MB  flags=0x%x", i,
               (unsigned long long)(h.size / (1024*1024)), h.flags);
        if (h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) printf(" DEVICE_LOCAL");
        printf("\n");
    }
    printf("\n");
    fflush(stdout);

    Pipe pipe = make_pipe(&v, "ace_lite_read.spv");
    printf("  [init] Pipeline ready\n\n"); fflush(stdout);

    /* ═══ PHASE 1: Bandwidth sweep — single dispatch per submit ═══ */
    printf("  ═══ PHASE 1: GPU Bandwidth vs Buffer Size (single dispatch) ═══\n\n");
    printf("  Each size: 3 warmup + 10 measured dispatches\n");
    printf("  Reports: mean bandwidth (MB/s) and per-read time\n\n");
    printf("  %8s  %10s  %10s  %10s  %10s\n", "Size", "Mean(us)", "Min(us)", "BW(MB/s)", "BW_peak");
    printf("  %8s  %10s  %10s  %10s  %10s\n", "--------", "----------", "----------", "----------", "----------");
    fflush(stdout);

    const uint32_t CHUNK_SIZE = 8192;
    const int WARMUP = 3, ITERS = 10;

    /* Sizes: 4KB to 16MB in ~2x steps */
    size_t sizes[] = {
        4*1024, 8*1024, 16*1024, 32*1024, 64*1024,
        128*1024, 256*1024, 384*1024, 512*1024, 768*1024,
        1024*1024, 1536*1024, 2*1024*1024, 3*1024*1024, 4*1024*1024,
        6*1024*1024, 8*1024*1024, 12*1024*1024, 16*1024*1024
    };
    int NUM_SIZES = sizeof(sizes) / sizeof(sizes[0]);

    double bw_results[32] = {0};

    for (int si = 0; si < NUM_SIZES; si++) {
        size_t bytes = sizes[si];
        uint32_t num_f16 = (uint32_t)(bytes / 2);
        uint32_t num_wgs = (num_f16 + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (num_wgs == 0) num_wgs = 1;

        Buf in_buf = make_buf(&v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buf out_buf = make_buf(&v, num_wgs * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        /* Fill with data */
        if (in_buf.map) {
            uint16_t *data = (uint16_t *)in_buf.map;
            for(uint32_t i = 0; i < num_f16; i++)
                data[i] = f32_to_f16(((float)(i % 1000) / 1000.0f - 0.5f) * 2.0f);
        }

        bind_bufs(&v, &pipe, &in_buf, &out_buf);
        uint32_t pc[4] = { num_f16, CHUNK_SIZE, 0, 0 };

        /* Warmup */
        for (int i = 0; i < WARMUP; i++) fire_gpu(&v, &pipe, pc, num_wgs);

        /* Measure */
        double t[ITERS];
        for (int i = 0; i < ITERS; i++) t[i] = fire_gpu(&v, &pipe, pc, num_wgs);

        double sum=0, mn=1e18;
        for(int i=0;i<ITERS;i++) { sum+=t[i]; if(t[i]<mn)mn=t[i]; }
        double mean = sum / ITERS;
        double bw_mean = (double)bytes / mean;  /* bytes/us = MB/s */
        double bw_peak = (double)bytes / mn;
        bw_results[si] = bw_mean;

        const char *unit = "KB";
        double disp = bytes / 1024.0;
        if (bytes >= 1024*1024) { unit = "MB"; disp = bytes / (1024.0*1024.0); }
        printf("  %6.0f%s  %10.1f  %10.1f  %10.1f  %10.1f\n",
               disp, unit, mean, mn, bw_mean, bw_peak);
        fflush(stdout);

        free_buf(&v, &in_buf);
        free_buf(&v, &out_buf);
    }

    /* ═══ PHASE 2: Multi-read in single command buffer (amortize dispatch) ═══ */
    printf("\n  ═══ PHASE 2: Repeated reads (10x in one cmdbuf, amortize dispatch) ═══\n\n");
    printf("  %8s  %10s  %10s  %10s\n", "Size", "Total(us)", "Per-read", "BW(MB/s)");
    printf("  %8s  %10s  %10s  %10s\n", "--------", "----------", "----------", "----------");
    fflush(stdout);

    const int MULTI_READS = 10;

    for (int si = 0; si < NUM_SIZES; si++) {
        size_t bytes = sizes[si];
        uint32_t num_f16 = (uint32_t)(bytes / 2);
        uint32_t num_wgs = (num_f16 + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (num_wgs == 0) num_wgs = 1;

        Buf in_buf = make_buf(&v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buf out_buf = make_buf(&v, num_wgs * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        if (in_buf.map) {
            uint16_t *data = (uint16_t *)in_buf.map;
            for(uint32_t i = 0; i < num_f16; i++)
                data[i] = f32_to_f16(((float)(i % 1000) / 1000.0f - 0.5f) * 2.0f);
        }

        bind_bufs(&v, &pipe, &in_buf, &out_buf);
        uint32_t pc[4] = { num_f16, CHUNK_SIZE, 0, 0 };

        /* Warmup */
        for (int i = 0; i < 2; i++) fire_gpu_multi(&v, &pipe, pc, num_wgs, MULTI_READS);

        /* Measure */
        double t[ITERS];
        for (int i = 0; i < ITERS; i++)
            t[i] = fire_gpu_multi(&v, &pipe, pc, num_wgs, MULTI_READS);

        double sum=0;
        for(int i=0;i<ITERS;i++) sum+=t[i];
        double total_mean = sum / ITERS;
        double per_read = total_mean / MULTI_READS;
        double bw = (double)bytes / per_read; /* MB/s per read */

        const char *unit = "KB";
        double disp = bytes / 1024.0;
        if (bytes >= 1024*1024) { unit = "MB"; disp = bytes / (1024.0*1024.0); }
        printf("  %6.0f%s  %10.1f  %10.1f  %10.1f\n", disp, unit, total_mean, per_read, bw);
        fflush(stdout);

        free_buf(&v, &in_buf);
        free_buf(&v, &out_buf);
    }

    /* ═══ PHASE 3: Test different memory types ═══ */
    printf("\n  ═══ PHASE 3: Bandwidth by Vulkan memory type (256KB buffer) ═══\n\n");
    fflush(stdout);

    size_t test_sz = 256 * 1024;
    uint32_t test_f16 = (uint32_t)(test_sz / 2);
    uint32_t test_wgs = (test_f16 + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (uint32_t mt = 0; mt < v.mprops.memoryTypeCount; mt++) {
        VkMemoryPropertyFlags f = v.mprops.memoryTypes[mt].propertyFlags;

        /* Need at least host-visible to fill data, or we skip */
        if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            printf("    type[%u]: NOT HOST_VISIBLE — skipping (can't fill data)\n", mt);
            continue;
        }

        Buf in_buf = make_buf_memtype(&v, test_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mt);
        if (in_buf.buf == VK_NULL_HANDLE) continue;

        Buf out_buf = make_buf(&v, test_wgs * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        if (in_buf.map) {
            uint16_t *data = (uint16_t *)in_buf.map;
            for(uint32_t i = 0; i < test_f16; i++)
                data[i] = f32_to_f16(((float)(i % 1000) / 1000.0f - 0.5f) * 2.0f);
        }

        bind_bufs(&v, &pipe, &in_buf, &out_buf);
        uint32_t pc[4] = { test_f16, CHUNK_SIZE, 0, 0 };

        /* Warmup + measure */
        for (int i = 0; i < WARMUP; i++) fire_gpu(&v, &pipe, pc, test_wgs);
        double t[ITERS];
        for (int i = 0; i < ITERS; i++) t[i] = fire_gpu(&v, &pipe, pc, test_wgs);

        double sum=0, mn=1e18;
        for(int i=0;i<ITERS;i++) { sum+=t[i]; if(t[i]<mn)mn=t[i]; }
        double mean = sum / ITERS;
        double bw = (double)test_sz / mean;

        printf("    type[%u]: flags=0x%03x", mt, f);
        if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)  printf(" DEV_LOCAL");
        if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)  printf(" HOST_VIS");
        if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) printf(" HOST_COH");
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)   printf(" HOST_CACHED");
        printf("  → mean=%7.1f us  min=%7.1f us  BW=%.0f MB/s\n", mean, mn, bw);
        fflush(stdout);

        free_buf(&v, &in_buf);
        free_buf(&v, &out_buf);
    }

    printf("\n══════════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);

    return 0;
}
