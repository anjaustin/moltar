/*
 * ace_lite_probe.c — ACE-Lite Cache Coherency Probe
 *
 * THE QUESTION: Can the GPU read data from CPU L2 cache via ACE-Lite snooping,
 * bypassing the DRAM bottleneck that accounts for 98.4% of GPU time?
 *
 * EXPERIMENTAL DESIGN:
 *
 *   For each buffer size (64KB, 128KB, 256KB, 512KB, 1MB):
 *
 *   Test COLD:
 *     1. Allocate host-visible Vulkan buffer (VkBuffer with HOST_COHERENT)
 *     2. Fill with data
 *     3. Flush CPU caches by reading a LARGE unrelated buffer (evict target data)
 *     4. GPU reads the buffer via compute shader → measure time
 *
 *   Test HOT:
 *     1. Same buffer, already filled
 *     2. CPU reads entire buffer sequentially (pull into L2 cache)
 *     3. GPU reads IMMEDIATELY (no delay) → measure time
 *     4. If ACE-Lite snooping works, this should be measurably faster than COLD
 *
 *   Test CONTENTION:
 *     1. CPU heats buffer into L2
 *     2. GPU reads buffer WHILE CPU runs concurrent memcpy on DIFFERENT data
 *     3. Compare bus contention between HOT and COLD scenarios
 *
 * WHAT THE NUMBERS MEAN:
 *   HOT ≈ COLD → ACE-Lite snooping is NOT working (both go to DRAM)
 *   HOT < COLD → ACE-Lite IS snooping CPU cache (potential game-changer)
 *   HOT >> COLD → Something pathological (coherency protocol overhead)
 *
 * Build:
 *   glslc --target-env=vulkan1.1 -o ace_lite_read.spv ace_lite_read.comp
 *   $CC -O2 -march=armv8.2-a+dotprod+fp16 -o ace_lite_probe ace_lite_probe.c -lvulkan -lm
 *
 * Run:
 *   adb shell "cd /data/local/tmp && taskset c0 ./ace_lite_probe"
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
#include <unistd.h>
#include <vulkan/vulkan.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define CHECK_VK(call) do { VkResult _r = (call); \
    if (_r != VK_SUCCESS) { fprintf(stderr, "VK error %d at %s:%d\n", _r, __FILE__, __LINE__); exit(1); } } while(0)

/* ── Timing ── */
static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
static void pin(int cpu) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}
static int cmp_dbl(const void *a, const void *b) {
    double da=*(const double*)a, db=*(const double*)b; return (da>db)-(da<db);
}
static void pstats(const char *label, double *t, int n) {
    double s=0, mn=1e18, mx=0;
    for(int i=0;i<n;i++) { s+=t[i]; if(t[i]<mn)mn=t[i]; if(t[i]>mx)mx=t[i]; }
    double q[n]; memcpy(q,t,n*sizeof(double)); qsort(q,n,sizeof(double),cmp_dbl);
    printf("    %-35s mean=%8.1f  min=%8.1f  p50=%8.1f  max=%8.1f us\n",
           label, s/n, mn, q[n/2], mx);
}

/* ── Vulkan infrastructure (same pattern as other probes) ── */
typedef struct {
    VkInstance inst; VkPhysicalDevice pdev; VkDevice dev;
    uint32_t qf; VkQueue queue; VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties mprops;
} Vk;

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; VkDeviceSize sz; } Buf;

static uint32_t find_mem(Vk *v, uint32_t bits, VkMemoryPropertyFlags f) {
    for(uint32_t i=0; i<v->mprops.memoryTypeCount; i++)
        if((bits&(1u<<i)) && (v->mprops.memoryTypes[i].propertyFlags&f)==f) return i;
    fprintf(stderr, "No suitable memory type\n"); exit(1);
}

static Buf make_buf(Vk *v, VkDeviceSize sz, VkBufferUsageFlags usage) {
    Buf b = {.sz=sz};
    VkBufferCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size=sz, .usage=usage};
    CHECK_VK(vkCreateBuffer(v->dev, &ci, NULL, &b.buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(v->dev, b.buf, &req);
    VkMemoryAllocateInfo ai = {.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize=req.size,
        .memoryTypeIndex=find_mem(v, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    CHECK_VK(vkAllocateMemory(v->dev, &ai, NULL, &b.mem));
    CHECK_VK(vkBindBufferMemory(v->dev, b.buf, b.mem, 0));
    CHECK_VK(vkMapMemory(v->dev, b.mem, 0, sz, 0, &b.map));
    return b;
}

static void free_buf(Vk *v, Buf *b) {
    vkUnmapMemory(v->dev, b->mem);
    vkDestroyBuffer(v->dev, b->buf, NULL);
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

/* ── Pipeline setup (2 bindings: input + output) ── */
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

    /* 2 bindings: input buffer (0) + output buffer (1) */
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
    VkDescriptorBufferInfo dbis[2] = {
        {in->buf, 0, in->sz},
        {out->buf, 0, out->sz}
    };
    VkWriteDescriptorSet wds[2];
    for(int i=0; i<2; i++) {
        wds[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=p->ds, .dstBinding=i, .descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo=&dbis[i]};
    }
    vkUpdateDescriptorSets(v->dev, 2, wds, 0, NULL);
}

/* ── GPU dispatch with fence timing ── */
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

/* ── CPU cache operations ── */

/* Heat buffer into CPU cache: read every cache line (64B stride on A78) */
static volatile uint64_t g_sink = 0;

static void cpu_heat_cache(void *buf, size_t bytes) {
    volatile uint8_t *p = (volatile uint8_t *)buf;
    uint64_t acc = 0;
    /* Read every 64 bytes (one cache line) to pull into L1/L2 */
    for(size_t i = 0; i < bytes; i += 64) {
        acc += p[i];
    }
    g_sink = acc; /* prevent optimization */
}

/*
 * Evict data from CPU cache:
 * Read a large unrelated buffer (4MB) to push target data out of L2.
 * A78 has 256KB L2 — reading 4MB will thoroughly evict everything.
 */
static volatile uint8_t *g_evict_buf = NULL;
#define EVICT_SIZE (4 * 1024 * 1024)

static void cpu_evict_cache(void) {
    if (!g_evict_buf) {
        g_evict_buf = (volatile uint8_t *)malloc(EVICT_SIZE);
        memset((void*)g_evict_buf, 0xAA, EVICT_SIZE);
    }
    uint64_t acc = 0;
    for(size_t i = 0; i < EVICT_SIZE; i += 64) {
        acc += g_evict_buf[i];
    }
    g_sink = acc;
}

/* ── Concurrent CPU work for contention measurement ── */
typedef struct {
    volatile int start;
    volatile int stop;
    volatile double bandwidth_MBps;
    size_t work_size;
} ContendArgs;

static void *contend_thread(void *arg) {
    ContendArgs *a = (ContendArgs *)arg;
    /* Pin to big core 7 (probe runs on 6) */
    pin(7);

    /* Allocate separate buffer for CPU memcpy work */
    size_t sz = a->work_size;
    void *src = malloc(sz);
    void *dst = malloc(sz);
    memset(src, 0x55, sz);
    memset(dst, 0, sz);

    /* Wait for start signal */
    while(!a->start) { __asm__ volatile("yield"); }

    double t0 = now_us();
    uint64_t total = 0;
    while(!a->stop) {
        memcpy(dst, src, sz);
        total += sz;
    }
    double elapsed = now_us() - t0;
    a->bandwidth_MBps = (double)total / elapsed; /* bytes/us = MB/s */

    free(src);
    free(dst);
    return NULL;
}

/* ── F16 helpers ── */
static uint16_t f32_to_f16(float v) {
    uint32_t x; memcpy(&x, &v, 4);
    uint16_t s = (x >> 16) & 0x8000;
    int e = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7C00;
    return s | (e << 10) | (m >> 13);
}

/* ══════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  ACE-Lite Cache Coherency Probe\n");
    printf("  Can the GPU snoop CPU L2 cache via ACE-Lite?\n");
    printf("══════════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);

    pin(6); /* Host on big core 6 */

    Vk v;
    init_vk(&v);
    printf("  [init] Vulkan ready\n"); fflush(stdout);

    Pipe pipe = make_pipe(&v, "ace_lite_read.spv");
    printf("  [init] Pipeline ready\n\n"); fflush(stdout);

    /* Buffer sizes to test: 64KB, 128KB, 256KB, 512KB, 1MB
     * These span: fits-in-L2, fills-L2, exceeds-L2, way-exceeds-L2 */
    const size_t sizes[] = {
        64  * 1024,   /* 64KB  — fits in L2 (256KB) easily */
        128 * 1024,   /* 128KB — fits in L2 */
        256 * 1024,   /* 256KB — fills entire L2 */
        512 * 1024,   /* 512KB — exceeds L2, spills to DRAM */
        1024 * 1024,  /* 1MB   — way beyond L2 */
    };
    const int NUM_SIZES = sizeof(sizes) / sizeof(sizes[0]);
    const int WARMUP = 3;
    const int ITERS = 15;

    /* Elements per workgroup (each WG has 128 threads reading stride-128) */
    const uint32_t CHUNK_SIZE = 8192; /* 8K f16 elements = 16KB per workgroup */

    printf("  ═══ PHASE 1: CPU-HOT vs COLD GPU reads ═══\n\n");
    printf("  Protocol:\n");
    printf("    COLD: evict CPU cache (read 4MB trash) → GPU reads buffer\n");
    printf("    HOT:  CPU reads entire buffer (pull to L2) → GPU reads immediately\n");
    printf("    If HOT < COLD → ACE-Lite snooping is active!\n\n");
    fflush(stdout);

    for(int si = 0; si < NUM_SIZES; si++) {
        size_t bytes = sizes[si];
        uint32_t num_f16 = (uint32_t)(bytes / 2);
        uint32_t num_wgs = (num_f16 + CHUNK_SIZE - 1) / CHUNK_SIZE;

        printf("  ── Buffer: %zuKB (%u f16 elements, %u workgroups) ──\n",
               bytes/1024, num_f16, num_wgs);
        fflush(stdout);

        /* Allocate Vulkan buffers */
        Buf in_buf = make_buf(&v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buf out_buf = make_buf(&v, num_wgs * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        /* Fill input with data */
        uint16_t *data = (uint16_t *)in_buf.map;
        srand(42);
        for(uint32_t i = 0; i < num_f16; i++)
            data[i] = f32_to_f16(((float)rand()/(float)RAND_MAX - 0.5f) * 2.0f);

        /* Bind descriptors */
        bind_bufs(&v, &pipe, &in_buf, &out_buf);

        uint32_t pc[4] = { num_f16, CHUNK_SIZE, 0, 0 };

        /* ── COLD: evict cache, then GPU reads ── */
        double t_cold[ITERS];
        /* Warmup */
        for(int i = 0; i < WARMUP; i++) {
            cpu_evict_cache();
            fire_gpu(&v, &pipe, pc, num_wgs);
        }
        /* Measure */
        for(int i = 0; i < ITERS; i++) {
            cpu_evict_cache();
            /* Small delay to ensure eviction is complete */
            usleep(100);
            t_cold[i] = fire_gpu(&v, &pipe, pc, num_wgs);
        }
        pstats("COLD (evicted, DRAM path)", t_cold, ITERS);

        /* ── HOT: CPU heats cache, then GPU reads ── */
        double t_hot[ITERS];
        /* Warmup */
        for(int i = 0; i < WARMUP; i++) {
            cpu_heat_cache(in_buf.map, bytes);
            fire_gpu(&v, &pipe, pc, num_wgs);
        }
        /* Measure */
        for(int i = 0; i < ITERS; i++) {
            cpu_heat_cache(in_buf.map, bytes);
            /* NO delay — GPU reads immediately while data is cache-hot */
            t_hot[i] = fire_gpu(&v, &pipe, pc, num_wgs);
        }
        pstats("HOT  (CPU L2 hot, snoop?)", t_hot, ITERS);

        /* ── Stats ── */
        double cold_mean = 0, hot_mean = 0;
        for(int i = 0; i < ITERS; i++) { cold_mean += t_cold[i]; hot_mean += t_hot[i]; }
        cold_mean /= ITERS; hot_mean /= ITERS;

        double ratio = hot_mean / cold_mean;
        double bw_cold = (double)bytes / cold_mean; /* bytes/us = MB/s */
        double bw_hot  = (double)bytes / hot_mean;

        printf("    → COLD mean: %7.1f us (%.1f MB/s)  HOT mean: %7.1f us (%.1f MB/s)\n",
               cold_mean, bw_cold, hot_mean, bw_hot);
        printf("    → HOT/COLD ratio: %.3f", ratio);
        if (ratio < 0.85)
            printf("  ★★ SNOOPING DETECTED — %.0f%% faster!\n", (1.0 - ratio) * 100);
        else if (ratio < 0.95)
            printf("  ★ Possible snooping — %.0f%% faster\n", (1.0 - ratio) * 100);
        else if (ratio > 1.05)
            printf("  ✗ HOT is SLOWER (coherency overhead?)\n");
        else
            printf("  — No difference (both go to DRAM)\n");
        printf("\n");
        fflush(stdout);

        free_buf(&v, &in_buf);
        free_buf(&v, &out_buf);
    }

    /* ═══ PHASE 2: Bus contention — HOT vs COLD with concurrent CPU work ═══ */
    printf("  ═══ PHASE 2: Bus contention with concurrent CPU memcpy ═══\n\n");
    printf("  Protocol:\n");
    printf("    GPU reads 256KB buffer while CPU runs memcpy on separate 1MB buffer\n");
    printf("    Compare: COLD+contend vs HOT+contend vs solo (no contend)\n\n");
    fflush(stdout);

    {
        size_t bytes = 256 * 1024; /* 256KB — the L2 boundary */
        uint32_t num_f16 = (uint32_t)(bytes / 2);
        uint32_t num_wgs = (num_f16 + CHUNK_SIZE - 1) / CHUNK_SIZE;

        Buf in_buf = make_buf(&v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buf out_buf = make_buf(&v, num_wgs * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        uint16_t *data = (uint16_t *)in_buf.map;
        srand(42);
        for(uint32_t i = 0; i < num_f16; i++)
            data[i] = f32_to_f16(((float)rand()/(float)RAND_MAX - 0.5f) * 2.0f);

        bind_bufs(&v, &pipe, &in_buf, &out_buf);
        uint32_t pc[4] = { num_f16, CHUNK_SIZE, 0, 0 };

        double t_solo[ITERS], t_cold_contend[ITERS], t_hot_contend[ITERS];
        double bw_solo_cpu[ITERS], bw_cold_cpu[ITERS], bw_hot_cpu[ITERS];

        /* Solo (no contention, no heating) — baseline */
        for(int i = 0; i < WARMUP; i++) fire_gpu(&v, &pipe, pc, num_wgs);
        for(int i = 0; i < ITERS; i++) t_solo[i] = fire_gpu(&v, &pipe, pc, num_wgs);
        pstats("SOLO (GPU only, no CPU work)", t_solo, ITERS);

        /* COLD + contention */
        for(int i = 0; i < ITERS; i++) {
            cpu_evict_cache();
            usleep(100);

            ContendArgs args = { .start=0, .stop=0, .work_size=1024*1024 };
            pthread_t th;
            pthread_create(&th, NULL, contend_thread, &args);
            usleep(200); /* let contention thread spin up */
            args.start = 1;

            t_cold_contend[i] = fire_gpu(&v, &pipe, pc, num_wgs);
            args.stop = 1;
            pthread_join(th, NULL);
            bw_cold_cpu[i] = args.bandwidth_MBps;
        }
        pstats("COLD + CPU contention", t_cold_contend, ITERS);

        /* HOT + contention */
        for(int i = 0; i < ITERS; i++) {
            cpu_heat_cache(in_buf.map, bytes);

            ContendArgs args = { .start=0, .stop=0, .work_size=1024*1024 };
            pthread_t th;
            pthread_create(&th, NULL, contend_thread, &args);
            usleep(200);
            args.start = 1;

            t_hot_contend[i] = fire_gpu(&v, &pipe, pc, num_wgs);
            args.stop = 1;
            pthread_join(th, NULL);
            bw_hot_cpu[i] = args.bandwidth_MBps;
        }
        pstats("HOT  + CPU contention", t_hot_contend, ITERS);

        /* Summary */
        double solo_mean=0, cold_c_mean=0, hot_c_mean=0;
        double bw_cold_c_mean=0, bw_hot_c_mean=0;
        for(int i=0;i<ITERS;i++) {
            solo_mean += t_solo[i];
            cold_c_mean += t_cold_contend[i];
            hot_c_mean += t_hot_contend[i];
            bw_cold_cpu[i] = bw_cold_cpu[i]; /* already MB/s */
            bw_hot_cpu[i] = bw_hot_cpu[i];
            bw_cold_c_mean += bw_cold_cpu[i];
            bw_hot_c_mean += bw_hot_cpu[i];
        }
        solo_mean /= ITERS;
        cold_c_mean /= ITERS;
        hot_c_mean /= ITERS;
        bw_cold_c_mean /= ITERS;
        bw_hot_c_mean /= ITERS;

        printf("\n    ═══ CONTENTION SUMMARY (256KB buffer) ═══\n\n");
        printf("    GPU solo:            %7.1f us\n", solo_mean);
        printf("    GPU COLD + CPU work: %7.1f us  (%.0f%% slower, CPU BW: %.0f MB/s)\n",
               cold_c_mean, (cold_c_mean/solo_mean - 1)*100, bw_cold_c_mean);
        printf("    GPU HOT  + CPU work: %7.1f us  (%.0f%% slower, CPU BW: %.0f MB/s)\n",
               hot_c_mean, (hot_c_mean/solo_mean - 1)*100, bw_hot_c_mean);
        printf("\n");

        if (hot_c_mean < cold_c_mean * 0.90)
            printf("    ★★ HOT path reduces contention! GPU may be getting data from cache.\n");
        else if (hot_c_mean > cold_c_mean * 1.10)
            printf("    ✗ HOT path INCREASES contention (cache protocol overhead)\n");
        else
            printf("    — HOT and COLD contention are similar\n");

        printf("\n");
        fflush(stdout);

        free_buf(&v, &in_buf);
        free_buf(&v, &out_buf);
    }

    /* ═══ PHASE 3: Repeated rapid HOT reads (can GPU keep data warm?) ═══ */
    printf("  ═══ PHASE 3: Repeated GPU reads (does it get faster on repeat?) ═══\n\n");
    printf("  Protocol:\n");
    printf("    CPU heats 64KB buffer once, then GPU reads it 20 times rapidly\n");
    printf("    If GPU has internal cache or TLB, later reads may be faster\n\n");
    fflush(stdout);

    {
        size_t bytes = 64 * 1024;
        uint32_t num_f16 = (uint32_t)(bytes / 2);
        uint32_t num_wgs = (num_f16 + CHUNK_SIZE - 1) / CHUNK_SIZE;

        Buf in_buf = make_buf(&v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buf out_buf = make_buf(&v, num_wgs * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        uint16_t *data = (uint16_t *)in_buf.map;
        srand(42);
        for(uint32_t i = 0; i < num_f16; i++)
            data[i] = f32_to_f16(((float)rand()/(float)RAND_MAX - 0.5f) * 2.0f);

        bind_bufs(&v, &pipe, &in_buf, &out_buf);
        uint32_t pc[4] = { num_f16, CHUNK_SIZE, 0, 0 };

        /* Warmup */
        for(int i = 0; i < WARMUP; i++) fire_gpu(&v, &pipe, pc, num_wgs);

        /* CPU heats once */
        cpu_heat_cache(in_buf.map, bytes);

        /* 20 rapid consecutive GPU reads */
        int RAPID = 20;
        double t_rapid[20];
        for(int i = 0; i < RAPID; i++) {
            t_rapid[i] = fire_gpu(&v, &pipe, pc, num_wgs);
        }

        printf("    64KB buffer, 20 consecutive GPU reads after CPU heat:\n");
        for(int i = 0; i < RAPID; i++) {
            printf("      read[%2d]: %7.1f us  (%.1f MB/s)\n",
                   i, t_rapid[i], (double)bytes / t_rapid[i]);
        }
        printf("\n");

        double first = t_rapid[0], last_5_mean = 0;
        for(int i = 15; i < 20; i++) last_5_mean += t_rapid[i];
        last_5_mean /= 5;

        printf("    First read: %7.1f us\n", first);
        printf("    Last 5 avg: %7.1f us\n", last_5_mean);
        if (last_5_mean < first * 0.85)
            printf("    ★ GPU warms up — later reads %.0f%% faster (internal caching?)\n",
                   (1.0 - last_5_mean/first) * 100);
        else
            printf("    — No warmup effect (every read goes through same path)\n");

        printf("\n");
        fflush(stdout);

        free_buf(&v, &in_buf);
        free_buf(&v, &out_buf);
    }

    printf("══════════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);

    if (g_evict_buf) free((void*)g_evict_buf);
    return 0;
}
