/*
 * gpu_cached_sweep.c — HOST_CACHED vs HOST_COHERENT bandwidth sweep
 *
 * The hierarchy probe found HOST_CACHED is 2.5x faster at 256KB.
 * This probe sweeps buffer sizes with BOTH memory types to map
 * where the SLC boundary falls for each.
 *
 * Build:
 *   $CC -O2 -march=armv8.2-a+dotprod+fp16 -o gpu_cached_sweep gpu_cached_sweep.c -lvulkan -lm
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

typedef struct {
    VkInstance inst; VkPhysicalDevice pdev; VkDevice dev;
    uint32_t qf; VkQueue queue; VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties mprops;
} Vk;
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; VkDeviceSize sz; } Buf;
typedef struct {
    VkPipeline pipe; VkPipelineLayout layout;
    VkDescriptorSet ds; VkDescriptorSetLayout dsl; VkDescriptorPool dp;
} Pipe;

static uint32_t find_mem_type(Vk *v, uint32_t bits, VkMemoryPropertyFlags f) {
    for(uint32_t i=0; i<v->mprops.memoryTypeCount; i++)
        if((bits&(1u<<i)) && (v->mprops.memoryTypes[i].propertyFlags&f)==f) return i;
    return UINT32_MAX;
}

static Buf make_buf_flags(Vk *v, VkDeviceSize sz, VkBufferUsageFlags usage, VkMemoryPropertyFlags mflags) {
    Buf b = {.sz=sz};
    VkBufferCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size=sz, .usage=usage};
    CHECK_VK(vkCreateBuffer(v->dev, &ci, NULL, &b.buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(v->dev, b.buf, &req);
    uint32_t mt = find_mem_type(v, req.memoryTypeBits, mflags);
    if (mt == UINT32_MAX) { fprintf(stderr, "No mem type for flags 0x%x\n", mflags); exit(1); }
    VkMemoryAllocateInfo ai = {.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize=req.size, .memoryTypeIndex=mt};
    CHECK_VK(vkAllocateMemory(v->dev, &ai, NULL, &b.mem));
    CHECK_VK(vkBindBufferMemory(v->dev, b.buf, b.mem, 0));
    CHECK_VK(vkMapMemory(v->dev, b.mem, 0, sz, 0, &b.map));
    return b;
}

static void free_buf(Vk *v, Buf *b) {
    if(b->map) vkUnmapMemory(v->dev, b->mem);
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

static Pipe make_pipe(Vk *v, const char *spv) {
    Pipe p;
    FILE *f = fopen(spv, "rb");
    if(!f) { fprintf(stderr, "Cannot open %s\n", spv); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc(sz); fread(code, 1, sz, f); fclose(f);
    VkShaderModule sm;
    VkShaderModuleCreateInfo smci = {.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize=sz, .pCode=code};
    CHECK_VK(vkCreateShaderModule(v->dev, &smci, NULL, &sm)); free(code);
    VkDescriptorSetLayoutBinding binds[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dslci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount=2, .pBindings=binds};
    CHECK_VK(vkCreateDescriptorSetLayout(v->dev, &dslci, NULL, &p.dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo plci = {.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1, .pSetLayouts=&p.dsl, .pushConstantRangeCount=1, .pPushConstantRanges=&pcr};
    CHECK_VK(vkCreatePipelineLayout(v->dev, &plci, NULL, &p.layout));
    VkComputePipelineCreateInfo cpci = {.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage=VK_SHADER_STAGE_COMPUTE_BIT, .module=sm, .pName="main"}, .layout=p.layout};
    CHECK_VK(vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &p.pipe));
    vkDestroyShaderModule(v->dev, sm, NULL);
    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets=1, .poolSizeCount=1, .pPoolSizes=&dps};
    CHECK_VK(vkCreateDescriptorPool(v->dev, &dpci, NULL, &p.dp));
    VkDescriptorSetAllocateInfo dsai = {.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool=p.dp, .descriptorSetCount=1, .pSetLayouts=&p.dsl};
    CHECK_VK(vkAllocateDescriptorSets(v->dev, &dsai, &p.ds));
    return p;
}

static void bind_bufs(Vk *v, Pipe *p, Buf *in, Buf *out) {
    VkDescriptorBufferInfo dbis[2] = {{in->buf, 0, in->sz}, {out->buf, 0, out->sz}};
    VkWriteDescriptorSet wds[2];
    for(int i=0;i<2;i++)
        wds[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=p->ds,.dstBinding=i,.descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbis[i]};
    vkUpdateDescriptorSets(v->dev, 2, wds, 0, NULL);
}

static double fire_multi(Vk *v, Pipe *p, uint32_t pc[4], uint32_t nwg, int reps) {
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
    for(int r=0; r<reps; r++) {
        vkCmdDispatch(cmd, nwg, 1, 1);
        if(r<reps-1) {
            VkMemoryBarrier mb = {.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        }
    }
    CHECK_VK(vkEndCommandBuffer(cmd));
    VkFence fence; VkFenceCreateInfo fci = {.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
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

int main(void) {
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  GPU Bandwidth: HOST_CACHED vs HOST_COHERENT\n");
    printf("  10 reads/cmdbuf, amortized dispatch overhead\n");
    printf("══════════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);

    pin(6);
    Vk v; init_vk(&v);
    Pipe pipe = make_pipe(&v, "ace_lite_read.spv");
    printf("  [init] Ready\n\n");

    const uint32_t CHUNK = 8192;
    const int REPS = 10, WARMUP = 3, ITERS = 10;

    size_t sizes[] = {
        32*1024, 64*1024, 96*1024, 128*1024, 160*1024, 192*1024,
        224*1024, 256*1024, 320*1024, 384*1024, 448*1024, 512*1024,
        640*1024, 768*1024, 1024*1024, 1536*1024, 2*1024*1024,
        3*1024*1024, 4*1024*1024
    };
    int N = sizeof(sizes)/sizeof(sizes[0]);

    VkMemoryPropertyFlags coherent = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkMemoryPropertyFlags cached = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    printf("  %8s  %12s  %12s  %8s\n", "Size", "COHERENT", "CACHED", "Ratio");
    printf("  %8s  %12s  %12s  %8s\n", "", "(MB/s)", "(MB/s)", "C/H");
    printf("  %8s  %12s  %12s  %8s\n", "--------", "------------", "------------", "--------");
    fflush(stdout);

    for(int si = 0; si < N; si++) {
        size_t bytes = sizes[si];
        uint32_t nf16 = (uint32_t)(bytes / 2);
        uint32_t nwg = (nf16 + CHUNK - 1) / CHUNK;
        if(nwg == 0) nwg = 1;
        uint32_t pc[4] = { nf16, CHUNK, 0, 0 };

        /* Allocate output buffer once (small, shared) */
        Buf out = make_buf_flags(&v, nwg * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, coherent);

        double bw_coh = 0, bw_cac = 0;

        /* Test COHERENT */
        {
            Buf in = make_buf_flags(&v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, coherent);
            uint16_t *d = (uint16_t*)in.map;
            for(uint32_t i=0; i<nf16; i++) d[i] = f32_to_f16((float)(i%1000)/1000.0f);
            bind_bufs(&v, &pipe, &in, &out);
            for(int i=0;i<WARMUP;i++) fire_multi(&v, &pipe, pc, nwg, REPS);
            double sum=0;
            for(int i=0;i<ITERS;i++) {
                double t = fire_multi(&v, &pipe, pc, nwg, REPS);
                sum += t / REPS;
            }
            double per_read = sum / ITERS;
            bw_coh = (double)bytes / per_read;
            free_buf(&v, &in);
        }

        /* Test CACHED */
        {
            Buf in = make_buf_flags(&v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, cached);
            uint16_t *d = (uint16_t*)in.map;
            for(uint32_t i=0; i<nf16; i++) d[i] = f32_to_f16((float)(i%1000)/1000.0f);
            bind_bufs(&v, &pipe, &in, &out);
            for(int i=0;i<WARMUP;i++) fire_multi(&v, &pipe, pc, nwg, REPS);
            double sum=0;
            for(int i=0;i<ITERS;i++) {
                double t = fire_multi(&v, &pipe, pc, nwg, REPS);
                sum += t / REPS;
            }
            double per_read = sum / ITERS;
            bw_cac = (double)bytes / per_read;
            free_buf(&v, &in);
        }

        const char *unit = "KB";
        double disp = bytes / 1024.0;
        if(bytes >= 1024*1024) { unit = "MB"; disp = bytes/(1024.0*1024.0); }

        double ratio = bw_cac / bw_coh;
        const char *marker = "";
        if (ratio > 1.5) marker = " ★★";
        else if (ratio > 1.2) marker = " ★";

        printf("  %5.0f%s  %10.0f    %10.0f    %6.2fx%s\n",
               disp, unit, bw_coh, bw_cac, ratio, marker);
        fflush(stdout);

        free_buf(&v, &out);
    }

    printf("\n══════════════════════════════════════════════════════════════════\n\n");
    return 0;
}
