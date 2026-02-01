#include <android/log.h>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

#include <vulkan/vulkan.h>

#define TAG "InterposerDemo"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

[[noreturn]] void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  __android_log_vprint(ANDROID_LOG_ERROR, TAG, fmt, ap);
  va_end(ap);
  abort();
}

static uint32_t find_memory_type(
    const VkPhysicalDeviceMemoryProperties& mem_props,
    uint32_t type_bits,
    VkMemoryPropertyFlags req) {
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((type_bits & (1u << i)) && ((mem_props.memoryTypes[i].propertyFlags & req) == req)) {
      return i;
    }
  }
  return UINT32_MAX;
}

struct Buffer {
  VkDevice device{};
  VkBuffer buffer{};
  VkDeviceMemory memory{};
  VkDeviceSize size{};
  void* mapped{};
};

static Buffer make_buffer(
    VkPhysicalDevice phys,
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags mem_flags) {
  Buffer b;
  b.device = device;
  b.size = size;

  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(device, &bi, nullptr, &b.buffer) != VK_SUCCESS) {
    die("vkCreateBuffer failed");
  }

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(device, b.buffer, &req);

  VkPhysicalDeviceMemoryProperties mem_props{};
  vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
  uint32_t mem_type = find_memory_type(mem_props, req.memoryTypeBits, mem_flags);
  if (mem_type == UINT32_MAX) {
    die("No suitable memory type (flags=0x%x, typeBits=0x%x)", mem_flags, req.memoryTypeBits);
  }

  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = mem_type;
  if (vkAllocateMemory(device, &ai, nullptr, &b.memory) != VK_SUCCESS) {
    die("vkAllocateMemory failed (size=%" PRIu64 ")", (uint64_t)req.size);
  }

  if (vkBindBufferMemory(device, b.buffer, b.memory, 0) != VK_SUCCESS) {
    die("vkBindBufferMemory failed");
  }

  if (vkMapMemory(device, b.memory, 0, req.size, 0, &b.mapped) != VK_SUCCESS) {
    die("vkMapMemory failed");
  }

  return b;
}

static void destroy_buffer(Buffer& b) {
  if (!b.device) return;
  if (b.mapped) {
    vkUnmapMemory(b.device, b.memory);
    b.mapped = nullptr;
  }
  if (b.buffer) {
    vkDestroyBuffer(b.device, b.buffer, nullptr);
    b.buffer = VK_NULL_HANDLE;
  }
  if (b.memory) {
    vkFreeMemory(b.device, b.memory, nullptr);
    b.memory = VK_NULL_HANDLE;
  }
}

static std::vector<uint32_t> read_spv(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    die("Failed to open SPIR-V: %s (errno=%d)", path, errno);
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0 || (n % 4) != 0) {
    fclose(f);
    die("Invalid SPIR-V size: %ld", n);
  }
  std::vector<uint32_t> words((size_t)n / 4);
  if (fread(words.data(), 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    die("Failed to read SPIR-V (errno=%d)", errno);
  }
  fclose(f);
  return words;
}

} // namespace

int main(int argc, char** argv) {
  // Defaults match a "wave" of N elements.
  const char* spv_path = "/data/local/tmp/interposer_demo.spv";
  const char* mode = "single"; // single | multi_submit | persistent(experimental)
  uint32_t N = 1024;
  uint32_t waves = 16;
  uint32_t sleep_us = 200;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--spv") && i + 1 < argc) {
      spv_path = argv[++i];
    } else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
      mode = argv[++i];
    } else if (!strcmp(argv[i], "--n") && i + 1 < argc) {
      N = (uint32_t)strtoul(argv[++i], nullptr, 10);
    } else if (!strcmp(argv[i], "--waves") && i + 1 < argc) {
      waves = (uint32_t)strtoul(argv[++i], nullptr, 10);
    } else if (!strcmp(argv[i], "--sleep_us") && i + 1 < argc) {
      sleep_us = (uint32_t)strtoul(argv[++i], nullptr, 10);
    } else {
      LOGE("Unknown arg: %s", argv[i]);
      LOGE("Usage: interposer_demo [--mode single|multi_submit|persistent] [--spv /path/to/*.spv] [--n N] [--waves M] [--sleep_us U]");
      return 2;
    }
  }

  LOGI("Neural Interposer demo starting. mode=%s N=%u waves=%u spv=%s", mode, N, waves, spv_path);

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "interposer_demo";
  app.applicationVersion = 1;
  app.pEngineName = "neural_interposer";
  app.engineVersion = 1;
  app.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;

  VkInstance instance{};
  if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
    die("vkCreateInstance failed");
  }

  uint32_t phys_count = 0;
  vkEnumeratePhysicalDevices(instance, &phys_count, nullptr);
  if (phys_count == 0) {
    die("No Vulkan physical devices found");
  }
  std::vector<VkPhysicalDevice> phys(phys_count);
  vkEnumeratePhysicalDevices(instance, &phys_count, phys.data());
  VkPhysicalDevice pd = phys[0];

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(pd, &props);
  LOGI("Using GPU: %s", props.deviceName);

  uint32_t qcount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(qcount);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops.data());

  uint32_t compute_qf = UINT32_MAX;
  for (uint32_t i = 0; i < qcount; i++) {
    if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      compute_qf = i;
      break;
    }
  }
  if (compute_qf == UINT32_MAX) {
    die("No compute queue family found");
  }

  float qprio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = compute_qf;
  qci.queueCount = 1;
  qci.pQueuePriorities = &qprio;

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;

  VkDevice device{};
  if (vkCreateDevice(pd, &dci, nullptr, &device) != VK_SUCCESS) {
    die("vkCreateDevice failed");
  }

  VkQueue queue{};
  vkGetDeviceQueue(device, compute_qf, 0, &queue);

  // Channel buffers (coherent host-visible for Phase 0).
  const VkDeviceSize bytes_f = (VkDeviceSize)N * sizeof(float);
  const VkDeviceSize bytes_sig = 4 * sizeof(uint32_t);

  Buffer in = make_buffer(
      pd,
      device,
      bytes_f,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  Buffer st = make_buffer(
      pd,
      device,
      bytes_f,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  Buffer out = make_buffer(
      pd,
      device,
      bytes_f,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  Buffer sig = make_buffer(
      pd,
      device,
      bytes_sig,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Initialize "channels".
  auto* in_f = reinterpret_cast<float*>(in.mapped);
  auto* st_f = reinterpret_cast<float*>(st.mapped);
  auto* out_f = reinterpret_cast<float*>(out.mapped);
  auto* sig_u = reinterpret_cast<uint32_t*>(sig.mapped);

  for (uint32_t i = 0; i < N; i++) {
    in_f[i] = 1.0f;     // constant wave input
    st_f[i] = 0.0f;     // initial state voltage
    out_f[i] = 0.0f;
  }
  sig_u[0] = 0;     // phase
  sig_u[1] = N;     // element count
  sig_u[2] = 0;     // wave_id
  sig_u[3] = waves; // max_waves (persistent mode)

  // Load SPIR-V and create pipeline.
  std::vector<uint32_t> spv = read_spv(spv_path);

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv.size() * sizeof(uint32_t);
  smci.pCode = spv.data();
  VkShaderModule shader{};
  if (vkCreateShaderModule(device, &smci, nullptr, &shader) != VK_SUCCESS) {
    die("vkCreateShaderModule failed");
  }

  VkDescriptorSetLayoutBinding bindings[4]{};
  for (uint32_t i = 0; i < 4; i++) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dsli.bindingCount = 4;
  dsli.pBindings = bindings;
  VkDescriptorSetLayout dsl{};
  if (vkCreateDescriptorSetLayout(device, &dsli, nullptr, &dsl) != VK_SUCCESS) {
    die("vkCreateDescriptorSetLayout failed");
  }

  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  VkPipelineLayout pl{};
  if (vkCreatePipelineLayout(device, &plci, nullptr, &pl) != VK_SUCCESS) {
    die("vkCreatePipelineLayout failed");
  }

  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.layout = pl;
  cpci.stage = VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = shader;
  cpci.stage.pName = "main";
  VkPipeline pipeline{};
  if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) != VK_SUCCESS) {
    die("vkCreateComputePipelines failed");
  }

  VkDescriptorPoolSize pool_sizes[1]{};
  pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_sizes[0].descriptorCount = 4;
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = pool_sizes;
  VkDescriptorPool pool{};
  if (vkCreateDescriptorPool(device, &dpci, nullptr, &pool) != VK_SUCCESS) {
    die("vkCreateDescriptorPool failed");
  }

  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet ds{};
  if (vkAllocateDescriptorSets(device, &dsai, &ds) != VK_SUCCESS) {
    die("vkAllocateDescriptorSets failed");
  }

  VkDescriptorBufferInfo dbi_in{in.buffer, 0, in.size};
  VkDescriptorBufferInfo dbi_st{st.buffer, 0, st.size};
  VkDescriptorBufferInfo dbi_out{out.buffer, 0, out.size};
  VkDescriptorBufferInfo dbi_sig{sig.buffer, 0, sig.size};

  VkWriteDescriptorSet writes[4]{};
  auto fill_write = [&](uint32_t binding, VkDescriptorBufferInfo* info) {
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = ds;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = info;
    return w;
  };
  writes[0] = fill_write(0, &dbi_in);
  writes[1] = fill_write(1, &dbi_st);
  writes[2] = fill_write(2, &dbi_out);
  writes[3] = fill_write(3, &dbi_sig);
  vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

  VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cp.queueFamilyIndex = compute_qf;
  VkCommandPool cmd_pool{};
  if (vkCreateCommandPool(device, &cp, nullptr, &cmd_pool) != VK_SUCCESS) {
    die("vkCreateCommandPool failed");
  }

  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = cmd_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd{};
  if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
    die("vkAllocateCommandBuffers failed");
  }

  // Record a dispatch. In persistent mode, the shader processes multiple waves and
  // exits when CPU requests stop or wave budget is exhausted.
  VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  if (vkBeginCommandBuffer(cmd, &cbi) != VK_SUCCESS) {
    die("vkBeginCommandBuffer failed");
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
  uint32_t wg = (N + 255u) / 256u;
  vkCmdDispatch(cmd, wg, 1, 1);
  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    die("vkEndCommandBuffer failed");
  }

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence{};
  if (vkCreateFence(device, &fci, nullptr, &fence) != VK_SUCCESS) {
    die("vkCreateFence failed");
  }

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;

  auto host_fence_wait = [&](uint64_t timeout_ns) {
    VkResult r = vkWaitForFences(device, 1, &fence, VK_TRUE, timeout_ns);
    if (r == VK_TIMEOUT) return false;
    if (r != VK_SUCCESS) die("vkWaitForFences failed");
    return true;
  };

  if (!strcmp(mode, "single")) {
    // Kick: set signal[0] = 1 (run wave). Fence signals after wave completes.
    sig_u[0] = 1;
    if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) {
      die("vkQueueSubmit failed (single)");
    }
    if (!host_fence_wait(UINT64_MAX)) {
      die("Unexpected fence timeout (single)");
    }

    // Validate: state should now be all 1.0, output should be all 1.0, and signal[0]=2.
    LOGI("Done. signal[0]=%u", sig_u[0]);
    float max_abs_err = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
      float expected = 1.0f;
      float e0 = fabsf(out_f[i] - expected);
      float e1 = fabsf(st_f[i] - expected);
      if (e0 > max_abs_err) max_abs_err = e0;
      if (e1 > max_abs_err) max_abs_err = e1;
    }
    LOGI("max_abs_err=%f (expected 0.0)", max_abs_err);
  } else if (!strcmp(mode, "multi_submit")) {
    // Multi-wave scheduling via repeated submits (stable baseline).
    // This keeps "state as a channel voltage" (state buffer persists across waves),
    // while avoiding long-running/persistent dispatch issues.

    // Ensure we start from idle.
    sig_u[0] = 0;

    for (uint32_t w = 0; w < waves; w++) {
      // Deterministic input wave: all ones.
      for (uint32_t i = 0; i < N; i++) {
        in_f[i] = 1.0f;
      }

      // Request execution for this wave.
      sig_u[0] = 1;

      if (vkResetFences(device, 1, &fence) != VK_SUCCESS) {
        die("vkResetFences failed");
      }

      if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) {
        die("vkQueueSubmit failed (multi_submit)");
      }

      if (!host_fence_wait(UINT64_MAX)) {
        die("Unexpected fence timeout (multi_submit)");
      }

      // Validate a handful of elements.
      float expected = (float)(w + 1);
      float max_abs_err = 0.0f;
      for (uint32_t i = 0; i < (N < 64u ? N : 64u); i++) {
        float e0 = fabsf(out_f[i] - expected);
        float e1 = fabsf(st_f[i] - expected);
        if (e0 > max_abs_err) max_abs_err = e0;
        if (e1 > max_abs_err) max_abs_err = e1;
      }
      LOGI("Wave %u done. expected=%f phase=%u max_abs_err(first64)=%f", w, expected, sig_u[0], max_abs_err);

      // Reset to idle for next wave.
      sig_u[0] = 0;
    }
  } else if (!strcmp(mode, "persistent")) {
    LOGE("WARNING: persistent mode is experimental on this GPU/driver.");
    if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) {
      die("vkQueueSubmit failed (persistent)");
    }
    // Multi-wave host-driven protocol:
    //   - CPU sets wave_id=i, fills input, sets phase=1
    //   - GPU sets phase=2 when done
    //   - CPU reads output/state, sets phase=0 (ack)
    //   - Repeat; finally set phase=3 (stop) and wait fence.
    for (uint32_t w = 0; w < waves; w++) {
      // Create a deterministic wave: input is all 1.0 every time.
      // Expected: state becomes (w+1) after wave w.
      for (uint32_t i = 0; i < N; i++) {
        in_f[i] = 1.0f;
      }

      // Publish wave id and request execution.
      sig_u[2] = w;
      std::atomic_thread_fence(std::memory_order_seq_cst);
      sig_u[0] = 1;

      // Wait for GPU completion (phase=2), with timeout guard.
      const uint32_t max_spins = 20000; // ~max_spins*sleep_us microseconds
      uint32_t spins = 0;
      while (sig_u[0] != 2) {
        if (spins++ >= max_spins) {
          LOGE("Timeout waiting for wave %u (phase=%u). Requesting stop.", w, sig_u[0]);
          sig_u[0] = 3;
          break;
        }
        if (sleep_us) usleep(sleep_us);
      }
      if (sig_u[0] == 3) {
        break;
      }

      // Validate a handful of elements for speed.
      float expected = (float)(w + 1);
      float max_abs_err = 0.0f;
      for (uint32_t i = 0; i < (N < 64u ? N : 64u); i++) {
        float e0 = fabsf(out_f[i] - expected);
        float e1 = fabsf(st_f[i] - expected);
        if (e0 > max_abs_err) max_abs_err = e0;
        if (e1 > max_abs_err) max_abs_err = e1;
      }
      LOGI("Wave %u done. expected=%f phase=%u max_abs_err(first64)=%f", w, expected, sig_u[0], max_abs_err);

      // Ack so GPU can proceed to next wave.
      sig_u[0] = 0;
    }

    // Request stop so the GPU exits its bounded loop, then wait for fence.
    sig_u[0] = 3;
    // 5s should be plenty for cleanup on this tiny shader.
    if (!host_fence_wait(5ull * 1000ull * 1000ull * 1000ull)) {
      die("Fence timeout waiting for persistent shader to exit");
    }
    LOGI("Persistent mode finished (stop acknowledged by fence).");
  } else {
    die("Unknown mode: %s", mode);
  }

  // Cleanup.
  vkDestroyFence(device, fence, nullptr);
  vkDestroyCommandPool(device, cmd_pool, nullptr);
  vkDestroyDescriptorPool(device, pool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pl, nullptr);
  vkDestroyDescriptorSetLayout(device, dsl, nullptr);
  vkDestroyShaderModule(device, shader, nullptr);

  destroy_buffer(in);
  destroy_buffer(st);
  destroy_buffer(out);
  destroy_buffer(sig);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);

  LOGI("Neural Interposer demo finished.");
  return 0;
}

