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

struct Pipeline {
  VkDevice device{};
  VkPipelineLayout layout{};
  VkPipeline pipeline{};
  VkDescriptorSetLayout dsl{};
  VkDescriptorPool pool{};
  VkDescriptorSet ds{};
  VkShaderModule shader{};
  uint32_t binding_count{};
  uint32_t push_const_bytes{};
};

static void destroy_pipeline(Pipeline& p) {
  if (!p.device) return;
  if (p.pool) vkDestroyDescriptorPool(p.device, p.pool, nullptr);
  if (p.pipeline) vkDestroyPipeline(p.device, p.pipeline, nullptr);
  if (p.layout) vkDestroyPipelineLayout(p.device, p.layout, nullptr);
  if (p.dsl) vkDestroyDescriptorSetLayout(p.device, p.dsl, nullptr);
  if (p.shader) vkDestroyShaderModule(p.device, p.shader, nullptr);
  p = {};
}

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

static Pipeline make_compute_pipeline(
    VkPhysicalDevice /*pd*/,
    VkDevice device,
    const char* spv_path,
    uint32_t binding_count,
    uint32_t push_const_bytes) {
  Pipeline p;
  p.device = device;
  p.binding_count = binding_count;
  p.push_const_bytes = push_const_bytes;

  std::vector<uint32_t> spv = read_spv(spv_path);
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv.size() * sizeof(uint32_t);
  smci.pCode = spv.data();
  if (vkCreateShaderModule(device, &smci, nullptr, &p.shader) != VK_SUCCESS) {
    die("vkCreateShaderModule failed (%s)", spv_path);
  }

  std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
  for (uint32_t i = 0; i < binding_count; i++) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dsli.bindingCount = (uint32_t)bindings.size();
  dsli.pBindings = bindings.data();
  if (vkCreateDescriptorSetLayout(device, &dsli, nullptr, &p.dsl) != VK_SUCCESS) {
    die("vkCreateDescriptorSetLayout failed");
  }

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = push_const_bytes;

  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &p.dsl;
  if (push_const_bytes) {
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
  }
  if (vkCreatePipelineLayout(device, &plci, nullptr, &p.layout) != VK_SUCCESS) {
    die("vkCreatePipelineLayout failed");
  }

  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.layout = p.layout;
  cpci.stage = VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = p.shader;
  cpci.stage.pName = "main";
  if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline) != VK_SUCCESS) {
    die("vkCreateComputePipelines failed");
  }

  VkDescriptorPoolSize pool_sizes[1]{};
  pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_sizes[0].descriptorCount = binding_count;
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = pool_sizes;
  if (vkCreateDescriptorPool(device, &dpci, nullptr, &p.pool) != VK_SUCCESS) {
    die("vkCreateDescriptorPool failed");
  }

  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = p.pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &p.dsl;
  if (vkAllocateDescriptorSets(device, &dsai, &p.ds) != VK_SUCCESS) {
    die("vkAllocateDescriptorSets failed");
  }

  return p;
}

static void update_desc_set(VkDevice device, VkDescriptorSet ds, const std::vector<VkDescriptorBufferInfo>& infos) {
  std::vector<VkWriteDescriptorSet> writes(infos.size());
  for (uint32_t i = 0; i < (uint32_t)infos.size(); i++) {
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = ds;
    w.dstBinding = i;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &infos[i];
    writes[i] = w;
  }
  vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}

static std::vector<uint8_t> read_file_bytes(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    die("Failed to open file: %s (errno=%d)", path, errno);
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n < 0) {
    fclose(f);
    die("ftell failed for %s", path);
  }
  std::vector<uint8_t> buf((size_t)n);
  if (n && fread(buf.data(), 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    die("Failed to read file: %s (errno=%d)", path, errno);
  }
  fclose(f);
  return buf;
}

static void load_f32_into(Buffer& b, const char* path, size_t expected_bytes) {
  std::vector<uint8_t> bytes = read_file_bytes(path);
  if (expected_bytes && bytes.size() != expected_bytes) {
    die("Size mismatch for %s: got=%zu expected=%zu", path, bytes.size(), expected_bytes);
  }
  if (bytes.size() > (size_t)b.size) {
    die("Buffer too small for %s: buf=%" PRIu64 " file=%zu", path, (uint64_t)b.size, bytes.size());
  }
  memcpy(b.mapped, bytes.data(), bytes.size());
}

} // namespace

int main(int argc, char** argv) {
  // Defaults match a "wave" of N elements.
  const char* spv_path = "/data/local/tmp/interposer_demo.spv";
  const char* spv_path_2 = nullptr; // optional second shader (shortconv_block)
  const char* chip = "add";   // add | shortconv
  const char* mode = "single"; // single | multi_submit | persistent(experimental)
  uint32_t N = 1024;
  uint32_t D = 1024; // ShortConv channels (dim)
  uint32_t waves = 16;
  uint32_t sleep_us = 200;
  const char* weights_dir = nullptr; // for lfm2_shortconv

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--spv") && i + 1 < argc) {
      spv_path = argv[++i];
    } else if (!strcmp(argv[i], "--spv2") && i + 1 < argc) {
      spv_path_2 = argv[++i];
    } else if (!strcmp(argv[i], "--chip") && i + 1 < argc) {
      chip = argv[++i];
    } else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
      mode = argv[++i];
    } else if (!strcmp(argv[i], "--n") && i + 1 < argc) {
      N = (uint32_t)strtoul(argv[++i], nullptr, 10);
    } else if (!strcmp(argv[i], "--d") && i + 1 < argc) {
      D = (uint32_t)strtoul(argv[++i], nullptr, 10);
    } else if (!strcmp(argv[i], "--waves") && i + 1 < argc) {
      waves = (uint32_t)strtoul(argv[++i], nullptr, 10);
    } else if (!strcmp(argv[i], "--sleep_us") && i + 1 < argc) {
      sleep_us = (uint32_t)strtoul(argv[++i], nullptr, 10);
    } else if (!strcmp(argv[i], "--weights_dir") && i + 1 < argc) {
      weights_dir = argv[++i];
    } else if (!strcmp(argv[i], "--suite_root") && i + 1 < argc) {
      // Special helper: run lfm2_shortconv against multiple layer directories.
      // We keep the same shaders and just swap weights_dir between runs.
      weights_dir = argv[++i];
    } else {
      LOGE("Unknown arg: %s", argv[i]);
      LOGE("Usage:");
      LOGE("  interposer_demo --chip add --mode single|multi_submit|persistent --spv /path/to/interposer_demo*.spv --n N --waves M");
      LOGE("  interposer_demo --chip shortconv --spv /path/to/shortconv_chip.spv --d D --waves M");
      LOGE("  interposer_demo --chip lfm2_shortconv --spv /path/to/shortconv_pre.spv --spv2 /path/to/matvec_out.spv --d D --waves M --weights_dir /data/local/tmp/lfm2_sc");
      return 2;
    }
  }

  LOGI("Neural Interposer demo starting. chip=%s mode=%s N=%u D=%u waves=%u spv=%s", chip, mode, N, D, waves, spv_path);

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

  // We use host-visible coherent buffers for early prototyping.
  // The "add" chip uses:
  //   in[N], state[N], out[N], signal[4]
  // The "shortconv" chip uses:
  //   bx[D], c[D], state_in[D*2], y[D], state_out[D*2], weights[D*3]
  const VkMemoryPropertyFlags host_coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  const bool is_shortconv = !strcmp(chip, "shortconv");
  const bool is_shortconv_block = !strcmp(chip, "shortconv_block");
  const bool is_lfm2_shortconv = !strcmp(chip, "lfm2_shortconv");
  const bool is_add_chip = !(is_shortconv || is_shortconv_block || is_lfm2_shortconv);

  Pipeline pipe1;
  Pipeline pipe2;
  if (is_shortconv_block || is_lfm2_shortconv) {
    if (!spv_path_2) {
      die("shortconv_block requires --spv and --spv2");
    }
    pipe1 = make_compute_pipeline(pd, device, spv_path, /*bindings*/ 8, /*push*/ 4);
    pipe2 = make_compute_pipeline(pd, device, spv_path_2, /*bindings*/ 3, /*push*/ 4);
  } else if (is_shortconv) {
    pipe1 = make_compute_pipeline(pd, device, spv_path, /*bindings*/ 6, /*push*/ 8);
  } else {
    pipe1 = make_compute_pipeline(pd, device, spv_path, /*bindings*/ 4, /*push*/ 0);
  }

  // Allocate buffers + write descriptors per chip.
  Buffer in{}, st{}, out{}, sig{};
  Buffer bx{}, cbuf{}, state0{}, state1{}, ybuf{}, wbuf{};
  Buffer xvec{}, ypre{}, outproj{};
  Buffer Wb{}, Wc{}, Wx{}, Wout{};
  float* in_f = nullptr;
  float* st_f = nullptr;
  float* out_f = nullptr;
  uint32_t* sig_u = nullptr;

  if (is_add_chip) {
    const VkDeviceSize bytes_f = (VkDeviceSize)N * sizeof(float);
    const VkDeviceSize bytes_sig = 4 * sizeof(uint32_t);

    in = make_buffer(pd, device, bytes_f, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    st = make_buffer(pd, device, bytes_f, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    out = make_buffer(pd, device, bytes_f, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    sig = make_buffer(pd, device, bytes_sig, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);

    in_f = reinterpret_cast<float*>(in.mapped);
    st_f = reinterpret_cast<float*>(st.mapped);
    out_f = reinterpret_cast<float*>(out.mapped);
    sig_u = reinterpret_cast<uint32_t*>(sig.mapped);

    for (uint32_t i = 0; i < N; i++) {
      in_f[i] = 1.0f;
      st_f[i] = 0.0f;
      out_f[i] = 0.0f;
    }
    sig_u[0] = 0;
    sig_u[1] = N;
    sig_u[2] = 0;
    sig_u[3] = waves;

    std::vector<VkDescriptorBufferInfo> infos{
        {in.buffer, 0, in.size},
        {st.buffer, 0, st.size},
        {out.buffer, 0, out.size},
        {sig.buffer, 0, sig.size},
    };
    update_desc_set(device, pipe1.ds, infos);
  } else {
    // ShortConv (L=3, state length L-1 = 2).
    const uint32_t L = 3;
    const uint32_t S = L - 1;
    const VkDeviceSize bytes_D = (VkDeviceSize)D * sizeof(float);
    const VkDeviceSize bytes_state = (VkDeviceSize)D * S * sizeof(float);
    const VkDeviceSize bytes_w = (VkDeviceSize)D * L * sizeof(float);

    bx = make_buffer(pd, device, bytes_D, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    cbuf = make_buffer(pd, device, bytes_D, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    state0 = make_buffer(pd, device, bytes_state, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    state1 = make_buffer(pd, device, bytes_state, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    ybuf = make_buffer(pd, device, bytes_D, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    wbuf = make_buffer(pd, device, bytes_w, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);

    auto* bx_f = reinterpret_cast<float*>(bx.mapped);
    auto* c_f = reinterpret_cast<float*>(cbuf.mapped);
    auto* s0_f = reinterpret_cast<float*>(state0.mapped);
    auto* s1_f = reinterpret_cast<float*>(state1.mapped);
    auto* y_f = reinterpret_cast<float*>(ybuf.mapped);
    auto* w_f = reinterpret_cast<float*>(wbuf.mapped);

    for (uint32_t i = 0; i < D; i++) {
      bx_f[i] = 1.0f;
      c_f[i] = 1.0f;
      y_f[i] = 0.0f;
      s0_f[i * S + 0] = 0.0f;
      s0_f[i * S + 1] = 0.0f;
      s1_f[i * S + 0] = 0.0f;
      s1_f[i * S + 1] = 0.0f;

      // Simple deterministic weights: [0.25, 0.5, 1.0]
      w_f[i * L + 0] = 0.25f;
      w_f[i * L + 1] = 0.50f;
      w_f[i * L + 2] = 1.00f;
    }

    std::vector<VkDescriptorBufferInfo> infos{
        {bx.buffer, 0, bx.size},         // 0
        {cbuf.buffer, 0, cbuf.size},     // 1
        {state0.buffer, 0, state0.size}, // 2 (updated per-wave)
        {ybuf.buffer, 0, ybuf.size},     // 3
        {state1.buffer, 0, state1.size}, // 4 (updated per-wave)
        {wbuf.buffer, 0, wbuf.size},     // 5
    };
    update_desc_set(device, pipe1.ds, infos);
  }

  if (is_shortconv_block || is_lfm2_shortconv) {
    // Allocate x, y_pre, out and 4 projection matrices (row-major) + depthwise weights.
    const VkDeviceSize bytes_D = (VkDeviceSize)D * sizeof(float);
    const VkDeviceSize bytes_state = (VkDeviceSize)D * 2 * sizeof(float);
    const VkDeviceSize bytes_mat = (VkDeviceSize)D * D * sizeof(float);
    const VkDeviceSize bytes_wdw = (VkDeviceSize)D * 3 * sizeof(float);

    xvec = make_buffer(pd, device, bytes_D, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    ypre = make_buffer(pd, device, bytes_D, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    outproj = make_buffer(pd, device, bytes_D, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    state0 = make_buffer(pd, device, bytes_state, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    state1 = make_buffer(pd, device, bytes_state, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    wbuf = make_buffer(pd, device, bytes_wdw, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    Wb = make_buffer(pd, device, bytes_mat, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    Wc = make_buffer(pd, device, bytes_mat, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    Wx = make_buffer(pd, device, bytes_mat, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);
    Wout = make_buffer(pd, device, bytes_mat, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_coherent);

    auto* x_f = reinterpret_cast<float*>(xvec.mapped);
    auto* y_f = reinterpret_cast<float*>(ypre.mapped);
    auto* o_f = reinterpret_cast<float*>(outproj.mapped);
    auto* s0_f = reinterpret_cast<float*>(state0.mapped);
    auto* s1_f = reinterpret_cast<float*>(state1.mapped);
    auto* wdw_f = reinterpret_cast<float*>(wbuf.mapped);
    auto* Wb_f = reinterpret_cast<float*>(Wb.mapped);
    auto* Wc_f = reinterpret_cast<float*>(Wc.mapped);
    auto* Wx_f = reinterpret_cast<float*>(Wx.mapped);
    auto* Wo_f = reinterpret_cast<float*>(Wout.mapped);

    // Default deterministic, bounded weights for a reproducible test:
    // - x is a simple ramp
    // - matrices are identity
    // - depthwise weights fixed
    for (uint32_t i = 0; i < D; i++) {
      x_f[i] = (float)(i % 17) / 17.0f;
      y_f[i] = 0.0f;
      o_f[i] = 0.0f;
      s0_f[i * 2 + 0] = 0.0f;
      s0_f[i * 2 + 1] = 0.0f;
      s1_f[i * 2 + 0] = 0.0f;
      s1_f[i * 2 + 1] = 0.0f;
      wdw_f[i * 3 + 0] = 0.25f;
      wdw_f[i * 3 + 1] = 0.50f;
      wdw_f[i * 3 + 2] = 1.00f;
    }
    for (uint32_t r = 0; r < D; r++) {
      for (uint32_t c = 0; c < D; c++) {
        float v = (r == c) ? 1.0f : 0.0f;
        Wb_f[r * D + c] = v;
        Wc_f[r * D + c] = v;
        Wx_f[r * D + c] = v;
        Wo_f[r * D + c] = v;
      }
    }

    if (is_lfm2_shortconv) {
      if (!weights_dir) {
        die("lfm2_shortconv requires --weights_dir");
      }
      // Load real weights and input/state from files produced by export_shortconv_layer_bins.py
      std::string dir(weights_dir);
      auto join = [&](const char* name) {
        return (dir.back() == '/' ? dir + name : dir + "/" + name);
      };
      load_f32_into(xvec, join("x_norm.bin").c_str(), (size_t)bytes_D);
      load_f32_into(state0, join("state0.bin").c_str(), (size_t)bytes_state);
      load_f32_into(wbuf, join("Wdw.bin").c_str(), (size_t)bytes_wdw);
      load_f32_into(Wb, join("Wb.bin").c_str(), (size_t)bytes_mat);
      load_f32_into(Wc, join("Wc.bin").c_str(), (size_t)bytes_mat);
      load_f32_into(Wx, join("Wx.bin").c_str(), (size_t)bytes_mat);
      load_f32_into(Wout, join("Wout.bin").c_str(), (size_t)bytes_mat);
    }

    // Pipeline 1: shortconv_pre
    // 0 x, 1 state_in, 2 y, 3 state_out, 4 Wb, 5 Wc, 6 Wx, 7 Wdw
    std::vector<VkDescriptorBufferInfo> infos1{
        {xvec.buffer, 0, xvec.size},
        {state0.buffer, 0, state0.size}, // updated per-wave
        {ypre.buffer, 0, ypre.size},
        {state1.buffer, 0, state1.size}, // updated per-wave
        {Wb.buffer, 0, Wb.size},
        {Wc.buffer, 0, Wc.size},
        {Wx.buffer, 0, Wx.size},
        {wbuf.buffer, 0, wbuf.size},
    };
    update_desc_set(device, pipe1.ds, infos1);

    // Pipeline 2: matvec_out
    // 0 y, 1 out, 2 Wout
    std::vector<VkDescriptorBufferInfo> infos2{
        {ypre.buffer, 0, ypre.size},
        {outproj.buffer, 0, outproj.size},
        {Wout.buffer, 0, Wout.size},
    };
    update_desc_set(device, pipe2.ds, infos2);
  }

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

  auto record_dispatch = [&](VkPipelineLayout layout, VkPipeline pipeline, VkDescriptorSet ds, uint32_t work_items, const void* pc_data, uint32_t pc_bytes) {
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (vkBeginCommandBuffer(cmd, &cbi) != VK_SUCCESS) {
      die("vkBeginCommandBuffer failed");
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
    if (pc_bytes) {
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc_data);
    }
    uint32_t wg = (work_items + 255u) / 256u;
    vkCmdDispatch(cmd, wg, 1, 1);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
      die("vkEndCommandBuffer failed");
    }
  };

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

  if (is_shortconv_block || is_lfm2_shortconv) {
    // run below
  } else if (is_add_chip && !strcmp(mode, "single")) {
    // Kick: set signal[0] = 1 (run wave). Fence signals after wave completes.
    if (!sig_u || !out_f || !st_f) {
      die("Internal error: add-chip buffers not initialized");
    }
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
  } else if (is_add_chip && !strcmp(mode, "multi_submit")) {
    // Multi-wave scheduling via repeated submits (stable baseline).
    // This keeps "state as a channel voltage" (state buffer persists across waves),
    // while avoiding long-running/persistent dispatch issues.

    // Ensure we start from idle.
    if (!sig_u || !in_f || !out_f || !st_f) {
      die("Internal error: add-chip buffers not initialized");
    }
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
  } else if (is_add_chip && !strcmp(mode, "persistent")) {
    LOGE("WARNING: persistent mode is experimental on this GPU/driver.");
    if (!sig_u || !in_f || !out_f || !st_f) {
      die("Internal error: add-chip buffers not initialized");
    }
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
  } else if (is_add_chip) {
    die("Unknown mode: %s", mode);
  }

  if (is_lfm2_shortconv) {
    // Compare GPU results to PyTorch-exported expected tensors.
    if (!weights_dir) {
      die("lfm2_shortconv requires --weights_dir");
    }
    std::string dir(weights_dir);
    auto join = [&](const char* name) {
      return (dir.back() == '/' ? dir + name : dir + "/" + name);
    };

    // Load expected arrays (waves, D) and (waves, D, 2).
    const size_t bytes_exp_y = (size_t)waves * (size_t)D * sizeof(float);
    const size_t bytes_exp_state = (size_t)waves * (size_t)D * 2u * sizeof(float);

    std::vector<uint8_t> exp_y_pre_bytes = read_file_bytes(join("expected_y_pre.bin").c_str());
    std::vector<uint8_t> exp_out_bytes = read_file_bytes(join("expected_out.bin").c_str());
    std::vector<uint8_t> exp_state_bytes = read_file_bytes(join("expected_state.bin").c_str());
    if (exp_y_pre_bytes.size() < bytes_exp_y || exp_out_bytes.size() < bytes_exp_y || exp_state_bytes.size() < bytes_exp_state) {
      die("Expected files are smaller than requested waves/D (check --waves/--d match exporter)");
    }
    const float* exp_y_pre = reinterpret_cast<const float*>(exp_y_pre_bytes.data());
    const float* exp_out = reinterpret_cast<const float*>(exp_out_bytes.data());
    const float* exp_state = reinterpret_cast<const float*>(exp_state_bytes.data());

    auto* y_f = reinterpret_cast<float*>(ypre.mapped);
    auto* o_f = reinterpret_cast<float*>(outproj.mapped);
    auto* s0_f = reinterpret_cast<float*>(state0.mapped);
    auto* s1_f = reinterpret_cast<float*>(state1.mapped);

    for (uint32_t w = 0; w < waves; w++) {
      const bool ping = (w % 2) == 0;
      float* state_out = ping ? s1_f : s0_f;

      // Update state bindings for pre-pass: binding1=state_in, binding3=state_out
      std::vector<VkDescriptorBufferInfo> infos1{
          {xvec.buffer, 0, xvec.size},
          {(ping ? state0.buffer : state1.buffer), 0, state0.size},
          {ypre.buffer, 0, ypre.size},
          {(ping ? state1.buffer : state0.buffer), 0, state1.size},
          {Wb.buffer, 0, Wb.size},
          {Wc.buffer, 0, Wc.size},
          {Wx.buffer, 0, Wx.size},
          {wbuf.buffer, 0, wbuf.size},
      };
      update_desc_set(device, pipe1.ds, infos1);

      // Dispatch pre-pass.
      uint32_t pcD = D;
      record_dispatch(pipe1.layout, pipe1.pipeline, pipe1.ds, D, &pcD, sizeof(pcD));
      if (vkResetFences(device, 1, &fence) != VK_SUCCESS) die("vkResetFences failed (lfm2 pre)");
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &cmd;
      if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) die("vkQueueSubmit failed (lfm2 pre)");
      if (!host_fence_wait(UINT64_MAX)) die("Fence timeout (lfm2 pre)");

      // Dispatch out-proj.
      uint32_t pcD2 = D;
      record_dispatch(pipe2.layout, pipe2.pipeline, pipe2.ds, D, &pcD2, sizeof(pcD2));
      if (vkResetFences(device, 1, &fence) != VK_SUCCESS) die("vkResetFences failed (lfm2 out)");
      if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) die("vkQueueSubmit failed (lfm2 out)");
      if (!host_fence_wait(UINT64_MAX)) die("Fence timeout (lfm2 out)");

      // Compare against exported expected for this wave.
      float max_y_err = 0.0f;
      float max_out_err = 0.0f;
      float max_state_err = 0.0f;
      const float* exp_yw = exp_y_pre + (size_t)w * (size_t)D;
      const float* exp_ow = exp_out + (size_t)w * (size_t)D;
      const float* exp_sw = exp_state + (size_t)w * (size_t)D * 2u;
      for (uint32_t i = 0; i < (D < 64u ? D : 64u); i++) {
        max_y_err = fmaxf(max_y_err, fabsf(y_f[i] - exp_yw[i]));
        max_out_err = fmaxf(max_out_err, fabsf(o_f[i] - exp_ow[i]));
        max_state_err = fmaxf(max_state_err, fabsf(state_out[i * 2 + 0] - exp_sw[i * 2 + 0]));
        max_state_err = fmaxf(max_state_err, fabsf(state_out[i * 2 + 1] - exp_sw[i * 2 + 1]));
      }
      LOGI("LFM2 ShortConv wave %u done. max_err(y)=%f max_err(out)=%f max_err(state)=%f", w, max_y_err, max_out_err, max_state_err);
    }
  }

  if (is_shortconv) {
    // ShortConv multi-wave run (always multi-submit).
    const uint32_t L = 3;
    const uint32_t S = L - 1;

    auto* bx_f = reinterpret_cast<float*>(bx.mapped);
    auto* c_f = reinterpret_cast<float*>(cbuf.mapped);
    auto* s0_f = reinterpret_cast<float*>(state0.mapped);
    auto* s1_f = reinterpret_cast<float*>(state1.mapped);
    auto* y_f = reinterpret_cast<float*>(ybuf.mapped);
    auto* w_f = reinterpret_cast<float*>(wbuf.mapped);

    auto cpu_step = [&](const float* state_in, float* state_out, float* y_out) {
      float max_err = 0.0f;
      (void)max_err;
      for (uint32_t i = 0; i < D; i++) {
        float s0 = state_in[i * S + 0];
        float s1 = state_in[i * S + 1];
        float x = bx_f[i];
        float conv_out = w_f[i * L + 0] * s0 + w_f[i * L + 1] * s1 + w_f[i * L + 2] * x;
        y_out[i] = c_f[i] * conv_out;
        state_out[i * S + 0] = s1;
        state_out[i * S + 1] = x;
      }
    };

    std::vector<float> y_ref(D);
    std::vector<float> s_ref(D * S);

    for (uint32_t w = 0; w < waves; w++) {
      const bool ping = (w % 2) == 0;
      float* state_in = ping ? s0_f : s1_f;
      float* state_out = ping ? s1_f : s0_f;

      // CPU reference.
      cpu_step(state_in, s_ref.data(), y_ref.data());

      // Update descriptor bindings 2 and 4 for state_in/state_out.
      std::vector<VkDescriptorBufferInfo> infos{
          {bx.buffer, 0, bx.size},                                        // 0
          {cbuf.buffer, 0, cbuf.size},                                    // 1
          {(ping ? state0.buffer : state1.buffer), 0, state0.size},       // 2
          {ybuf.buffer, 0, ybuf.size},                                    // 3
          {(ping ? state1.buffer : state0.buffer), 0, state1.size},       // 4
          {wbuf.buffer, 0, wbuf.size},                                    // 5
      };
      update_desc_set(device, pipe1.ds, infos);

      // Record and submit dispatch.
      uint32_t pc[2] = {D, L};
      record_dispatch(pipe1.layout, pipe1.pipeline, pipe1.ds, D, pc, sizeof(pc));

      if (vkResetFences(device, 1, &fence) != VK_SUCCESS) {
        die("vkResetFences failed (shortconv)");
      }
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &cmd;
      if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) {
        die("vkQueueSubmit failed (shortconv)");
      }
      if (!host_fence_wait(UINT64_MAX)) {
        die("Unexpected fence timeout (shortconv)");
      }

      // Compare GPU vs CPU reference.
      float max_abs_err = 0.0f;
      for (uint32_t i = 0; i < (D < 64u ? D : 64u); i++) {
        max_abs_err = fmaxf(max_abs_err, fabsf(y_f[i] - y_ref[i]));
      }
      // Compare state_out.
      float max_state_err = 0.0f;
      for (uint32_t i = 0; i < (D < 64u ? D : 64u); i++) {
        max_state_err = fmaxf(max_state_err, fabsf(state_out[i * S + 0] - s_ref[i * S + 0]));
        max_state_err = fmaxf(max_state_err, fabsf(state_out[i * S + 1] - s_ref[i * S + 1]));
      }
      LOGI("ShortConv wave %u done. max_abs_err(y,first64)=%f max_abs_err(state,first64)=%f", w, max_abs_err, max_state_err);
    }
  }

  if (is_shortconv_block) {
    auto* x_f = reinterpret_cast<float*>(xvec.mapped);
    auto* y_f = reinterpret_cast<float*>(ypre.mapped);
    auto* o_f = reinterpret_cast<float*>(outproj.mapped);
    auto* s0_f = reinterpret_cast<float*>(state0.mapped);
    auto* s1_f = reinterpret_cast<float*>(state1.mapped);
    auto* wdw_f = reinterpret_cast<float*>(wbuf.mapped);
    auto* Wb_f = reinterpret_cast<float*>(Wb.mapped);
    auto* Wc_f = reinterpret_cast<float*>(Wc.mapped);
    auto* Wx_f = reinterpret_cast<float*>(Wx.mapped);
    auto* Wo_f = reinterpret_cast<float*>(Wout.mapped);

    auto matvec = [&](const float* W, const float* v, float* outv) {
      for (uint32_t r = 0; r < D; r++) {
        float acc = 0.0f;
        for (uint32_t c = 0; c < D; c++) {
          acc += W[r * D + c] * v[c];
        }
        outv[r] = acc;
      }
    };

    std::vector<float> B(D), C(D), Xp(D), Bxv(D), y_ref(D), out_ref(D), s_ref(D * 2);

    for (uint32_t w = 0; w < waves; w++) {
      const bool ping = (w % 2) == 0;
      float* state_in = ping ? s0_f : s1_f;
      float* state_out = ping ? s1_f : s0_f;

      // CPU reference.
      matvec(Wb_f, x_f, B.data());
      matvec(Wc_f, x_f, C.data());
      matvec(Wx_f, x_f, Xp.data());
      for (uint32_t i = 0; i < D; i++) {
        Bxv[i] = B[i] * Xp[i];
        float s0 = state_in[i * 2 + 0];
        float s1 = state_in[i * 2 + 1];
        float conv_out = wdw_f[i * 3 + 0] * s0 + wdw_f[i * 3 + 1] * s1 + wdw_f[i * 3 + 2] * Bxv[i];
        y_ref[i] = C[i] * conv_out;
        s_ref[i * 2 + 0] = s1;
        s_ref[i * 2 + 1] = Bxv[i];
      }
      matvec(Wo_f, y_ref.data(), out_ref.data());

      // Update state bindings for pre-pass: binding1=state_in, binding3=state_out
      std::vector<VkDescriptorBufferInfo> infos1{
          {xvec.buffer, 0, xvec.size},
          {(ping ? state0.buffer : state1.buffer), 0, state0.size},
          {ypre.buffer, 0, ypre.size},
          {(ping ? state1.buffer : state0.buffer), 0, state1.size},
          {Wb.buffer, 0, Wb.size},
          {Wc.buffer, 0, Wc.size},
          {Wx.buffer, 0, Wx.size},
          {wbuf.buffer, 0, wbuf.size},
      };
      update_desc_set(device, pipe1.ds, infos1);

      // Dispatch pre-pass.
      uint32_t pcD = D;
      record_dispatch(pipe1.layout, pipe1.pipeline, pipe1.ds, D, &pcD, sizeof(pcD));
      if (vkResetFences(device, 1, &fence) != VK_SUCCESS) die("vkResetFences failed (pre)");
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &cmd;
      if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) die("vkQueueSubmit failed (pre)");
      if (!host_fence_wait(UINT64_MAX)) die("Fence timeout (pre)");

      // Dispatch out-proj.
      uint32_t pcD2 = D;
      record_dispatch(pipe2.layout, pipe2.pipeline, pipe2.ds, D, &pcD2, sizeof(pcD2));
      if (vkResetFences(device, 1, &fence) != VK_SUCCESS) die("vkResetFences failed (out)");
      if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) die("vkQueueSubmit failed (out)");
      if (!host_fence_wait(UINT64_MAX)) die("Fence timeout (out)");

      // Compare.
      float max_y_err = 0.0f;
      float max_out_err = 0.0f;
      float max_state_err = 0.0f;
      for (uint32_t i = 0; i < (D < 64u ? D : 64u); i++) {
        max_y_err = fmaxf(max_y_err, fabsf(y_f[i] - y_ref[i]));
        max_out_err = fmaxf(max_out_err, fabsf(o_f[i] - out_ref[i]));
        max_state_err = fmaxf(max_state_err, fabsf(state_out[i * 2 + 0] - s_ref[i * 2 + 0]));
        max_state_err = fmaxf(max_state_err, fabsf(state_out[i * 2 + 1] - s_ref[i * 2 + 1]));
      }
      LOGI("ShortConvBlock wave %u done. max_err(y)=%f max_err(out)=%f max_err(state)=%f", w, max_y_err, max_out_err, max_state_err);
    }
  }

  // Cleanup.
  vkDestroyFence(device, fence, nullptr);
  vkDestroyCommandPool(device, cmd_pool, nullptr);
  destroy_pipeline(pipe2);
  destroy_pipeline(pipe1);

  destroy_buffer(in);
  destroy_buffer(st);
  destroy_buffer(out);
  destroy_buffer(sig);
  destroy_buffer(bx);
  destroy_buffer(cbuf);
  destroy_buffer(state0);
  destroy_buffer(state1);
  destroy_buffer(ybuf);
  destroy_buffer(wbuf);
  destroy_buffer(xvec);
  destroy_buffer(ypre);
  destroy_buffer(outproj);
  destroy_buffer(Wb);
  destroy_buffer(Wc);
  destroy_buffer(Wx);
  destroy_buffer(Wout);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);

  LOGI("Neural Interposer demo finished.");
  return 0;
}

