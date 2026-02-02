// Minimal Vulkan-backed custom op for ExecuTorch.
//
// This is a “make it real” integration point:
// - registers a custom op `ni::shortconv3_step.out`
// - executes the same shader as the Neural Interposer demo (shortconv_chip)
//
// Notes:
// - To keep packaging simple, SPIR-V is loaded from disk at runtime.
//   Set env var `NI_SHORTCONV3_SPV` to the .spv path, or push to:
//     /data/local/tmp/shortconv_chip.spv
// - This is a v0 “smoke integration” focused on correctness and stability.

#include "ni_shortconv3_op.h"
#include "ni_attention_op.h"
#include "ni_channel.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <vulkan/vulkan.h>

#include <executorch/runtime/platform/log.h>

namespace {

static const char* kDefaultSpvPath = "/data/local/tmp/shortconv_chip.spv";

// Global TriX context (shared across all ops)
static ni_trix_context_t* g_trix_context = nullptr;
static VkDevice g_vk_device = VK_NULL_HANDLE;
static VkPhysicalDevice g_vk_physical_device = VK_NULL_HANDLE;
static VkQueue g_vk_queue = VK_NULL_HANDLE;
static std::mutex g_context_mutex;

// Initialize TriX context
static bool ensure_trix_context(VkDevice device, VkPhysicalDevice phys_device, VkQueue queue) {
    std::lock_guard<std::mutex> lock(g_context_mutex);

    if (g_trix_context) return true;

    g_vk_device = device;
    g_vk_physical_device = phys_device;
    g_vk_queue = queue;

    // LFM2-350M dimensions
    const uint32_t hidden_dim = 1024;
    const uint32_t state_dim = hidden_dim * 3;  // kernel_size - 1 = 3

    g_trix_context = ni_trix_context_create(device, phys_device, queue, hidden_dim, state_dim);

    if (!g_trix_context) {
        ET_LOG(Error, "Failed to create TriX context");
        return false;
    }

    ET_LOG(Info, "Initialized TriX context for Neural Interposer integration");
    return true;
}

std::vector<uint8_t> read_file_bytes(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    return {};
  }
  const std::size_t n = static_cast<std::size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(n);
  if (n > 0 && !f.read(reinterpret_cast<char*>(bytes.data()), n)) {
    return {};
  }
  return bytes;
}

bool is_f32(const executorch::aten::Tensor& t) {
  return t.scalar_type() == executorch::aten::ScalarType::Float;
}

bool is_contig_dim_order(const executorch::aten::Tensor& t) {
  return torch::executor::is_contiguous_dim_order(
      t.dim_order().data(), t.dim());
}

struct VkCtx {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice dev = VK_NULL_HANDLE;
  uint32_t qfam = 0;
  VkQueue queue = VK_NULL_HANDLE;

  VkCommandPool cmd_pool = VK_NULL_HANDLE;

  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VkPipelineLayout pl = VK_NULL_HANDLE;
  VkPipeline pipe = VK_NULL_HANDLE;

  bool init_ok = false;
};

VkResult pick_compute_queue_family(
    VkPhysicalDevice phys,
    uint32_t* out_qfam) {
  uint32_t n = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
  std::vector<VkQueueFamilyProperties> props(n);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, props.data());
  for (uint32_t i = 0; i < n; i++) {
    if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      *out_qfam = i;
      return VK_SUCCESS;
    }
  }
  return VK_ERROR_INITIALIZATION_FAILED;
}

VkResult create_instance(VkInstance* out_instance) {
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "ni_shortconv3_op";
  app.applicationVersion = 1;
  app.pEngineName = "moltar";
  app.engineVersion = 1;
  app.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app;
  return vkCreateInstance(&ci, nullptr, out_instance);
}

VkResult create_device(
    VkPhysicalDevice phys,
    uint32_t qfam,
    VkDevice* out_dev,
    VkQueue* out_queue) {
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;

  VkResult r = vkCreateDevice(phys, &dci, nullptr, out_dev);
  if (r != VK_SUCCESS) {
    return r;
  }
  vkGetDeviceQueue(*out_dev, qfam, 0, out_queue);
  return VK_SUCCESS;
}

VkResult find_mem_type(
    VkPhysicalDevice phys,
    uint32_t type_bits,
    VkMemoryPropertyFlags flags,
    uint32_t* out_type_index) {
  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((type_bits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & flags) == flags)) {
      *out_type_index = i;
      return VK_SUCCESS;
    }
  }
  return VK_ERROR_MEMORY_MAP_FAILED;
}

struct Buf {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize nbytes = 0;
};

VkResult make_mapped_storage_buffer(
    VkCtx& c,
    VkDeviceSize nbytes,
    VkBufferUsageFlags usage,
    Buf* out) {
  out->nbytes = nbytes;

  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = nbytes;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult r = vkCreateBuffer(c.dev, &bi, nullptr, &out->buf);
  if (r != VK_SUCCESS) return r;

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(c.dev, out->buf, &req);
  uint32_t ti = 0;
  r = find_mem_type(
      c.phys,
      req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      &ti);
  if (r != VK_SUCCESS) return r;

  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = ti;
  r = vkAllocateMemory(c.dev, &ai, nullptr, &out->mem);
  if (r != VK_SUCCESS) return r;

  r = vkBindBufferMemory(c.dev, out->buf, out->mem, 0);
  if (r != VK_SUCCESS) return r;

  r = vkMapMemory(c.dev, out->mem, 0, nbytes, 0, &out->mapped);
  return r;
}

void destroy_buf(VkCtx& c, Buf& b) {
  if (b.mapped) {
    vkUnmapMemory(c.dev, b.mem);
    b.mapped = nullptr;
  }
  if (b.mem) {
    vkFreeMemory(c.dev, b.mem, nullptr);
    b.mem = VK_NULL_HANDLE;
  }
  if (b.buf) {
    vkDestroyBuffer(c.dev, b.buf, nullptr);
    b.buf = VK_NULL_HANDLE;
  }
}

VkResult create_pipeline(VkCtx& c, const std::vector<uint8_t>& spv) {
  if (spv.empty() || (spv.size() % 4) != 0) return VK_ERROR_INITIALIZATION_FAILED;

  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spv.size();
  smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());

  VkShaderModule sm = VK_NULL_HANDLE;
  VkResult r = vkCreateShaderModule(c.dev, &smci, nullptr, &sm);
  if (r != VK_SUCCESS) return r;

  // Bindings match shortconv_chip.comp in neural_interposer_demo:
  // 0 bx, 1 c, 2 state_in, 3 y_out, 4 state_out, 5 w
  VkDescriptorSetLayoutBinding b[6]{};
  for (uint32_t i = 0; i < 6; i++) {
    b[i].binding = i;
    b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[i].descriptorCount = 1;
    b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dsci{};
  dsci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsci.bindingCount = 6;
  dsci.pBindings = b;
  r = vkCreateDescriptorSetLayout(c.dev, &dsci, nullptr, &c.dsl);
  if (r != VK_SUCCESS) return r;

  // Push constants: { uint D; uint L; }
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = 8;

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &c.dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  r = vkCreatePipelineLayout(c.dev, &plci, nullptr, &c.pl);
  if (r != VK_SUCCESS) return r;

  VkPipelineShaderStageCreateInfo ss{};
  ss.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ss.module = sm;
  ss.pName = "main";

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage = ss;
  cpci.layout = c.pl;
  r = vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &c.pipe);

  vkDestroyShaderModule(c.dev, sm, nullptr);
  return r;
}

VkCtx& global_vk() {
  static VkCtx g;
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  if (g.init_ok) return g;

  VkResult r = create_instance(&g.instance);
  if (r != VK_SUCCESS) {
    ET_LOG(Error, "ni_shortconv3: vkCreateInstance failed (%d)", (int)r);
    return g;
  }

  uint32_t ndev = 0;
  r = vkEnumeratePhysicalDevices(g.instance, &ndev, nullptr);
  if (r != VK_SUCCESS || ndev == 0) {
    ET_LOG(Error, "ni_shortconv3: no Vulkan physical devices (%d, n=%u)", (int)r, ndev);
    return g;
  }
  std::vector<VkPhysicalDevice> devs(ndev);
  vkEnumeratePhysicalDevices(g.instance, &ndev, devs.data());
  g.phys = devs[0];

  r = pick_compute_queue_family(g.phys, &g.qfam);
  if (r != VK_SUCCESS) {
    ET_LOG(Error, "ni_shortconv3: no compute queue family");
    return g;
  }

  r = create_device(g.phys, g.qfam, &g.dev, &g.queue);
  if (r != VK_SUCCESS) {
    ET_LOG(Error, "ni_shortconv3: vkCreateDevice failed (%d)", (int)r);
    return g;
  }

  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.queueFamilyIndex = g.qfam;
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  r = vkCreateCommandPool(g.dev, &cpci, nullptr, &g.cmd_pool);
  if (r != VK_SUCCESS) {
    ET_LOG(Error, "ni_shortconv3: vkCreateCommandPool failed (%d)", (int)r);
    return g;
  }

  const char* spv_path = std::getenv("NI_SHORTCONV3_SPV");
  if (!spv_path || !spv_path[0]) spv_path = kDefaultSpvPath;
  std::vector<uint8_t> spv = read_file_bytes(spv_path);
  if (spv.empty()) {
    ET_LOG(
        Error,
        "ni_shortconv3: failed to read SPV at '%s' (set NI_SHORTCONV3_SPV)",
        spv_path);
    return g;
  }

  r = create_pipeline(g, spv);
  if (r != VK_SUCCESS) {
    ET_LOG(Error, "ni_shortconv3: create_pipeline failed (%d)", (int)r);
    return g;
  }

  g.init_ok = true;
  ET_LOG(Info, "ni_shortconv3: Vulkan init OK (spv=%s)", spv_path);
  return g;
}

} // namespace

namespace ni {

executorch::aten::Tensor& shortconv3_step_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& bx,
    const executorch::aten::Tensor& c,
    const executorch::aten::Tensor& state,
    const executorch::aten::Tensor& w,
    executorch::aten::Tensor& out) {
  using torch::executor::Error;

  // Basic validation (v0).
  ET_KERNEL_CHECK(ctx, is_f32(bx) && is_f32(c) && is_f32(state) && is_f32(w) && is_f32(out), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, is_contig_dim_order(bx) && is_contig_dim_order(c) && is_contig_dim_order(state) && is_contig_dim_order(w) && is_contig_dim_order(out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, bx.dim() == 1, InvalidArgument, out);
  const int64_t D = bx.size(0);
  ET_KERNEL_CHECK(ctx, c.dim() == 1 && c.size(0) == D, InvalidArgument, out);

  // state: accept [D,2] or [D*2]
  int64_t state_numel = state.numel();
  ET_KERNEL_CHECK(ctx, state_numel == D * 2, InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, w.numel() == D * 3, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, bx.sizes()) == Error::Ok,
      InvalidArgument,
      out);

  VkCtx& vk = global_vk();
  ET_KERNEL_CHECK(ctx, vk.init_ok, InvalidArgument, out);

  // Initialize TriX context if needed
  if (!ensure_trix_context(vk.dev, vk.phys, vk.queue)) {
    ET_LOG(Error, "Failed to initialize TriX context");
    return out;
  }

  // Use TriX context for Neural Interposer execution
  {
    std::lock_guard<std::mutex> lock(g_context_mutex);

    if (!g_trix_context) {
      ET_LOG(Error, "TriX context not initialized");
      return out;
    }

    // Execute ShortConv chip via TriX context
    const float* input_data = bx.const_data_ptr<float>();
    const float* state_data = state.const_data_ptr<float>();
    const float* weights_data = w.const_data_ptr<float>();
    float* output_data = out.mutable_data_ptr<float>();

    // Create temporary buffers for next state (not used in this v0)
    std::vector<float> next_state(D * 2, 0.0f);

    bool success = ni_trix_execute_shortconv(g_trix_context,
                                            input_data, state_data,
                                            output_data, next_state.data(),
                                            weights_data, (uint32_t)D);

    if (!success) {
      ET_LOG(Error, "TriX ShortConv execution failed");
      return out;
    }

    ET_LOG(Info, "Executed ShortConv3 via Neural Interposer (D=%lld)", D);
  }

  return out;
}

} // namespace ni

// Register with ExecuTorch runtime as custom ops
EXECUTORCH_LIBRARY(ni, "shortconv3_step.out", ni::shortconv3_step_out);
EXECUTORCH_LIBRARY(ni, "attention_step.out", ni::attention_step_out);

