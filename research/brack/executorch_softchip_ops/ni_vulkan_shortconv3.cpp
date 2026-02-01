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

  const std::size_t bx_bytes = (std::size_t)D * sizeof(float);
  const std::size_t c_bytes = (std::size_t)D * sizeof(float);
  const std::size_t st_bytes = (std::size_t)D * 2 * sizeof(float);
  const std::size_t w_bytes = (std::size_t)D * 3 * sizeof(float);
  const std::size_t out_bytes = (std::size_t)D * sizeof(float);

  Buf bx_b{}, c_b{}, st_in_b{}, y_b{}, st_out_b{}, w_b{};
  VkResult r = VK_SUCCESS;
  r = make_mapped_storage_buffer(vk, bx_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bx_b);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);
  r = make_mapped_storage_buffer(vk, c_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &c_b);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);
  r = make_mapped_storage_buffer(vk, st_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &st_in_b);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);
  r = make_mapped_storage_buffer(vk, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_b);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);
  r = make_mapped_storage_buffer(vk, st_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &st_out_b);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);
  r = make_mapped_storage_buffer(vk, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_b);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);

  std::memcpy(bx_b.mapped, bx.const_data_ptr(), bx_bytes);
  std::memcpy(c_b.mapped, c.const_data_ptr(), c_bytes);
  std::memcpy(st_in_b.mapped, state.const_data_ptr(), st_bytes);
  std::memcpy(w_b.mapped, w.const_data_ptr(), w_bytes);

  // Descriptor pool + set (per-call for v0 simplicity).
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 6;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool dpool = VK_NULL_HANDLE;
  r = vkCreateDescriptorPool(vk.dev, &dpci, nullptr, &dpool);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);

  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = dpool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &vk.dsl;
  VkDescriptorSet dset = VK_NULL_HANDLE;
  r = vkAllocateDescriptorSets(vk.dev, &dsai, &dset);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);

  auto write_buf = [&](uint32_t binding, VkBuffer buf, VkDeviceSize nbytes) {
    VkDescriptorBufferInfo bi{};
    bi.buffer = buf;
    bi.offset = 0;
    bi.range = nbytes;
    VkWriteDescriptorSet wds{};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = dset;
    wds.dstBinding = binding;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &bi;
    vkUpdateDescriptorSets(vk.dev, 1, &wds, 0, nullptr);
  };
  write_buf(0, bx_b.buf, bx_bytes);
  write_buf(1, c_b.buf, c_bytes);
  write_buf(2, st_in_b.buf, st_bytes);
  write_buf(3, y_b.buf, out_bytes);
  write_buf(4, st_out_b.buf, st_bytes);
  write_buf(5, w_b.buf, w_bytes);

  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = vk.cmd_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  r = vkAllocateCommandBuffers(vk.dev, &cbai, &cmd);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);

  VkCommandBufferBeginInfo cbbi{};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  r = vkBeginCommandBuffer(cmd, &cbbi);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk.pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk.pl, 0, 1, &dset, 0, nullptr);

  struct Push {
    uint32_t D;
    uint32_t L;
  } pc;
  pc.D = (uint32_t)D;
  pc.L = 3u;
  vkCmdPushConstants(cmd, vk.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &pc);

  const uint32_t wg = 256u;
  const uint32_t gx = (pc.D + wg - 1u) / wg;
  vkCmdDispatch(cmd, gx, 1, 1);

  r = vkEndCommandBuffer(cmd);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);

  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  r = vkCreateFence(vk.dev, &fci, nullptr, &fence);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  r = vkQueueSubmit(vk.queue, 1, &si, fence);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);
  r = vkWaitForFences(vk.dev, 1, &fence, VK_TRUE, UINT64_MAX);
  ET_KERNEL_CHECK(ctx, r == VK_SUCCESS, InvalidArgument, out);

  // Copy back to ExecuTorch tensors.
  std::memcpy(out.mutable_data_ptr(), y_b.mapped, out_bytes);

  // Cleanup per-call objects.
  vkDestroyFence(vk.dev, fence, nullptr);
  vkFreeCommandBuffers(vk.dev, vk.cmd_pool, 1, &cmd);
  vkDestroyDescriptorPool(vk.dev, dpool, nullptr);

  destroy_buf(vk, bx_b);
  destroy_buf(vk, c_b);
  destroy_buf(vk, st_in_b);
  destroy_buf(vk, y_b);
  destroy_buf(vk, st_out_b);
  destroy_buf(vk, w_b);

  return out;
}

} // namespace ni

// Register with ExecuTorch runtime as custom op: ni::shortconv3_step.out
EXECUTORCH_LIBRARY(ni, "shortconv3_step.out", ni::shortconv3_step_out);

