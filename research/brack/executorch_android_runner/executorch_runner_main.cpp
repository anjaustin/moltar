// Local ExecuTorch runner wrapper.
//
// Why this exists:
// - The upstream `executor_runner.cpp` tends to fail “quietly” (Release builds
//   often compile logging out), which makes on-device debugging painful.
// - This file keeps behavior similar, but prints explicit errors to stderr and
//   increases allocator pools so large models (LFM2) can at least load.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::aten::Tensor;
using executorch::extension::FileDataLoader;
using executorch::extension::FlatTensorDataMap;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

DEFINE_string(model_path, "model.pte", "Model serialized in flatbuffer format.");
DEFINE_string(data_path, "", "Path to data file (.ptd).");
DEFINE_string(inputs, "", "Comma-separated list of input files");
DEFINE_uint32(num_executions, 1, "Number of times to run the model.");
DEFINE_bool(
    print_all_output,
    false,
    "Print all output scalars (very large for LLMs).");

// Bump pools substantially vs upstream.
static uint8_t method_allocator_pool[64 * 1024U * 1024U]; // 64 MB
static uint8_t temp_allocator_pool[64 * 1024U * 1024U]; // 64 MB

static void die(const std::string& msg, int code = 1) {
  std::cerr << "executorch_runner: " << msg << std::endl;
  std::exit(code);
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::stringstream ss;
    ss << "extra commandline args:";
    for (int i = 1; i < argc; i++) ss << " " << argv[i];
    die(ss.str());
  }

  // Optional input buffers from files.
  std::vector<std::string> inputs_storage;
  std::vector<std::pair<char*, size_t>> input_buffers;
  if (!FLAGS_inputs.empty()) {
    std::stringstream list(FLAGS_inputs);
    std::string path;
    std::vector<std::string> file_paths;
    while (std::getline(list, path, ',')) file_paths.push_back(std::move(path));

    inputs_storage.reserve(file_paths.size());
    for (const auto& file_path : file_paths) {
      std::ifstream f(file_path, std::ios::binary | std::ios::ate);
      if (!f) {
        die("failed to open input file: " + file_path);
      }
      std::streamsize n = f.tellg();
      f.seekg(0, std::ios::beg);
      inputs_storage.emplace_back((size_t)n, '\0');
      if (n > 0 && !f.read(inputs_storage.back().data(), n)) {
        die("failed to read input file: " + file_path);
      }
      input_buffers.emplace_back(&inputs_storage.back()[0], (size_t)n);
    }
  }

  std::unique_ptr<FileDataLoader> ptd_loader;
  std::unique_ptr<FlatTensorDataMap> ptd_data_map;
  if (!FLAGS_data_path.empty()) {
    auto loader_res = FileDataLoader::from(FLAGS_data_path.c_str());
    if (!loader_res.ok()) {
      std::stringstream ss;
      ss << "FileDataLoader::from(.ptd) failed: 0x" << std::hex
         << (uint32_t)loader_res.error();
      die(ss.str());
    }
    ptd_loader = std::make_unique<FileDataLoader>(std::move(loader_res.get()));
    auto map_res = FlatTensorDataMap::load(ptd_loader.get());
    if (!map_res.ok()) {
      std::stringstream ss;
      ss << "FlatTensorDataMap::load(.ptd) failed: 0x" << std::hex
         << (uint32_t)map_res.error();
      die(ss.str());
    }
    ptd_data_map =
        std::make_unique<FlatTensorDataMap>(std::move(map_res.get()));
  }

  // Load program.
  auto file_loader = FileDataLoader::from(FLAGS_model_path.c_str());
  if (!file_loader.ok()) {
    std::stringstream ss;
    ss << "FileDataLoader::from(.pte) failed: 0x" << std::hex
       << (uint32_t)file_loader.error();
    die(ss.str());
  }
  FileDataLoader loader(std::move(file_loader.get()));

  Result<Program> program = Program::load(&loader);
  if (!program.ok()) {
    std::stringstream ss;
    ss << "Program::load failed for " << FLAGS_model_path << " with 0x"
       << std::hex << (uint32_t)program.error();
    die(ss.str());
  }

  const char* method_name = nullptr;
  {
    const auto name_res = program->get_method_name(0);
    if (!name_res.ok()) die("program has no methods");
    method_name = *name_res;
  }
  std::cerr << "Using method: " << method_name << std::endl;

  Result<MethodMeta> meta = program->method_meta(method_name);
  if (!meta.ok()) {
    std::stringstream ss;
    ss << "method_meta failed: 0x" << std::hex << (uint32_t)meta.error();
    die(ss.str());
  }

  // Allocate planned buffers (can be large).
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  const size_t nbuf = meta->num_memory_planned_buffers();
  planned_buffers.reserve(nbuf);
  planned_spans.reserve(nbuf);

  for (size_t id = 0; id < nbuf; ++id) {
    const size_t bytes =
        static_cast<size_t>(meta->memory_planned_buffer_size(id).get());
    std::cerr << "Planned buffer " << id << " size " << bytes << " bytes"
              << std::endl;
    uint8_t* raw = new (std::nothrow) uint8_t[bytes];
    if (!raw) {
      die("OOM allocating planned buffer " + std::to_string(id) + " (" +
          std::to_string(bytes) + " bytes)");
    }
    planned_buffers.emplace_back(raw);
    planned_spans.push_back({planned_buffers.back().get(), bytes});
  }

  MemoryAllocator method_allocator(sizeof(method_allocator_pool), method_allocator_pool);
  MemoryAllocator temp_allocator(sizeof(temp_allocator_pool), temp_allocator_pool);
  HierarchicalAllocator planned_memory({planned_spans.data(), planned_spans.size()});
  MemoryManager mm(&method_allocator, &planned_memory, &temp_allocator);

  Result<Method> method =
      program->load_method(method_name, &mm, /*event_tracer=*/nullptr, ptd_data_map.get());
  if (!method.ok()) {
    std::stringstream ss;
    ss << "load_method failed: 0x" << std::hex << (uint32_t)method.error();
    die(ss.str());
  }

  // Execute.
  for (uint32_t i = 0; i < FLAGS_num_executions; i++) {
    auto inputs_res = executorch::extension::prepare_input_tensors(*method, {}, input_buffers);
    if (!inputs_res.ok()) {
      std::stringstream ss;
      ss << "prepare_input_tensors failed: 0x" << std::hex << (uint32_t)inputs_res.error();
      die(ss.str());
    }
    auto cleanup = std::make_optional(std::move(inputs_res.get()));

    Error st = method->execute();
    if (st != Error::Ok) {
      std::stringstream ss;
      ss << "execute failed: 0x" << std::hex << (uint32_t)st;
      die(ss.str());
    }
  }

  std::vector<EValue> outputs(method->outputs_size());
  Error out_st = method->get_outputs(outputs.data(), outputs.size());
  if (out_st != Error::Ok) {
    std::stringstream ss;
    ss << "get_outputs failed: 0x" << std::hex << (uint32_t)out_st;
    die(ss.str());
  }

  if (FLAGS_print_all_output) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::cout << "Output[" << i << "]: " << outputs[i] << std::endl;
    }
  } else {
    std::cout << executorch::extension::evalue_edge_items(100);
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::cout << "OutputX " << i << ": " << outputs[i] << std::endl;
    }
  }

  return 0;
}

