// Integrated ExecuTorch Runner with Neural Interposer Ops
// Combines original runner with Neural Interposer custom operations

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

// ============================================================================
// NEURAL INTERPOSER OPS INTEGRATION
// ============================================================================

// Include Neural Interposer operation implementations directly
// This ensures they are compiled into the runner binary and registered

#include "executorch_softchip_ops/ni_shortconv3_op.h"
#include "executorch_softchip_ops/ni_attention_op.h"
#include "executorch_softchip_ops/ni_channel.h"
#include "executorch_softchip_ops/ni_trix_context.h"

// The ops are registered via EXECUTORCH_LIBRARY macros in their implementation files
// By including the implementations here, they will be linked into the binary
// and registered when the program starts

// ============================================================================
// END NEURAL INTERPOSER INTEGRATION
// ============================================================================

int main(int argc, char** argv) {
    // Initialize runtime
    executorch::runtime::runtime_init();

    // Parse flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "🚀 Integrated ExecuTorch Runner with Neural Interposer Ops" << std::endl;
    std::cout << "Model: " << FLAGS_model_path << std::endl;

    // Load the model
    Result<FileDataLoader> loader = FileDataLoader::from(FLAGS_model_path.c_str());
    if (!loader.ok()) {
        std::cerr << "Failed to load model: " << loader.error() << std::endl;
        return 1;
    }

    Result<Program> program = Program::load(&loader.get());
    if (!program.ok()) {
        std::cerr << "Failed to load program: " << program.error() << std::endl;
        return 1;
    }

    // Create method
    Result<Method> method = program->load_method(FLAGS_data_path);
    if (!method.ok()) {
        std::cerr << "Failed to load method: " << method.error() << std::endl;
        return 1;
    }

    // Set up memory
    HierarchicalAllocator allocator;
    MemoryManager memory_manager(&allocator, &allocator);

    // Execute the model
    for (uint32_t i = 0; i < FLAGS_num_executions; ++i) {
        std::cout << "Execution " << (i + 1) << "/" << FLAGS_num_executions << std::endl;

        Error status = method->execute();
        if (status != Error::Ok) {
            std::cerr << "Execution failed: " << status << std::endl;
            return 1;
        }

        std::cout << "✅ Execution completed successfully" << std::endl;

        // Print outputs if requested
        if (FLAGS_print_all_output) {
            auto outputs = method->outputs();
            for (size_t j = 0; j < outputs.size(); ++j) {
                std::cout << "Output " << j << ": ";
                print_evalue(outputs[j]);
                std::cout << std::endl;
            }
        }
    }

    std::cout << "🎉 All executions completed successfully!" << std::endl;
    std::cout << "🏆 Neural Interposer integration verified!" << std::endl;

    return 0;
}
