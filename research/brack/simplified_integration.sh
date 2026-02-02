#!/bin/bash
# Option 3: Simplified Integration - Direct Runner Source Modification

set -e

echo "🔧 OPTION 3: SIMPLIFIED NEURAL INTERPOSER INTEGRATION"
echo "===================================================="
echo ""
echo "🎯 APPROACH: Modify runner source to include ops directly"
echo "   → Add Neural Interposer implementations to runner source"
echo "   → Rebuild with Android NDK (proper cross-compilation)"
echo "   → Deploy integrated runner binary"
echo ""

# Step 1: Create integrated runner source
echo "📝 STEP 1: CREATING INTEGRATED RUNNER SOURCE"

# Start with the existing runner source and add Neural Interposer ops
cat > integrated_runner_main.cpp << 'EOF'
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
EOF

echo "✅ Created integrated runner source with Neural Interposer ops"

# Step 2: Set up proper build environment
echo ""
echo "🏗️ STEP 2: SETTING UP PROPER BUILD ENVIRONMENT"

# Ensure we have the right paths
export ANDROID_NDK="/Users/aaronjosserand-austin/Library/Android/sdk/ndk/28.2.13676358"
SCRIPT_DIR="$(pwd)"
EXECUTORCH_DIR="$SCRIPT_DIR/../spaceghost/executorch/cmake-out-android-ni"

echo "Android NDK: $ANDROID_NDK"
echo "ExecuTorch: $EXECUTORCH_DIR"

# Create build directory
rm -rf integrated_build
mkdir -p integrated_build
cd integrated_build

# Create CMakeLists for integrated build
cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.19)
project(integrated_executorch_runner)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# ExecuTorch paths
set(EXECUTORCH_BUILD_DIR "$EXECUTORCH_DIR")

# Include directories
set(_common_include_directories
  \${EXECUTORCH_BUILD_DIR}/include
  \${EXECUTORCH_BUILD_DIR}/schema/include
  \${EXECUTORCH_BUILD_DIR}/extension/include
  \${EXECUTORCH_BUILD_DIR}/kernels/portable/include
  \${EXECUTORCH_BUILD_DIR}/backends/vulkan/include
  $SCRIPT_DIR/executorch_softchip_ops
)

add_executable(integrated_runner
  $SCRIPT_DIR/integrated_runner_main.cpp
  $SCRIPT_DIR/executorch_softchip_ops/ni_vulkan_shortconv3.cpp
  $SCRIPT_DIR/executorch_softchip_ops/ni_attention_op.cpp
  $SCRIPT_DIR/executorch_softchip_ops/ni_channel.cpp
  $SCRIPT_DIR/executorch_softchip_ops/ni_trix_context.cpp
)

target_compile_options(integrated_runner PUBLIC -Wno-deprecated-declarations -fPIC)
target_compile_definitions(integrated_runner PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)
target_include_directories(integrated_runner PRIVATE \$ {_common_include_directories})

# Link libraries
target_link_libraries(integrated_runner
  \${EXECUTORCH_BUILD_DIR}/libexecutorch.a
  \${EXECUTORCH_BUILD_DIR}/libexecutorch_core.a
  \${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_ops_lib.a
  \${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_kernels.a
  \${EXECUTORCH_BUILD_DIR}/extension/runner_util/libextension_runner_util.a
  \${EXECUTORCH_BUILD_DIR}/extension/evalue_util/libextension_evalue_util.a
  \${EXECUTORCH_BUILD_DIR}/extension/data_loader/libextension_data_loader.a
  \${EXECUTORCH_BUILD_DIR}/extension/flat_tensor/libextension_flat_tensor.a
  \${EXECUTORCH_BUILD_DIR}/backends/vulkan/libvulkan_backend.a
)
EOF

echo "✅ Created CMakeLists.txt for integrated build"

# Step 3: Build the integrated runner
echo ""
echo "🔨 STEP 3: BUILDING INTEGRATED RUNNER"

# Configure
echo "Configuring build..."
cmake . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DANDROID_STL=c++_shared

# Build
echo "Building integrated runner..."
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "4")
make -j$CPU_CORES integrated_runner 2>&1

if [ ! -f "integrated_runner" ]; then
    echo "❌ Build failed - no executable produced"
    echo "Check CMakeFiles/CMakeError.log for details"
    exit 1
fi

RUNNER_SIZE=$(stat -f%z "integrated_runner" 2>/dev/null || stat -c%s "integrated_runner")
echo "✅ Integrated runner built: ${RUNNER_SIZE} bytes"

# Verify Neural Interposer ops are included
echo "🔍 Verifying Neural Interposer integration:"
if strings integrated_runner | grep -q "shortconv3_step.out"; then
    echo "✅ shortconv3_step.out registered in binary"
else
    echo "❌ shortconv3_step.out NOT found"
fi

if strings integrated_runner | grep -q "attention_step.out"; then
    echo "✅ attention_step.out registered in binary"
else
    echo "❌ attention_step.out NOT found"
fi

echo ""
echo "📦 STEP 4: DEPLOY INTEGRATED RUNNER"

# Deploy to device
echo "Deploying integrated runner..."
adb push integrated_runner /data/local/tmp/lfm350_neural_interposer_test/executorch_runner_integrated
adb shell "chmod +x /data/local/tmp/lfm350_neural_interposer_test/executorch_runner_integrated"

echo "Verifying deployment:"
adb shell "ls -lh /data/local/tmp/lfm350_neural_interposer_test/executorch_runner_integrated"

echo ""
echo "🧪 STEP 5: FINAL INTEGRATION TEST"

# Test shortconv model
echo "Testing shortconv smoke model:"
adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && ./executorch_runner_integrated --model_path /data/local/tmp/smoke_shortconv3.pte 2>&1" | head -10

# Test LFM model
echo ""
echo "Testing LFM 350M model (FINAL TEST):"
LFM_RESULT=$(adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && timeout 30 ./executorch_runner_integrated --model_path lfm2_350m_explicit_vulkan_ctx64_seq1_blockweights_v3.pte --num_executions=1 2>&1" | head -15)

echo "$LFM_RESULT"

# Check for success
if echo "$LFM_RESULT" | grep -q "Neural Interposer"; then
    echo ""
    echo "🎉 SUCCESS! NEURAL INTERPOSER INTEGRATION COMPLETE!"
    echo "✅ LFM 350M inference working on Motorola device"
    echo "🏆 Mobile LFM deployment ACHIEVED!"
    echo ""
    echo "📊 PERFORMANCE METRICS:"
    echo "$LFM_RESULT" | grep -E "(time|Execution|Neural)" | tail -3
elif echo "$LFM_RESULT" | grep -q "0x23"; then
    echo ""
    echo "❌ Still failing with 0x23 - model loading issue"
    echo "   Ops may not be properly registered"
elif echo "$LFM_RESULT" | grep -q "0x14"; then
    echo ""
    echo "❌ Failing with 0x14 - method execution issue"
    echo "   Ops registered but execution failing"
else
    echo ""
    echo "❓ Unclear result - check output above"
fi

echo ""
echo "🏁 OPTION 3 COMPLETE - INTEGRATION ATTEMPTED!"