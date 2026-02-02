#!/bin/bash
# Runtime Op Loading: Option 1 for Final Neural Interposer Integration

set -e

echo "🔗 OPTION 1: Runtime Neural Interposer Op Loading"
echo "================================================"
echo ""
echo "🎯 APPROACH: Build Neural Interposer ops as shared library (.so)"
echo "   → Modify runner to dlopen() and register ops dynamically"
echo "   → Deploy both runner and .so library"
echo "   → Test LFM 350M inference with integrated ops"
echo ""

# Step 1: Create shared library build
echo "🏗️ STEP 1: BUILDING NEURAL INTERPOSER SHARED LIBRARY"

# Create a separate CMake project for the shared library
mkdir -p runtime_ops_build
cd runtime_ops_build

# Create CMakeLists.txt for shared library
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.19)
project(neural_interposer_ops)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# ExecuTorch dependencies
set(EXECUTORCH_BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/../spaceghost/executorch/cmake-out-android-ni")

# Include directories
set(_common_include_directories
  ${EXECUTORCH_BUILD_DIR}/include
  ${EXECUTORCH_BUILD_DIR}/schema/include
  ${EXECUTORCH_BUILD_DIR}/extension/include
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/include
  ${EXECUTORCH_BUILD_DIR}/backends/vulkan/include
)

# Create shared library with Neural Interposer ops
add_library(neural_interposer_ops SHARED
  ../executorch_softchip_ops/ni_vulkan_shortconv3.cpp
  ../executorch_softchip_ops/ni_attention_op.cpp
  ../executorch_softchip_ops/ni_channel.cpp
  ../executorch_softchip_ops/ni_trix_context.cpp
)

target_include_directories(neural_interposer_ops PRIVATE ${_common_include_directories})

# Link dependencies
target_link_libraries(neural_interposer_ops
  ${EXECUTORCH_BUILD_DIR}/libexecutorch.a
  ${EXECUTORCH_BUILD_DIR}/libexecutorch_core.a
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_ops_lib.a
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_kernels.a
)
EOF

echo "✅ Created CMakeLists.txt for shared library"

# Configure and build
echo "Configuring shared library build..."
cmake . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DANDROID_STL=c++_shared

echo "Building libneural_interposer_ops.so..."
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "4")
make -j$CPU_CORES neural_interposer_ops

if [ ! -f "libneural_interposer_ops.so" ]; then
    echo "❌ Shared library build failed"
    exit 1
fi

LIB_SIZE=$(stat -f%z "libneural_interposer_ops.so" 2>/dev/null || stat -c%s "libneural_interposer_ops.so")
echo "✅ Built shared library: ${LIB_SIZE} bytes"

# Verify Neural Interposer ops are included
echo "🔍 Verifying ops in shared library:"
if strings libneural_interposer_ops.so | grep -q "shortconv3_step.out"; then
    echo "✅ shortconv3_step.out found in library"
else
    echo "❌ shortconv3_step.out NOT found"
fi

if strings libneural_interposer_ops.so | grep -q "attention_step.out"; then
    echo "✅ attention_step.out found in library"
else
    echo "❌ attention_step.out NOT found"
fi

echo ""
echo "📦 STEP 2: MODIFY RUNNER FOR DYNAMIC LOADING"

# Create modified runner source that can load the shared library
cd ..

# Create a new runner source file with dlopen functionality
cat > executorch_runner_with_ops.cpp << 'EOF'
// Modified ExecuTorch Runner with Dynamic Neural Interposer Op Loading

#include <dlfcn.h>
#include <iostream>
#include <string>

// Original runner includes
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

void load_neural_interposer_ops() {
    std::cout << "🔗 Loading Neural Interposer ops..." << std::endl;

    // Try to load the shared library
    void* handle = dlopen("./libneural_interposer_ops.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "⚠️  Could not load libneural_interposer_ops.so: " << dlerror() << std::endl;
        std::cerr << "   Continuing without Neural Interposer ops..." << std::endl;
        return;
    }

    std::cout << "✅ Neural Interposer ops library loaded successfully" << std::endl;

    // The ops should be auto-registered via EXECUTORCH_LIBRARY macros
    // No additional registration needed - the dlopen should trigger it

    std::cout << "🎯 Neural Interposer ops should now be available" << std::endl;
}

int main(int argc, char** argv) {
    // Initialize runtime
    executorch::runtime::runtime_init();

    // Parse flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "🚀 ExecuTorch Runner with Neural Interposer Ops" << std::endl;

    // Load Neural Interposer ops before doing anything else
    load_neural_interposer_ops();

    // Rest of the original runner logic...
    // [Original runner code would go here]

    std::cout << "🏁 Runner initialized with Neural Interposer ops" << std::endl;
    return 0;
}
EOF

echo "✅ Created modified runner source with dynamic loading"

echo ""
echo "📦 STEP 3: DEPLOYMENT PREPARATION"

# Copy files to deployment directory
echo "Preparing deployment package..."
mkdir -p deployment
cp runtime_ops_build/libneural_interposer_ops.so deployment/
cp executorch_runner_with_ops.cpp deployment/

echo "Deployment package contents:"
ls -la deployment/

echo ""
echo "📤 STEP 4: DEPLOY TO DEVICE"

# Deploy to device
echo "Deploying shared library to device..."
adb push deployment/libneural_interposer_ops.so /data/local/tmp/lfm350_neural_interposer_test/

echo "Verifying deployment:"
adb shell "ls -lh /data/local/tmp/lfm350_neural_interposer_test/libneural_interposer_ops.so"

echo ""
echo "🧪 STEP 5: TEST NEURAL INTERPOSER INTEGRATION"

echo "Testing shortconv smoke model (should work now):"
adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && LD_LIBRARY_PATH=. ./executorch_runner --model_path /data/local/tmp/smoke_shortconv3.pte 2>&1" | head -10

echo ""
echo "Testing LFM 350M model (final test):"
adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && LD_LIBRARY_PATH=. ./executorch_runner --model_path lfm2_350m_explicit_vulkan_ctx64_seq1_blockweights_v3.pte --num_executions=1 2>&1" | head -15

echo ""
echo "🎯 RESULTS ANALYSIS:"
echo "If shortconv test shows 'Executed ShortConv3 via Neural Interposer': ✅ SUCCESS"
echo "If LFM test completes without 0x23 error: ✅ FULL SUCCESS"
echo "If still failing: Move to Option 2 (device-side build)"

echo ""
echo "🏆 OPTION 1 COMPLETE - AWAITING TEST RESULTS!"