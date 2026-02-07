#!/bin/bash
# rebuild_llama_pure_asm.sh - Rebuild llama.cpp without KleidiAI for pure assembly performance

set -e

echo "🔧 Rebuilding llama.cpp without KleidiAI for pure assembly performance"
echo "====================================================================="

# Go to llama.cpp directory
cd research/llama.cpp

# Clean previous build
echo "Cleaning previous KleidiAI builds..."
rm -rf build-android-pure-asm build-android-no-kleidiai

# Create build directory without KleidiAI
echo "Creating build directory without KleidiAI..."
mkdir -p build-android-pure-asm
cd build-android-pure-asm

# Configure CMake WITHOUT KleidiAI for pure assembly performance
echo "Configuring build WITHOUT KleidiAI for pure assembly..."

# Android NDK (use existing toolchain)
# Use the existing Android build but disable KleidiAI
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$HOME/Library/Android/sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod+fp16 -mtune=cortex-a78" \
    -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod+fp16 -mtune=cortex-a78" \
    -DGGML_CPU_KLEIDIAI=OFF \
    -DGGML_VULKAN=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=OFF \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"

echo "Building llama.cpp without KleidiAI for pure assembly..."
make -j$(sysctl -n hw.ncpu)

echo "✅ Build complete without KleidiAI!"
echo ""
echo "Built files (without KleidiAI):"
ls -la bin/ | head -5

echo ""
echo "🚀 Ready for pure assembly deployment!"
echo "   Next: Deploy to Motorola device and benchmark pure assembly performance"