#!/bin/bash
# rebuild_llama_mt6855v.sh - Rebuild llama.cpp with MT6855V Assembly Optimization

set -e

echo "🔧 Rebuilding llama.cpp with MT6855V Assembly Optimization"
echo "========================================================="

# Go to llama.cpp directory
cd research/llama.cpp

# Clean previous build
echo "Cleaning previous build..."
rm -rf build-android-vulkan-mt6855v

# Create build directory with MT6855V optimization
echo "Creating build directory for MT6855V..."
mkdir -p build-android-vulkan-mt6855v
cd build-android-vulkan-mt6855v

# Configure with MT6855V specific optimizations
echo "Configuring build with MT6855V optimizations..."

# Android NDK (adjust path as needed)
NDK="${ANDROID_NDK:-$HOME/Library/Android/sdk/ndk/28.2.13676358}"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/darwin-x86_64"

# MT6855V specific flags for Cortex-A78/A55
ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16 -mtune=cortex-a78"
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM="android-28"

# Configure CMake with assembly optimization
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=$ANDROID_ABI \
    -DANDROID_PLATFORM=$ANDROID_PLATFORM \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$ARCH_FLAGS" \
    -DCMAKE_CXX_FLAGS="$ARCH_FLAGS" \
    -DGGML_CPU_KLEIDIAI=ON \
    -DGGML_VULKAN=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=OFF \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/install"

echo "Building llama.cpp with MT6855V optimizations..."
make -j$(sysctl -n hw.ncpu)

echo "✅ Build complete!"
echo ""
echo "Built files:"
ls -la bin/ | head -5

echo ""
echo "🚀 Ready for deployment to Motorola device!"
echo "   Next: Copy the optimized binaries to device and test performance"