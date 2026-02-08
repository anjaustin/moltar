#!/bin/bash
# rebuild_llama_mt6855v.sh - Rebuild llama.cpp with MT6855V DOTPROD Optimization
#
# Builds llama-bench and llama-cli for Android arm64-v8a
# with armv8.2-a+dotprod+fp16 (SDOT instructions for Q4_0 GEMV)
#
# Requirements:
#   - Android NDK r27c+ at /opt/android-ndk-r27c or $ANDROID_NDK
#   - cmake 3.16+

set -e

echo "Rebuilding llama.cpp with MT6855V DOTPROD optimization"
echo "======================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/research/llama.cpp"

if [ ! -f "${LLAMA_DIR}/CMakeLists.txt" ]; then
    echo "Error: llama.cpp not found at ${LLAMA_DIR}"
    echo "Run: git clone https://github.com/ggerganov/llama.cpp.git ${LLAMA_DIR}"
    exit 1
fi

# Android NDK detection
NDK="${ANDROID_NDK:-/opt/android-ndk-r27c}"
if [ ! -d "$NDK" ]; then
    echo "Error: Android NDK not found at $NDK"
    echo "Set ANDROID_NDK or install to /opt/android-ndk-r27c"
    exit 1
fi

TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo "Error: Android toolchain not found at $TOOLCHAIN_FILE"
    exit 1
fi

BUILD_DIR="${LLAMA_DIR}/build-android"

# MT6855V specific flags: Cortex-A78/A55 with DOTPROD + FP16
# This enables SDOT instructions which are critical for Q4_0 GEMV performance
ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"

echo "NDK:        $NDK"
echo "Build dir:  $BUILD_DIR"
echo "Arch flags: $ARCH_FLAGS"
echo ""

# Clean and configure
if [ "$1" = "clean" ]; then
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

echo "Configuring CMake..."
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_NATIVE_API_LEVEL=24 \
    -DCMAKE_C_FLAGS="$ARCH_FLAGS" \
    -DCMAKE_CXX_FLAGS="$ARCH_FLAGS" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=OFF

echo ""
echo "Building ($(nproc) threads)..."
cmake --build "$BUILD_DIR" --config Release -j$(nproc) -- llama-bench llama-cli

echo ""
echo "Build complete!"
echo ""
echo "Binaries:"
ls -lh "$BUILD_DIR/bin/llama-bench" "$BUILD_DIR/bin/llama-cli" 2>/dev/null
echo ""
echo "Next: ./deploy_to_device.sh"
