#!/bin/bash
# bisect_bench.sh - Build and benchmark a specific llama.cpp commit on device
# Usage: ./bisect_bench.sh <commit_hash>

set -e

COMMIT="$1"
if [ -z "$COMMIT" ]; then
    echo "Usage: $0 <commit_hash>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/research/llama.cpp"
NDK="/opt/android-ndk-r27c"
BUILD_DIR="${LLAMA_DIR}/build-android"
DEVICE_DIR="/data/local/tmp"

echo "=== Testing commit: ${COMMIT} ==="

# Checkout
cd "$LLAMA_DIR"
git checkout "$COMMIT" 2>&1 | tail -1

SHORT=$(git rev-parse --short HEAD)
echo "Checked out: ${SHORT} $(git log -1 --format='%s' | head -c 60)"

# Clean and build
rm -rf "$BUILD_DIR"
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_NATIVE_API_LEVEL=24 \
    -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod+fp16" \
    -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod+fp16" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=OFF 2>&1 | tail -3

cmake --build "$BUILD_DIR" --config Release -j$(nproc) -- llama-bench 2>&1 | tail -3

if [ ! -f "$BUILD_DIR/bin/llama-bench" ]; then
    echo "FAILED: llama-bench not built"
    exit 1
fi

# Deploy
echo "Deploying..."
adb push "$BUILD_DIR/bin/llama-bench" "${DEVICE_DIR}/llama-bench" 2>&1 | tail -1
for lib in libggml-base.so libggml-cpu.so libggml.so libllama.so; do
    [ -f "$BUILD_DIR/bin/${lib}" ] && adb push "$BUILD_DIR/bin/${lib}" "${DEVICE_DIR}/${lib}" 2>&1 | tail -1
done
adb shell "chmod 755 ${DEVICE_DIR}/llama-bench"

# Benchmark
echo ""
echo "=== BENCHMARK: ${SHORT} ==="
adb shell "cd ${DEVICE_DIR} && LD_LIBRARY_PATH=. taskset c0 ./llama-bench -m LFM2-350M-Q4_0.gguf -t 2 -fa 1 -n 32 -p 32" 2>&1

# Return to main
cd "$LLAMA_DIR"
git checkout main 2>&1 | tail -1
