#!/usr/bin/env bash
# deploy_lfm2_350m.sh - Build and deploy LFM2-350M inference to Motorola device
# Tested on: moto g power 5G (2023), MediaTek Dimensity 930, Android 14
# Result: 26 tokens/sec generation, 63 tokens/sec prompt eval
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="$REPO_ROOT/research/llama.cpp"
MODEL_DIR="$REPO_ROOT/research/brack/models/LFM2-350M-GGUF"
BUILD_DIR="$LLAMA_DIR/build-android"
DEVICE_DIR="/data/local/tmp"

# Android NDK - adjust path as needed
NDK="${ANDROID_NDK:-$HOME/Library/Android/sdk/ndk/28.2.13676358}"
NDK_TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"

# Build configuration
# armv8.2-a+dotprod+fp16: enables NEON, dot product (SDOT/UDOT), and FP16 vector arithmetic
# KleidiAI: Arm's optimized GEMM/GEMV micro-kernels for Q4_0 and Q8_0
ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"

# Model configuration
MODEL_REPO="LiquidAI/LFM2-350M-GGUF"
MODEL_FILE="LFM2-350M-Q4_0.gguf"

# Inference defaults
THREADS=4       # 2x A76 big + 2x A55; avoid saturating all 8 cores
CONTEXT=2048    # Practical context; 128K default wastes 1.5GB on KV cache

echo "========================================="
echo "LFM2-350M Deployment for Motorola"
echo "========================================="

# --- Step 1: Verify device ---
echo ""
echo "[1/5] Checking device connection..."
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device connected. Enable USB debugging and reconnect."
    exit 1
fi
DEVICE_MODEL=$(adb shell getprop ro.product.model | tr -d '\r')
echo "  Device: $DEVICE_MODEL"

# --- Step 2: Build llama.cpp ---
echo ""
echo "[2/5] Building llama.cpp for ARM64 Android..."

if [ ! -f "$NDK_TOOLCHAIN" ]; then
    echo "ERROR: Android NDK not found at $NDK"
    echo "  Set ANDROID_NDK environment variable or install NDK 28+"
    exit 1
fi

mkdir -p "$BUILD_DIR"
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK_TOOLCHAIN" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$ARCH_FLAGS" \
    -DCMAKE_CXX_FLAGS="$ARCH_FLAGS" \
    -DGGML_CPU_KLEIDIAI=ON \
    -DGGML_VULKAN=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=OFF \
    > /dev/null 2>&1

cmake --build "$BUILD_DIR" --config Release -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" > /dev/null 2>&1
echo "  Build complete."

# --- Step 3: Download model ---
echo ""
echo "[3/5] Downloading $MODEL_FILE..."

mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$MODEL_REPO', '$MODEL_FILE', local_dir='$MODEL_DIR')
print('  Downloaded.')
"
else
    echo "  Already present ($(du -sh "$MODEL_DIR/$MODEL_FILE" | cut -f1))."
fi

# --- Step 4: Deploy to device ---
echo ""
echo "[4/5] Deploying to device..."

# Find OpenMP runtime from NDK
LIBOMP=$(find "$NDK" -name "libomp.so" -path "*aarch64*" | head -1)

# Push shared libraries
for lib in libggml-base.so libggml-cpu.so libggml.so libllama.so; do
    adb push "$BUILD_DIR/bin/$lib" "$DEVICE_DIR/" > /dev/null 2>&1
done
adb push "$LIBOMP" "$DEVICE_DIR/" > /dev/null 2>&1

# Push binaries
for bin in llama-completion llama-simple-chat llama-bench; do
    if [ -f "$BUILD_DIR/bin/$bin" ]; then
        adb push "$BUILD_DIR/bin/$bin" "$DEVICE_DIR/" > /dev/null 2>&1
        adb shell "chmod +x $DEVICE_DIR/$bin"
    fi
done

# Push model
adb push "$MODEL_DIR/$MODEL_FILE" "$DEVICE_DIR/" 2>&1 | grep -o '[0-9.]* MB/s'
echo "  Deployed."

# --- Step 5: Test inference ---
echo ""
echo "[5/5] Running inference test..."
echo ""

adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./llama-completion \
    -m $MODEL_FILE \
    -p 'The capital of France is' \
    -n 32 \
    -t $THREADS \
    -c $CONTEXT \
    --no-warmup \
    2>&1"

echo ""
echo "========================================="
echo "Deployment complete."
echo ""
echo "To run interactive chat on device:"
echo "  adb shell"
echo "  cd $DEVICE_DIR"
echo "  LD_LIBRARY_PATH=$DEVICE_DIR ./llama-completion -m $MODEL_FILE -t $THREADS -c $CONTEXT -cnv"
echo "========================================="
