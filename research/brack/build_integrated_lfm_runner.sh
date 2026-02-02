#!/bin/bash
# Build Integrated LFM Pipeline Runner for Phase 3 Device Verification

set -e

echo "🔨 Building Integrated LFM Pipeline Runner (Phase 3)"
echo "=================================================="

# Configuration
BUILD_TYPE=${1:-Release}
ANDROID_ABI=${2:-arm64-v8a}
ANDROID_API_LEVEL=${3:-28}

echo "Build Type: $BUILD_TYPE"
echo "Android ABI: $ANDROID_ABI"
echo "API Level: $ANDROID_API_LEVEL"

# Set up directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/brack/build"
INSTALL_DIR="$BUILD_DIR/install"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "📋 Configuring with CMake..."

cmake "$PROJECT_ROOT" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="android-$ANDROID_API_LEVEL" \
    -DANDROID_STL=c++_shared \
    -DBUILD_EXECUTORCH=ON \
    -DBUILD_INTEGRATED_LFM_PIPELINE=ON \
    -DENABLE_NEURAL_INTERPOSER=ON \
    -DENABLE_QUANTIZATION=ON \
    -DENABLE_LAYER_SHARDING=ON \
    -DENABLE_ION_MEMORY=ON \
    -DENABLE_VULKAN_ACCELERATION=ON \
    -DENABLE_MEMORY_MONITORING=ON \
    -DENABLE_ACCURACY_VALIDATION=ON \
    -DENABLE_PHASE3_VERIFICATION=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

# Build
echo "🔨 Building..."

make -j$(nproc) install

# Verify build
echo "✅ Verifying build..."

if [ ! -f "$INSTALL_DIR/bin/executorch_runner_integrated" ]; then
    echo "❌ Build failed: executorch_runner_integrated not found"
    exit 1
fi

if [ ! -f "$INSTALL_DIR/lib/libneural_interposer.so" ]; then
    echo "❌ Build failed: libneural_interposer.so not found"
    exit 1
fi

# Check file sizes
RUNNER_SIZE=$(stat -f%z "$INSTALL_DIR/bin/executorch_runner_integrated" 2>/dev/null || stat -c%s "$INSTALL_DIR/bin/executorch_runner_integrated")
LIB_SIZE=$(stat -f%z "$INSTALL_DIR/lib/libneural_interposer.so" 2>/dev/null || stat -c%s "$INSTALL_DIR/lib/libneural_interposer.so")

echo "✅ Build successful!"
echo "   Runner: $INSTALL_DIR/bin/executorch_runner_integrated (${RUNNER_SIZE} bytes)"
echo "   Library: $INSTALL_DIR/lib/libneural_interposer.so (${LIB_SIZE} bytes)"

# Copy additional files needed for deployment
echo "📦 Preparing deployment package..."

# Copy shaders
mkdir -p "$BUILD_DIR/shaders"
if [ -d "$PROJECT_ROOT/brack/neural_interposer_demo/shaders" ]; then
    cp "$PROJECT_ROOT/brack/neural_interposer_demo/shaders"/*.spv "$BUILD_DIR/shaders/" 2>/dev/null || true
    echo "   Copied $(ls "$BUILD_DIR/shaders"/*.spv 2>/dev/null | wc -l) shader files"
fi

# Copy test models (placeholder - in practice would copy actual quantized model)
mkdir -p "$BUILD_DIR/models"
echo "   Created models directory (add quantized LFM2-350M model here)"

# Create deployment info
cat > "$BUILD_DIR/deployment_info.txt" << EOF
Phase 3 Integrated LFM Pipeline Deployment
=========================================

Build Date: $(date)
Build Type: $BUILD_TYPE
Android ABI: $ANDROID_ABI
API Level: $ANDROID_API_LEVEL

Components:
- executorch_runner_integrated: Integrated LFM2-350M pipeline
- libneural_interposer.so: Neural Interposer runtime library
- Shaders: Vulkan compute kernels for quantization
- Models: Quantized LFM2-350M weights

Targets:
- End-to-end latency: <300ms
- Peak memory usage: <280MB
- Accuracy preservation: >99%
- Hardware: MediaTek MT6855V + Mali-G52

Verification:
Run: ./phase3_device_verification.py
EOF

echo "✅ Deployment package ready at: $BUILD_DIR"
echo "   Run 'python research/brack/phase3_device_verification.py' to test on device"

echo ""
echo "🎯 Phase 3 Build Complete!"
echo "   Ready for device deployment and falsification testing"