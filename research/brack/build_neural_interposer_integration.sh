#!/bin/bash

# Build Neural Interposer Integration
# Compiles the updated ExecuTorch runner with Neural Interposer support

set -e

echo "=== Building Neural Interposer Integration ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for Android NDK
if [ -z "$ANDROID_NDK" ]; then
    log_error "ANDROID_NDK environment variable not set"
    log_info "Please set ANDROID_NDK to your Android NDK path"
    log_info "Example: export ANDROID_NDK=/opt/android-ndk-r26c"
    exit 1
fi

if [ ! -d "$ANDROID_NDK" ]; then
    log_error "Android NDK not found at $ANDROID_NDK"
    exit 1
fi

log_info "Using Android NDK: $ANDROID_NDK"

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for ExecuTorch submodule
EXECUTORCH_DIR="$SCRIPT_DIR/../spaceghost/executorch"
if [ ! -d "$EXECUTORCH_DIR" ]; then
    log_error "ExecuTorch submodule not found at $EXECUTORCH_DIR"
    log_info "Please initialize submodules:"
    log_info "  git submodule update --init --recursive"
    exit 1
fi

# Create build directory
BUILD_DIR="$SCRIPT_DIR/executorch_android_runner/build-neural-interposer"
log_info "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure build
log_info "Configuring build with Neural Interposer support..."

cmake \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DEXECUTORCH_ROOT="$EXECUTORCH_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_NEURAL_INTERPOSER=ON \
    "$SCRIPT_DIR/executorch_android_runner"

if [ $? -ne 0 ]; then
    log_error "CMake configuration failed"
    exit 1
fi

# Build
log_info "Building ExecuTorch runner with Neural Interposer..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    log_error "Build failed"
    exit 1
fi

# Check output
EXECUTORCH_RUNNER="$BUILD_DIR/executorch_runner"
if [ ! -f "$EXECUTORCH_RUNNER" ]; then
    log_error "Build completed but executorch_runner not found"
    exit 1
fi

log_info "✓ Build successful!"
log_info "Executable: $EXECUTORCH_RUNNER"

# Get file size
FILE_SIZE=$(stat -c%s "$EXECUTORCH_RUNNER" 2>/dev/null || stat -f%z "$EXECUTORCH_RUNNER" 2>/dev/null || echo "unknown")
log_info "File size: $FILE_SIZE bytes"

# Verify Neural Interposer symbols
log_info "Verifying Neural Interposer integration..."
if nm "$EXECUTORCH_RUNNER" 2>/dev/null | grep -q "ni_channel_create_ion"; then
    log_info "✓ ION channel functions linked"
else
    log_warn "! ION channel functions not found in binary"
fi

if nm "$EXECUTORCH_RUNNER" 2>/dev/null | grep -q "ni_trix_context_create"; then
    log_info "✓ TriX context functions linked"
else
    log_warn "! TriX context functions not found in binary"
fi

if nm "$EXECUTORCH_RUNNER" 2>/dev/null | grep -q "shortconv3_step_out"; then
    log_info "✓ ShortConv3 custom op linked"
else
    log_warn "! ShortConv3 custom op not found in binary"
fi

# Instructions for device deployment
echo
log_info "=== Deployment Instructions ==="
echo
log_info "1. Push the executable to device:"
echo "   adb push $EXECUTORCH_RUNNER /data/local/tmp/"
echo
log_info "2. Push required SPIR-V shader:"
echo "   adb push $SCRIPT_DIR/neural_interposer_demo/build-android/shortconv_chip.spv /data/local/tmp/"
echo
log_info "3. Generate test model:"
echo "   python3 $SCRIPT_DIR/lfm2_explicit_state/ni_shortconv3_smoke_export.py --out smoke_test.pte --D 1024"
echo "   adb push smoke_test.pte /data/local/tmp/"
echo
log_info "4. Run integration test:"
echo "   bash $SCRIPT_DIR/test_neural_interposer_integration.sh"
echo
log_info "=== Neural Interposer Integration Build Complete ==="