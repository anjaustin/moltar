#!/bin/bash
# Deploy Phase 3 Integrated Pipeline to Motorola Device and Run Verification

set -e

echo "📦 Deploying Phase 3 Integrated LFM Pipeline to Device"
echo "======================================================"

# Configuration
DEVICE_IP=${1:-"192.168.1.100"}
ADB_PORT=${2:-"5555"}
DEVICE_PATH="/data/local/tmp/neural_interposer_phase3"

echo "Device IP: $DEVICE_IP"
echo "ADB Port: $ADB_PORT"
echo "Device Path: $DEVICE_PATH"

# Check if build exists
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ Build directory not found. Run build_integrated_lfm_runner.sh first"
    exit 1
fi

# Connect to device
echo "🔌 Connecting to device..."
adb disconnect >/dev/null 2>&1 || true
adb connect "$DEVICE_IP:$ADB_PORT"

# Wait for connection
sleep 2

# Verify connection
if ! adb devices | grep -q "$DEVICE_IP"; then
    echo "❌ Failed to connect to device at $DEVICE_IP:$ADB_PORT"
    echo "   Make sure device is connected via USB and ADB over network is enabled"
    echo "   Or check IP address and port"
    exit 1
fi

echo "✅ Connected to device"

# Get device info
echo "📱 Device Information:"
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.mediatek.platform

# Create device directory
echo "📁 Creating device directory..."
adb shell mkdir -p "$DEVICE_PATH"

# Push executorch runner
echo "📤 Pushing executorch runner..."
if [ -f "$BUILD_DIR/install/bin/executorch_runner_integrated" ]; then
    adb push "$BUILD_DIR/install/bin/executorch_runner_integrated" "$DEVICE_PATH/"
    adb shell chmod +x "$DEVICE_PATH/executorch_runner_integrated"
    echo "✅ Pushed executorch runner"
else
    echo "❌ Executorch runner not found in build directory"
    exit 1
fi

# Push neural interposer library
echo "📤 Pushing neural interposer library..."
if [ -f "$BUILD_DIR/install/lib/libneural_interposer.so" ]; then
    adb push "$BUILD_DIR/install/lib/libneural_interposer.so" "$DEVICE_PATH/"
    echo "✅ Pushed neural interposer library"
else
    echo "❌ Neural interposer library not found in build directory"
    exit 1
fi

# Push shaders
echo "📤 Pushing Vulkan shaders..."
if [ -d "$BUILD_DIR/shaders" ] && [ "$(ls -A "$BUILD_DIR/shaders"/*.spv 2>/dev/null)" ]; then
    adb push "$BUILD_DIR/shaders"/*.spv "$DEVICE_PATH/"
    SHADER_COUNT=$(ls "$BUILD_DIR/shaders"/*.spv | wc -l)
    echo "✅ Pushed $SHADER_COUNT shader files"
else
    echo "⚠️  No shader files found (non-critical for basic testing)"
fi

# Push test model (placeholder)
echo "📤 Pushing test model..."
if [ -f "research/brack/models/lfm2_350m_quantized.pte" ]; then
    adb push "research/brack/models/lfm2_350m_quantized.pte" "$DEVICE_PATH/"
    echo "✅ Pushed LFM2-350M quantized model"
else
    echo "⚠️  LFM2-350M model not found - using synthetic test data"
    # Create a simple test input file
    adb shell "echo '512,1024,2048' > $DEVICE_PATH/test_sequences.txt"
fi

# Set up environment
echo "🔧 Setting up device environment..."

# Set library path
adb shell "export LD_LIBRARY_PATH=$DEVICE_PATH:$LD_LIBRARY_PATH"

# Verify Vulkan support
echo "🔍 Checking Vulkan support..."
if adb shell "ls /vendor/lib64/libvulkan.so" >/dev/null 2>&1; then
    echo "✅ Vulkan library found"
else
    echo "⚠️  Vulkan library not found - GPU acceleration may not work"
fi

# Check ION memory
echo "🔍 Checking ION memory support..."
if adb shell "ls /dev/ion" >/dev/null 2>&1; then
    echo "✅ ION memory device available"
else
    echo "⚠️  ION memory device not found - zero-copy may not work"
fi

# Test basic execution
echo "🧪 Testing basic execution..."
TEST_CMD="$DEVICE_PATH/executorch_runner_integrated --help"
if adb shell "$TEST_CMD" >/dev/null 2>&1; then
    echo "✅ Basic execution test passed"
else
    echo "❌ Basic execution test failed"
    echo "   Command: $TEST_CMD"
    exit 1
fi

# Run verification tests
echo "🔬 Running Phase 3 verification tests..."

# Copy verification script to device
echo "📤 Pushing verification script..."
adb push "research/brack/phase3_device_verification.py" "$DEVICE_PATH/"

# Run verification from host (since it needs to orchestrate device tests)
echo "🚀 Starting verification..."
python3 research/brack/phase3_device_verification.py

echo ""
echo "🎯 Phase 3 Device Deployment Complete!"
echo "   Pipeline deployed to: $DEVICE_PATH"
echo "   Verification results saved to: research/brack/phase3_device_verification_results.json"
echo ""
echo "📊 Check the verification results above for:"
echo "   - Latency performance (<300ms target)"
echo "   - Memory usage (<280MB target)"
echo "   - Accuracy preservation (>99% target)"
echo "   - Hardware utilization (GPU, thermal, power)"
echo ""
echo "🔧 If verification fails, proceed to Phase 3 Week 8 optimization"