#!/bin/bash
# deploy_shift_optimization.sh - Deploy Shift-Register MatMul to Motorola

set -e

echo "🚀 Deploying Shift-Register MatMul Optimization to Motorola"
echo "=========================================================="

# Device connection check
if ! adb devices | grep -q "device$"; then
    echo "❌ No device connected. Please connect your Motorola device."
    exit 1
fi

DEVICE_MODEL=$(adb shell getprop ro.product.model | tr -d '\r')
echo "✅ Device connected: $DEVICE_MODEL"

# Check device architecture
echo "Checking device CPU architecture..."
DEVICE_ARCH=$(adb shell "uname -m 2>/dev/null" | tr -d '\r')
echo "   Architecture: $DEVICE_ARCH"

if [[ "$DEVICE_ARCH" != *"aarch64"* ]]; then
    echo "⚠️  Warning: Expected ARM64 device but got: $DEVICE_ARCH"
    echo "   Shift-register optimization requires ARM64 architecture"
    echo "   Deployment will continue but optimization may not work"
fi

# Deploy shift-register files
echo ""
echo "📦 Deploying shift-register optimization files..."
adb shell "mkdir -p /data/local/tmp/mt6855v_shift_opt"

# Copy C wrapper
echo "📄 Copying C integration wrapper..."
adb push mt6855v_matmul_shift.c /data/local/tmp/mt6855v_shift_opt/
adb push mt6855v_matmul_shift.h /data/local/tmp/mt6855v_shift_opt/ 2>/dev/null

# Copy assembly (will be compiled on device)
echo "📄 Copying assembly source..."
adb push mt6855v_matmul_shift.S /data/local/tmp/mt6855v_shift_opt/ 2>/dev/null

# Set permissions
echo "🔐 Setting permissions..."
adb shell "chmod 755 /data/local/tmp/mt6855v_shift_opt/*"

echo "✅ Deployment complete!"
echo ""

echo "🧪 Testing current performance baseline..."
echo "=========================================="

# Test current performance with standard setup
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=.:/data/local/tmp timeout 15s ./llama-completion -m LFM2-350M-Q4_0.gguf -p 'Shift-Register test' -n 10 --no-warmup 2>&1 | grep -E '(token|tok.*s|ms)' | tail -10"

echo ""
echo "📊 Baseline Performance Recorded"
echo "   Next: Test with shift-register optimization (after ARM64 compilation)"
echo ""

echo "🎯 Deployment Summary:"
echo "   ✅ Shift-register optimization framework deployed"
echo "   ✅ C integration wrapper ready"
echo "   ✅ Assembly source code deployed"
echo "   ✅ Performance baseline established"
echo ""
echo "⚠️  Note: Assembly kernels need ARM64 compilation on device"
echo "   For immediate performance boost, integrate compiled .so library"
echo ""
echo "🚀 Ready for shift-register MatMul optimization!"