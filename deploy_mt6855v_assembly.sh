#!/bin/bash
# deploy_mt6855v_assembly.sh - Deploy MT6855V Assembly Optimization to Motorola

set -e

echo "🚀 Deploying MT6855V Assembly Optimization to Motorola Device"
echo "==========================================================="

# Device connection check
echo "Checking device connection..."
if ! adb devices | grep -q "device$"; then
    echo "❌ No device connected. Please connect your Motorola device."
    exit 1
fi

DEVICE_MODEL=$(adb shell getprop ro.product.model | tr -d '\r')
echo "✅ Device connected: $DEVICE_MODEL"

# Check if it's our target device
if [[ "$DEVICE_MODEL" != *"moto g power 5G"* ]]; then
    echo "⚠️  Warning: Not the expected Motorola device, but continuing..."
fi

# Deploy assembly optimization files
echo ""
echo "Deploying assembly optimization files..."

# Create deployment directory
adb shell "mkdir -p /data/local/tmp/mt6855v_assembly"

# Deploy the assembly library
echo "📦 Deploying MT6855V assembly library..."
adb push mt6855v_sdot_matvec_simple.o /data/local/tmp/mt6855v_assembly/
adb push mt6855v_sdot_matvec.h /data/local/tmp/mt6855v_assembly/

# Deploy test program
echo "🧪 Deploying test program..."
adb push test_mt6855v_asm /data/local/tmp/mt6855v_assembly/
adb shell "chmod +x /data/local/tmp/mt6855v_assembly/test_mt6855v_asm"

# Deploy performance simulator
echo "📊 Deploying performance simulator..."
adb push simulate_mt6855v_performance.o /data/local/tmp/mt6855v_assembly/

# Set permissions
echo "🔐 Setting permissions..."
adb shell "chmod 755 /data/local/tmp/mt6855v_assembly/*"

# Test deployment
echo ""
echo "Testing deployment..."
echo ""

# Run the assembly test
adb shell "cd /data/local/tmp/mt6855v_assembly && ./test_mt6855v_asm"

# Check results
echo ""
echo "🎯 Checking test results..."

# Get performance metrics
adb shell "cd /data/local/tmp/mt6855v_assembly && echo 'Performance check:' && ./test_mt6855v_asm 2>&1 | grep -E '(TARGET ACHIEVED|Assembly vs C)'"

# Integration with existing llama.cpp
echo ""
echo "🔧 Integrating with existing llama.cpp..."

# Check current llama.cpp status
echo "Current llama.cpp status:"
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-completion -m LFM2-350M-Q4_0.gguf -p 'Test' -n 1 --no-warmup 2>&1 | grep -E '(token|performance|speed)' | head -3"

# Future integration notes
echo ""
echo "📋 Integration Notes:"
echo "   1. Assembly kernels are deployed to /data/local/tmp/mt6855v_assembly/"
echo "   2. To integrate with llama.cpp, add -L/data/local/tmp/mt6855v_assembly -lmt6855v_asm"
echo "   3. Call mt6855v_matvec_dispatch() instead of standard matvec when available"
echo "   4. Performance target: 35-38 tok/s (35-46% improvement)"
echo ""
echo "✅ MT6855V Assembly Optimization Deployed Successfully!"
echo "   Ready for production testing on actual ARM64 hardware!"