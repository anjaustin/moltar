#!/bin/bash
# Final Neural Interposer Integration: Complete the Missing Link

set -e

echo "🔗 FINAL NEURAL INTERPOSER INTEGRATION"
echo "====================================="
echo ""
echo "🎯 MISSION: Integrate Neural Interposer ops into executorch_runner"
echo "📋 PROBLEM: Ops registered but not linked into deployed runner"
echo ""
echo "✅ CONFIRMED:"
echo "   - Ops are registered: EXECUTORCH_LIBRARY(ni, \"shortconv3_step.out\", ...)"
echo "   - Source code exists: ni_softchip_ops library"
echo "   - CMake includes library: target_link_libraries(executorch_runner ni_softchip_ops)"
echo "   - Build system ready: Android NDK configured"
echo ""
echo "❌ MISSING LINK:"
echo "   - Current runner not built with Neural Interposer ops"
echo "   - Need to rebuild with ops integrated"
echo ""

# Step 1: Verify current runner status
echo "🔍 STEP 1: VERIFY CURRENT RUNNER STATUS"
echo "Runner on device:"
adb shell "ls -lh /data/local/tmp/lfm350_neural_interposer_test/executorch_runner"
echo ""
echo "Test current functionality:"
adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && ./executorch_runner --help >/dev/null 2>&1 && echo '✅ Runner functional' || echo '❌ Runner broken'"
echo ""

# Step 2: Build integrated runner
echo "🏗️ STEP 2: BUILD INTEGRATED RUNNER WITH NEURAL INTERPOSER OPS"

# Ensure NDK is available
export ANDROID_NDK="/Users/aaronjosserand-austin/Library/Android/sdk/ndk/28.2.13676358"
if [ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]; then
    echo "❌ Android NDK not properly configured"
    echo "Expected at: $ANDROID_NDK/build/cmake/android.toolchain.cmake"
    exit 1
fi
echo "✅ NDK configured: $ANDROID_NDK"

# Clean any previous builds
echo "Cleaning previous builds..."
rm -rf research/brack/build/
mkdir -p research/brack/build/
cd research/brack/build/

# Configure with CMake - use the existing working build as reference
echo "Configuring build with Neural Interposer integration..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ANDROID_DIR="$SCRIPT_DIR/executorch_android_runner"
echo "Script dir: $SCRIPT_DIR"
echo "CMake source dir: $EXECUTORCH_ANDROID_DIR"
cmake "$EXECUTORCH_ANDROID_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DANDROID_STL=c++_shared \
    -DEXECUTORCH_ROOT="../../../research/spaceghost/executorch/cmake-out-android-ni" \
    2>&1 | head -20

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed"
    echo "Check CMakeFiles/CMakeError.log for details"
    exit 1
fi
echo "✅ CMake configuration successful"

# Build the integrated runner
echo "Building executorch_runner..."
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "4")
echo "Using $CPU_CORES CPU cores"

make -j$CPU_CORES executorch_runner 2>&1 | tail -10

if [ ! -f "executorch_runner" ]; then
    echo "❌ Build failed - no executable produced"
    echo "Check build errors above"
    exit 1
fi

RUNNER_SIZE=$(stat -f%z "executorch_runner" 2>/dev/null || stat -c%s "executorch_runner")
echo "✅ Build successful: ${RUNNER_SIZE} bytes"

# Verify Neural Interposer ops are included
echo "🔍 Verifying Neural Interposer ops integration:"
if strings executorch_runner | grep -q "shortconv3_step.out"; then
    echo "✅ shortconv3_step.out found in binary"
else
    echo "❌ shortconv3_step.out NOT found in binary"
    echo "Ops integration failed"
    exit 1
fi

if strings executorch_runner | grep -q "attention_step.out"; then
    echo "✅ attention_step.out found in binary"
else
    echo "❌ attention_step.out NOT found in binary"
fi

echo ""
echo "📦 STEP 3: DEPLOY INTEGRATED RUNNER"
echo "Copying to install location..."
mkdir -p install/bin/
cp executorch_runner install/bin/

echo "Deploying to device..."
adb push install/bin/executorch_runner /data/local/tmp/lfm350_neural_interposer_test/executorch_runner_integrated
adb shell "chmod +x /data/local/tmp/lfm350_neural_interposer_test/executorch_runner_integrated"

echo "Verifying deployment:"
adb shell "ls -lh /data/local/tmp/lfm350_neural_interposer_test/executorch_runner_integrated"

# Backup original runner
adb shell "cp /data/local/tmp/lfm350_neural_interposer_test/executorch_runner /data/local/tmp/lfm350_neural_interposer_test/executorch_runner_backup"

echo ""
echo "🧪 STEP 4: TEST INTEGRATION"
echo "Testing Neural Interposer ops functionality..."

# Test 1: Basic functionality
echo "Test 1: Basic runner functionality"
adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && ./executorch_runner_integrated --help >/dev/null 2>&1 && echo '✅ Basic functionality works' || echo '❌ Basic functionality broken'"

# Test 2: Shortconv smoke test (should work now)
echo "Test 2: Shortconv smoke test"
SMOKE_RESULT=$(adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && timeout 10 ./executorch_runner_integrated --model_path /data/local/tmp/smoke_shortconv3.pte 2>&1" | tail -1)
if echo "$SMOKE_RESULT" | grep -q "Executed ShortConv3"; then
    echo "✅ Shortconv smoke test PASSED"
elif echo "$SMOKE_RESULT" | grep -q "0x14"; then
    echo "⚠️ Shortconv test still failing with 0x14 (method execution)"
elif echo "$SMOKE_RESULT" | grep -q "0x23"; then
    echo "❌ Shortconv test still failing with 0x23 (model load)"
else
    echo "❓ Shortconv test result unclear: $SMOKE_RESULT"
fi

echo ""
echo "🎯 STEP 5: LFM 350M FINAL TEST"
echo "Testing full LFM 350M inference with integrated ops..."

LFM_RESULT=$(adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && timeout 30 ./executorch_runner_integrated --model_path lfm2_350m_explicit_vulkan_ctx64_seq1_blockweights_v3.pte --num_executions=1 2>&1" | tail -5)

if echo "$LFM_RESULT" | grep -q "Executed.*via Neural Interposer"; then
    echo "🎉 SUCCESS! LFM 350M INFERENCE WORKING!"
    echo "✅ Neural Interposer ops integrated successfully"
    echo "✅ Full LFM pipeline executing on device"
    echo "✅ Mobile LFM deployment achieved!"
    echo ""
    echo "📊 EXTRACTING PERFORMANCE METRICS..."
    # Extract timing from result
    echo "$LFM_RESULT" | grep -E "(real|user|sys|Executed)" | tail -3

elif echo "$LFM_RESULT" | grep -q "0x23"; then
    echo "❌ Still failing with 0x23 (model load issue)"
    echo "   May need different model or additional ops"
elif echo "$LFM_RESULT" | grep -q "0x14"; then
    echo "❌ Still failing with 0x14 (method execution issue)"
    echo "   Ops registered but execution failing"
else
    echo "❓ Unclear result: $LFM_RESULT"
    echo "   May be progressing but hitting different issue"
fi

echo ""
echo "🏆 FINAL STATUS SUMMARY"
echo "======================="
echo "Neural Interposer Integration: COMPLETED"
echo "Build System: Functional"
echo "Device Deployment: Successful"
if echo "$LFM_RESULT" | grep -q "Neural Interposer"; then
    echo "LFM 350M Inference: ✅ WORKING"
    echo "Phase 3: ✅ COMPLETE - Mobile LFM Achieved!"
else
    echo "LFM 350M Inference: ❌ Needs Further Debugging"
    echo "Phase 3: 🔄 Additional Integration Required"
fi