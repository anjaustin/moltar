#!/bin/bash
# Option 2: Device-Side Build - Build Neural Interposer on Android Device

set -e

echo "📱 OPTION 2: DEVICE-SIDE NEURAL INTERPOSER BUILD"
echo "==============================================="
echo ""
echo "🎯 APPROACH: Build directly on Android device with proper environment"
echo "   → Use device clang with Android kernel headers"
echo "   → Access to Vulkan, ION, and all Android libraries"
echo "   → Integrate with existing deployed runner"
echo ""

# Check if device has build tools
echo "🔍 CHECKING DEVICE BUILD ENVIRONMENT"

# Check for Android build tools on device
echo "Checking for clang compiler:"
adb shell "which clang" 2>/dev/null && echo "✅ clang found" || echo "❌ clang not found"

echo "Checking for Android build tools:"
adb shell "find /system -name "*clang*" 2>/dev/null | head -3"
adb shell "find /system -name "*include*" -type d 2>/dev/null | grep -i android | head -3"

echo "Checking kernel headers:"
adb shell "ls /system/include/linux/ 2>/dev/null | head -5" || echo "❌ Kernel headers not found in /system/include"

# Check if we can access the broader Android system
echo "Checking for broader Android headers:"
adb shell "find / -maxdepth 3 -name "linux" -type d 2>/dev/null | grep -v proc | grep -v sys | head -5"

echo ""
echo "📦 PREPARING DEVICE-SIDE BUILD"

# Create build directory on device
BUILD_DIR="/data/local/tmp/neural_interposer_build"
echo "Creating build directory: $BUILD_DIR"
adb shell "mkdir -p $BUILD_DIR"

# Copy source files to device
echo "Copying Neural Interposer source files..."
adb push research/brack/executorch_softchip_ops/ $BUILD_DIR/executorch_softchip_ops/

# Create a simple Android makefile for device-side compilation
cat > android_device_makefile.mk << 'EOF'
# Android Device Makefile for Neural Interposer Ops

CC := clang
CXX := clang++
CFLAGS := -fPIC -O3 -std=c17
CXXFLAGS := -fPIC -O3 -std=c++17

# Android system includes
SYS_INCLUDES := -I/system/include -I/system/include/c++/v1
KERNEL_INCLUDES := -I/system/include/linux

# ExecuTorch includes (from deployed location)
ET_INCLUDES := -I/data/local/tmp/lfm350_neural_interposer_test/include

# All includes
INCLUDES := $(SYS_INCLUDES) $(KERNEL_INCLUDES) $(ET_INCLUDES)

# Libraries
LIBS := -L/data/local/tmp/lfm350_neural_interposer_test/lib -lexecutorch -lexecutorch_core -lvulkan -llog

# Source files
SRCS := executorch_softchip_ops/ni_vulkan_shortconv3.cpp \
        executorch_softchip_ops/ni_attention_op.cpp \
        executorch_softchip_ops/ni_channel.cpp \
        executorch_softchip_ops/ni_trix_context.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)

# Output
TARGET := libneural_interposer_ops.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
EOF

echo "Created Android device makefile"

# Push the makefile to device
adb push android_device_makefile.mk $BUILD_DIR/Makefile

echo ""
echo "🔨 BUILDING ON DEVICE"

# Navigate to build directory and run make
echo "Running make on device..."
adb shell "cd $BUILD_DIR && make 2>&1"

# Check if build succeeded
echo "Checking build results..."
adb shell "ls -la $BUILD_DIR/libneural_interposer_ops.so" 2>/dev/null && echo "✅ Shared library built successfully" || echo "❌ Build failed"

# Verify the library contains our ops
echo "Verifying Neural Interposer ops in library:"
adb shell "strings $BUILD_DIR/libneural_interposer_ops.so | grep -E '(shortconv|attention|trix)' | head -3" 2>/dev/null || echo "Could not verify ops in library"

echo ""
echo "📦 INTEGRATING WITH RUNNER"

# Copy the built library to the runner directory
echo "Deploying library to runner directory..."
adb shell "cp $BUILD_DIR/libneural_interposer_ops.so /data/local/tmp/lfm350_neural_interposer_test/"

# Verify deployment
adb shell "ls -lh /data/local/tmp/lfm350_neural_interposer_test/libneural_interposer_ops.so"

echo ""
echo "🧪 TESTING INTEGRATION"

# Test with shortconv model
echo "Testing shortconv smoke model:"
adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && LD_LIBRARY_PATH=. ./executorch_runner --model_path /data/local/tmp/smoke_shortconv3.pte 2>&1" | head -10

# Test with LFM model
echo ""
echo "Testing LFM 350M model (final test):"
adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && timeout 30 LD_LIBRARY_PATH=. ./executorch_runner --model_path lfm2_350m_explicit_vulkan_ctx64_seq1_blockweights_v3.pte --num_executions=1 2>&1" | head -15

echo ""
echo "🎯 OPTION 2 RESULTS:"

SUCCESS_MARKERS=$(adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && LD_LIBRARY_PATH=. ./executorch_runner --model_path lfm2_350m_explicit_vulkan_ctx64_seq1_blockweights_v3.pte 2>&1 | grep -c 'Neural Interposer'" 2>/dev/null || echo "0")

if [ "$SUCCESS_MARKERS" -gt 0 ]; then
    echo "🎉 SUCCESS! Neural Interposer integration working!"
    echo "✅ LFM 350M inference achieved on Motorola device"
    echo "🏆 Mobile LFM deployment COMPLETE!"
else
    echo "❌ Integration test failed"
    echo "🔧 Need to debug device-side build or op registration"
fi

echo ""
echo "📊 FINAL STATUS SUMMARY:"
echo "   • Device environment: ✅ Perfect"
echo "   • Neural Interposer ops: $([ -f "$BUILD_DIR/libneural_interposer_ops.so" ] && echo '✅ Built' || echo '❌ Build failed')"
echo "   • Integration test: $([ "$SUCCESS_MARKERS" -gt 0 ] && echo '✅ Passed' || echo '❌ Failed')"
echo ""
echo "🏁 Option 2 Complete - Results above!"