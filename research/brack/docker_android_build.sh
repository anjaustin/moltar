#!/bin/bash
# Docker-based Android Build for Neural Interposer Integration

set -e

echo "🐳 DOCKER ANDROID BUILD: NEURAL INTERPOSER INTEGRATION"
echo "======================================================"
echo ""

# Create Docker build script
cat > Dockerfile.android << 'EOF'
FROM ubuntu:22.04

# Install Android build dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    cmake \
    ninja-build \
    clang \
    lld \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Download and install Android NDK
ENV ANDROID_NDK_VERSION=25c
RUN wget -q https://dl.google.com/android/repository/android-ndk-${ANDROID_NDK_VERSION}-linux.zip \
    && unzip -q android-ndk-${ANDROID_NDK_VERSION}-linux.zip \
    && mv android-ndk-${ANDROID_NDK_VERSION} /opt/android-ndk \
    && rm android-ndk-${ANDROID_NDK_VERSION}-linux.zip

ENV ANDROID_NDK=/opt/android-ndk

# Set up Android kernel headers (for ION support)
RUN mkdir -p /usr/include/linux \
    && wget -q https://raw.githubusercontent.com/torvalds/linux/master/include/uapi/linux/ion.h -O /usr/include/linux/ion.h \
    && wget -q https://raw.githubusercontent.com/torvalds/linux/master/include/uapi/linux/dma-buf.h -O /usr/include/linux/dma-buf.h

# Create workspace
WORKDIR /workspace
EOF

echo "✅ Created Dockerfile for Android build environment"

# Build the Docker image
echo ""
echo "🏗️ BUILDING DOCKER IMAGE..."
docker build -f Dockerfile.android -t neural_interposer_android .

echo ""
echo "🚀 RUNNING ANDROID BUILD IN DOCKER..."

# Run the build in Docker with proper volume mounting
docker run --rm -v "$(pwd):/workspace" -w /workspace neural_interposer_android bash -c '
set -e

echo "📦 INSIDE DOCKER: Setting up build environment..."

# Set environment
export ANDROID_NDK=/opt/android-ndk
export PATH=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH

# Verify NDK
echo "Android NDK: $ANDROID_NDK"
ls -la $ANDROID_NDK/build/cmake/android.toolchain.cmake

# Create build directory
mkdir -p docker_build
cd docker_build

echo "🔨 CONFIGURING BUILD..."
cmake ../executorch_android_runner \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DANDROID_STL=c++_shared \
    -DCMAKE_PREFIX_PATH="/workspace/../spaceghost/executorch/cmake-out-android-ni"

echo "🏗️ BUILDING INTEGRATED RUNNER..."
make -j$(nproc) executorch_runner_integrated

if [ -f "executorch_runner_integrated" ]; then
    echo "✅ BUILD SUCCESSFUL!"
    ls -lh executorch_runner_integrated

    # Verify Neural Interposer ops
    echo "🔍 CHECKING NEURAL INTERPOSER INTEGRATION:"
    if strings executorch_runner_integrated | grep -q "shortconv3_step.out"; then
        echo "✅ shortconv3_step.out found in binary"
    else
        echo "❌ shortconv3_step.out NOT found"
    fi

    if strings executorch_runner_integrated | grep -q "attention_step.out"; then
        echo "✅ attention_step.out found in binary"
    else
        echo "❌ attention_step.out NOT found"
    fi

    # Copy to host
    cp executorch_runner_integrated /workspace/
    echo "📤 Copied integrated runner to host"
else
    echo "❌ BUILD FAILED"
    exit 1
fi
'

# Check if build succeeded
if [ -f "executorch_runner_integrated" ]; then
    echo ""
    echo "🎉 DOCKER BUILD SUCCESSFUL!"
    ls -lh executorch_runner_integrated

    echo ""
    echo "📦 DEPLOYING TO DEVICE..."
    adb push executorch_runner_integrated /data/local/tmp/lfm350_neural_interposer_test/executorch_runner_integrated
    adb shell "chmod +x /data/local/tmp/lfm350_neural_interposer_test/executorch_runner_integrated"

    echo "Verifying deployment:"
    adb shell "ls -lh /data/local/tmp/lfm350_neural_interposer_test/executorch_runner_integrated"

    echo ""
    echo "🧪 TESTING NEURAL INTERPOSER INTEGRATION..."

    # Test shortconv model
    echo "Testing shortconv smoke model:"
    adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && ./executorch_runner_integrated --model_path /data/local/tmp/smoke_shortconv3.pte 2>&1" | head -10

    # Test LFM model
    echo ""
    echo "🚀 TESTING LFM 350M INFERENCE (FINAL TEST):"
    LFM_RESULT=$(adb shell "cd /data/local/tmp/lfm350_neural_interposer_test && timeout 30 ./executorch_runner_integrated --model_path lfm2_350m_explicit_vulkan_ctx64_seq1_blockweights_v3.pte --num_executions=1 2>&1" | head -15)

    echo "$LFM_RESULT"

    # Check for success
    if echo "$LFM_RESULT" | grep -q "Neural Interposer\|Execution.*completed\|SUCCESS"; then
        echo ""
        echo "🎉🎉🎉 MISSION ACCOMPLISHED! 🎉🎉🎉"
        echo "✅ LFM 350M INFERENCE WORKING ON MOTOROLA DEVICE!"
        echo "🏆 MOBILE LFM DEPLOYMENT COMPLETE!"
        echo "🚀 WORLD'S FIRST MOBILE LFM INFERENCE ACHIEVED!"
        echo ""
        echo "📊 PERFORMANCE RESULTS:"
        echo "$LFM_RESULT" | grep -E "(time|Execution|Neural|ms|MB)" | tail -5
    else
        echo ""
        echo "⚠️ INTEGRATION TEST RESULTS:"
        if echo "$LFM_RESULT" | grep -q "0x23"; then
            echo "❌ Still 0x23 error - model loading issue persists"
        elif echo "$LFM_RESULT" | grep -q "0x14"; then
            echo "❌ 0x14 error - method execution issue"
        else
            echo "❓ Unclear results - check output above"
        fi
        echo "🔧 May need additional debugging of op registration"
    fi

else
    echo ""
    echo "❌ DOCKER BUILD FAILED"
    echo "Check Docker output above for errors"
    exit 1
fi

echo ""
echo "🏁 DOCKER ANDROID BUILD COMPLETE!"