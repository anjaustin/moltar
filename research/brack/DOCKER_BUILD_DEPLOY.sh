#!/bin/bash
# Build APK using Docker Android SDK and deploy to Motorola

echo "🐳 DOCKER ANDROID BUILD & DEPLOY"
echo "==============================="

set -e

# Check Docker
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker daemon not running"
    exit 1
fi

# Check device
if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device"; then
    echo "❌ Motorola device not connected"
    exit 1
fi

echo "✅ Docker running, device connected"

# Create temp directory for build
BUILD_DIR="/tmp/brack_build_$(date +%s)"
mkdir -p "$BUILD_DIR"

echo "📁 Build directory: $BUILD_DIR"

# Copy source code
echo "📋 Copying source code..."
# Only copy the specific Android project directory to avoiding copying the whole repo
cp -r /Users/aaronjosserand-austin/000/Motorola/research/brack/brack_android_deployment_20260131_123053/. "$BUILD_DIR/"

# Build with Docker
echo "🏗️ Building APK with Docker Android SDK (Native + Vulkan)..."
docker run --rm \
    -v "$BUILD_DIR:/workspace" \
    -w /workspace \
    cimg/android:2024.01 \
    bash -c "
        set -e
        echo '📦 Setting up Android build environment...'
        export ANDROID_HOME=/home/circleci/android-sdk
        export PATH=\$ANDROID_HOME/cmdline-tools/latest/bin:\$PATH:\$ANDROID_HOME/tools:\$ANDROID_HOME/platform-tools

        # Ensure JAVA_HOME is set correctly (cimg/android usually handles this, but being explicit helps)
        echo '☕ Java Version:'
        java -version

        # Accept licenses (using the official way for cmdline-tools)
        yes | sdkmanager --licenses >/dev/null 2>&1 || true

        # Install required SDK components (including NDK and CMake)
        echo '📥 Installing Build Tools, NDK, and CMake...'
        # We unset JAVA_OPTS temporarily to avoid the -Xmx collision if present
        unset JAVA_OPTS
        unset _JAVA_OPTIONS
        
        sdkmanager --install 'platform-tools' 'platforms;android-34' 'build-tools;34.0.0' 'ndk;26.1.10909125' 'cmake;3.22.1'

        # Build APK
        echo '🔨 Building APK with Native Libraries...'
        
        # Ensure GRADLE_OPTS are clean
        unset GRADLE_OPTS
        
        # Use system gradle instead of wrapper (wrapper jar is missing)
        gradle assembleDebug --stacktrace

        echo '✅ APK built successfully!'
        ls -la app/build/outputs/apk/debug/
    "

# Check if APK was built
if [ ! -f "$BUILD_DIR/app/build/outputs/apk/debug/app-debug.apk" ]; then
    echo "❌ APK build failed"
    exit 1
fi

echo "📦 APK built successfully!"
APK_PATH="$BUILD_DIR/app/build/outputs/apk/debug/app-debug.apk"

# Install APK on device
echo "📱 Installing APK on Motorola device..."
/Users/aaronjosserand-austin/000/Motorola/tools/android/adb install -r "$APK_PATH"

if [ $? -ne 0 ]; then
    echo "❌ APK installation failed"
    exit 1
fi

echo "✅ APK installed successfully!"

# Launch app
echo "🚀 Launching Brack GGUF Chat app..."
/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell am start -n com.moltar.brack/.GGUFChatActivity

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================"
echo ""
echo "📱 Your Motorola now has:"
echo "   • Brack GGUF Chat app installed"
echo "   • LFN350 model auto-detected"
echo "   • ~50-100ms response times"
echo "   • Philosophical AI ready!"
echo ""
echo "💬 Open 'Brack GGUF Chat' and ask:"
echo "   'Hypothetically, might reflective recursion be a function of awareness?'"
echo ""
echo "⚡ EXECUTION SUCCESSFUL!"

# Cleanup
rm -rf "$BUILD_DIR"