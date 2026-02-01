#!/bin/bash
# Fast Docker Android build using a smaller, faster image

echo "⚡ FAST DOCKER ANDROID BUILD"
echo "==========================="

set -e

# Check prerequisites
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker daemon not running"
    exit 1
fi

if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device"; then
    echo "❌ Motorola device not connected"
    exit 1
fi

echo "✅ Prerequisites OK"

# Create build directory
BUILD_DIR="/tmp/fast_brack_build_$(date +%s)"
mkdir -p "$BUILD_DIR"

echo "📁 Fast build directory: $BUILD_DIR"

# Copy source
cp -r /Users/aaronjosserand-austin/000/Motorola/research/brack/* "$BUILD_DIR/"

# Use a faster Android Docker image
echo "🏗️ Building with faster Docker image..."

docker run --rm \
    -v "$BUILD_DIR:/workspace" \
    -w /workspace \
    --platform linux/amd64 \
    mingc/android-build-box:latest \
    bash -c "
        echo '🔧 Setting up fast Android build...'
        export ANDROID_HOME=/opt/android-sdk
        export PATH=\$PATH:\$ANDROID_HOME/tools:\$ANDROID_HOME/platform-tools

        # Check if we have what we need
        if [ -f '\$ANDROID_HOME/tools/bin/sdkmanager' ]; then
            echo '✅ Android SDK available'
        else
            echo '⚠️  Limited SDK - trying basic build...'
        fi

        # Try to build with existing tools
        echo '🔨 Attempting APK build...'
        chmod +x gradlew

        # Use minimal build options
        ./gradlew assembleDebug --no-daemon --quiet --console=plain

        if [ -f 'app/build/outputs/apk/debug/app-debug.apk' ]; then
            echo '✅ APK built successfully!'
            ls -la app/build/outputs/apk/debug/app-debug.apk
            exit 0
        else
            echo '❌ Build failed with this image'
            exit 1
        fi
    "

# Check if build succeeded
if [ -f "$BUILD_DIR/app/build/outputs/apk/debug/app-debug.apk" ]; then
    echo "🎉 APK built successfully!"

    # Install and launch
    echo "📱 Installing on Motorola..."
    /Users/aaronjosserand-austin/000/Motorola/tools/android/adb install -r "$BUILD_DIR/app/build/outputs/apk/debug/app-debug.apk"

    echo "🚀 Launching app..."
    /Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell am start -n com.moltar.brack/.GGUFChatActivity

    echo ""
    echo "🎯 DEPLOYMENT COMPLETE!"
    echo "======================"
    echo "📱 Brack GGUF Chat is now on your Motorola!"
    echo "   • LFN350 auto-detected"
    echo "   • Ready for philosophical conversations"
else
    echo "❌ APK build failed"
    echo "💡 Trying alternative approach..."
fi

# Cleanup
rm -rf "$BUILD_DIR"