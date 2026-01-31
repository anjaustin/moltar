#!/bin/bash
# Build and deploy Brack GGUF Android app for Motorola device

set -e

echo "🚀 Building Brack GGUF Android App"
echo "==================================="

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v java &> /dev/null; then
    echo "❌ Java not found. Please install JDK 11+"
    exit 1
fi

if ! command -v ./gradlew &> /dev/null && ! command -v gradle &> /dev/null; then
    echo "❌ Gradle not found. Please install Gradle or use gradlew"
    exit 1
fi

echo "✅ Prerequisites OK"

# Check Android SDK
if [ -z "$ANDROID_HOME" ]; then
    echo "⚠️ ANDROID_HOME not set. Using default location..."
    export ANDROID_HOME="$HOME/Android/Sdk"
fi

if [ ! -d "$ANDROID_HOME" ]; then
    echo "❌ Android SDK not found at $ANDROID_HOME"
    echo "💡 Please install Android Studio or SDK manually"
    exit 1
fi

echo "✅ Android SDK found at $ANDROID_HOME"

# Set up build environment
echo "🔧 Setting up build environment..."

# Clean previous build
./gradlew clean

# Build debug APK
echo "🔨 Building debug APK..."
./gradlew assembleDebug

if [ ! -f "app/build/outputs/apk/debug/app-debug.apk" ]; then
    echo "❌ APK build failed"
    exit 1
fi

echo "✅ APK built successfully: app/build/outputs/apk/debug/app-debug.apk"

# Check device connection
echo "📱 Checking device connection..."
if ! adb devices | grep -q "device$"; then
    echo "❌ No device connected. Please connect your Motorola device."
    echo "💡 Make sure USB debugging is enabled in Developer Options"
    exit 1
fi

echo "✅ Device connected"

# Install APK
echo "📦 Installing Brack GGUF app..."
adb install -r app/build/outputs/apk/debug/app-debug.apk

if [ $? -eq 0 ]; then
    echo "✅ App installed successfully!"
else
    echo "❌ App installation failed"
    exit 1
fi

# Launch app
echo "🚀 Launching Brack GGUF Chat app..."
adb shell am start -n com.moltar.brack/.GGUFChatActivity

echo ""
echo "🎉 SUCCESS! Brack GGUF Chat is now installed and running on your Motorola device!"
echo ""
echo "📱 App Features:"
echo "   • GGUF model inference (LFM700M/LFM1.2B)"
echo "   • Chat interface with performance monitoring"
echo "   • SpaceGhost optimization indicators"
echo "   • Real-time conversation with AI"
echo ""
echo "💡 To use the app:"
echo "   1. Open 'Brack GGUF Chat' from your app drawer"
echo "   2. The app will automatically detect deployed models"
echo "   3. Start chatting with your Liquid Foundation Model!"
echo ""
echo "📊 Expected Performance:"
echo "   • LFM700M: ~1.7 tokens/second"
echo "   • LFM1.2B: ~1 token/second"
echo "   • Smooth conversational AI experience"
echo ""
echo "🔧 Troubleshooting:"
echo "   • If models aren't found, redeploy with deploy_lfm700m_gguf.py"
echo "   • Check logs with: adb logcat | grep Brack"
echo "   • Reinstall with: adb install -r app-debug.apk"