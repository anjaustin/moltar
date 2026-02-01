#!/bin/bash
# Deploy built APK to Motorola device

echo "🚀 Deploying Brack GGUF App to Motorola Device"

# Check APK exists
if [ ! -f "app/build/outputs/apk/debug/app-debug.apk" ]; then
    echo "❌ APK not found. Build the app first:"
    echo "  ./gradlew assembleDebug"
    exit 1
fi

# Check device connected
echo "📱 Checking device connection..."
if ! adb devices | grep -q "device$"; then
    echo "❌ No device connected"
    echo "1. Connect Motorola device via USB"
    echo "2. Enable USB debugging in Settings > Developer Options"
    echo "3. Accept USB debugging authorization"
    exit 1
fi

echo "✅ Device connected"

# Install APK
echo "📦 Installing Brack GGUF app..."
adb install -r app/build/outputs/apk/debug/app-debug.apk

if [ $? -eq 0 ]; then
    echo "✅ App installed successfully!"
else
    echo "❌ Installation failed"
    exit 1
fi

# Launch app
echo "🚀 Launching app..."
adb shell am start -n com.moltar.brack/.GGUFChatActivity

echo ""
echo "🎉 SUCCESS!"
echo "📱 Open 'Brack GGUF Chat' from your app drawer"
echo "💬 The app will auto-detect LFN350 or other deployed models"
echo ""
echo "📊 Expected: ~50-100ms responses with LFN350 test model"

