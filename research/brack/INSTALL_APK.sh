#!/bin/bash
# Install APK on Motorola device once built

echo "📦 INSTALLING BRACK GGUF CHAT APK"
echo "================================="

# Check if APK file is provided
APK_FILE="$1"
if [ -z "$APK_FILE" ]; then
    echo "❌ Usage: $0 <path-to-app-debug.apk>"
    echo "Example: $0 /path/to/app-debug.apk"
    exit 1
fi

if [ ! -f "$APK_FILE" ]; then
    echo "❌ APK file not found: $APK_FILE"
    exit 1
fi

echo "📱 APK File: $APK_FILE"
echo "📏 Size: $(ls -lh "$APK_FILE" | awk '{print $5}')"

# Check device connection
echo ""
echo "📱 Checking Motorola device..."
if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device$"; then
    echo "❌ Device not connected!"
    echo "🔌 Connect Motorola device via USB"
    exit 1
fi
echo "✅ Device connected"

# Install APK
echo ""
echo "📦 Installing Brack GGUF Chat app..."
/Users/aaronjosserand-austin/000/Motorola/tools/android/adb install -r "$APK_FILE"

if [ $? -eq 0 ]; then
    echo "✅ INSTALLATION SUCCESSFUL!"
else
    echo "❌ Installation failed"
    echo ""
    echo "🔧 TROUBLESHOOTING:"
    echo "• Enable USB Debugging on device"
    echo "• Allow installation from unknown sources"
    echo "• Check device storage space"
    echo "• Try: adb uninstall com.moltar.brack && $0 $APK_FILE"
    exit 1
fi

# Launch app
echo ""
echo "🚀 Launching Brack GGUF Chat app..."
/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell am start -n com.moltar.brack/.GGUFChatActivity

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================"
echo ""
echo "📱 On your Motorola device:"
echo "   • Open App Drawer"
echo "   • Find 'Brack GGUF Chat'"
echo "   • Tap to open"
echo ""
echo "🤖 App will:"
echo "   • Auto-detect LFN350 model"
echo "   • Show welcome messages"
echo "   • Provide ~50-100ms responses"
echo ""
echo "💬 Try asking:"
echo "   'Hypothetically, might reflective recursion be a function of awareness?'"
echo ""
echo "⚡ Your Motorola is now a philosophical AI device!"