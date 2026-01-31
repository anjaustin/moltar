#!/bin/bash
# Install pre-built Brack GGUF APK to Motorola device

echo "📦 Installing Brack GGUF Android App to Motorola"
echo "================================================"

# Check if APK file is provided as argument
if [ $# -eq 0 ]; then
    # Look for APK in current directory
    APK_FILE=$(find . -name "*.apk" -type f | head -1)
    if [ -z "$APK_FILE" ]; then
        echo "❌ No APK file found"
        echo "Usage: $0 <path-to-apk-file>"
        echo "Or place APK file in current directory"
        exit 1
    fi
else
    APK_FILE="$1"
fi

if [ ! -f "$APK_FILE" ]; then
    echo "❌ APK file not found: $APK_FILE"
    exit 1
fi

echo "📱 APK File: $APK_FILE"
echo "📏 Size: $(ls -lh "$APK_FILE" | awk '{print $5}')"

# Check device connection
echo "📱 Checking Motorola device connection..."
if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device$"; then
    echo "❌ No device connected"
    echo ""
    echo "🔧 TROUBLESHOOTING:"
    echo "1. Connect Motorola device via USB cable"
    echo "2. Enable Developer Options:"
    echo "   - Settings > About Phone > Build Number (tap 7 times)"
    echo "3. Enable USB Debugging:"
    echo "   - Settings > Developer Options > USB Debugging"
    echo "4. Accept USB debugging authorization on device"
    echo ""
    echo "Then run this script again"
    exit 1
fi

echo "✅ Motorola device connected"

# Get device info
DEVICE_MODEL=$(/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell getprop ro.product.model 2>/dev/null || echo "Motorola Device")
echo "📱 Device: $DEVICE_MODEL"

# Install APK
echo "📦 Installing Brack GGUF Chat app..."
/Users/aaronjosserand-austin/000/Motorola/tools/android/adb install -r "$APK_FILE"

if [ $? -eq 0 ]; then
    echo "✅ App installed successfully!"
else
    echo "❌ App installation failed"
    echo ""
    echo "🔧 POSSIBLE FIXES:"
    echo "1. Uninstall existing app first:"
    echo "   adb uninstall com.moltar.brack"
    echo "2. Check device storage space"
    echo "3. Ensure APK is not corrupted"
    exit 1
fi

# Launch app
echo "🚀 Launching Brack GGUF Chat app..."
/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell am start -n com.moltar.brack/.GGUFChatActivity

echo ""
echo "🎉 SUCCESS! Brack GGUF Chat is now installed and running!"
echo ""
echo "📱 APP FEATURES:"
echo "   • Auto-detects LFN350 test model (default)"
echo "   • Chat interface with performance monitoring"
echo "   • ~50-100ms responses with LFN350"
echo "   • SpaceGhost optimization indicators"
echo ""
echo "💬 TRY ASKING:"
echo "   'Hypothetically, might reflective recursion be a function of awareness?'"
echo ""
echo "📊 The app will provide thoughtful responses appropriate for a 350M parameter model"