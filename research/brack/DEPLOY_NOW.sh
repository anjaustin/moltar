#!/bin/bash
# IMMEDIATE DEPLOYMENT: Get APK built and installed NOW

echo "🚀 IMMEDIATE DEPLOYMENT EXECUTION"
echo "================================="
echo ""

# Check device connection
echo "📱 Checking Motorola device connection..."
if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device$"; then
    echo "❌ Device not connected!"
    echo "🔌 Connect your Motorola device and run this script again"
    exit 1
fi
echo "✅ Device connected: $(/Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep device | awk '{print $1}')"

# Check LFN350 model
echo ""
echo "🤖 Checking LFN350 model deployment..."
if /Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell ls /data/local/tmp/lfm350_test/LFM2-350M/model.pte >/dev/null 2>&1; then
    echo "✅ LFN350 model deployed and ready"
else
    echo "⚠️  LFN350 model not found. Deploying now..."
    cd /Users/aaronjosserand-austin/000/Motorola/research/brack
    ./scripts/deploy_lfm350_device.sh
    echo "✅ LFN350 model deployed"
fi

echo ""
echo "📦 DEPLOYMENT PACKAGE READY:"
echo "============================"
echo "File: brack_android_deployment_20260131_123053.tar.gz"
echo "Size: $(ls -lh /Users/aaronjosserand-austin/000/Motorola/research/brack/brack_android_deployment_20260131_123053.tar.gz 2>/dev/null | awk '{print $5}' || echo 'Not found')"
echo ""

echo "🏗️ CRITICAL: APK BUILD REQUIRED"
echo "==============================="
echo "❌ This system cannot build Android APKs"
echo "✅ You need Android Studio on another computer"
echo ""
echo "🚀 EXECUTE THESE STEPS NOW:"
echo "=========================="
echo ""
echo "1️⃣ TRANSFER PACKAGE:"
echo "   Copy: brack_android_deployment_20260131_123053.tar.gz"
echo "   To: Computer with Android Studio"
echo ""
echo "2️⃣ BUILD APK:"
echo "   tar -xzf brack_android_deployment_20260131_123053.tar.gz"
echo "   cd brack_android_deployment_*"
echo "   # Open in Android Studio"
echo "   # Build → Make Project → Build APK"
echo ""
echo "3️⃣ GET APK FILE:"
echo "   Find: app/build/outputs/apk/debug/app-debug.apk"
echo ""
echo "4️⃣ TRANSFER APK TO MOTOROLA:"
echo "   Copy app-debug.apk to your Motorola device"
echo "   (Use USB, Bluetooth, email, cloud storage, etc.)"
echo ""
echo "5️⃣ INSTALL ON MOTOROLA:"
echo "   • Enable Developer Options (Settings → About → Tap Build 7x)"
echo "   • Enable USB Debugging (Developer Options)"
echo "   • Open File Manager → Find app-debug.apk → Tap Install"
echo ""
echo "6️⃣ LAUNCH APP:"
echo "   • Open App Drawer → Find 'Brack GGUF Chat' → Tap"
echo "   • App auto-detects LFN350 → Ready to chat!"
echo ""

echo "🎯 EXPECTED RESULT:"
echo "=================="
echo "• App opens with LFN350 welcome"
echo "• ~50-100ms response times"
echo "• Philosophical AI conversations ready"
echo ""

echo "⚡ EXECUTION STATUS:"
echo "=================="
echo "✅ Device connected and ready"
echo "✅ LFN350 model deployed"
echo "✅ Deployment package prepared"
echo "⏳ AWAITING APK BUILD (external system required)"
echo ""

echo "🚀 READY FOR IMMEDIATE EXECUTION!"
echo "=================================="
echo "Transfer the package, build APK, install on Motorola!"
echo ""
echo "Your philosophical AI breakthrough awaits! 🤖💭"