#!/bin/bash
# Create Android app deployment package for Moltar

echo "📦 Creating Brack GGUF Android App Deployment Package"
echo "===================================================="

PACKAGE_DIR="brack_android_deployment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PACKAGE_DIR"

echo "📁 Creating deployment package: $PACKAGE_DIR"

# Copy Android app source
echo "📱 Copying Android app source..."
mkdir -p "$PACKAGE_DIR/app"
cp -r app/ "$PACKAGE_DIR/"
cp build.gradle.kts "$PACKAGE_DIR/"
cp settings.gradle.kts "$PACKAGE_DIR/"
cp gradlew* "$PACKAGE_DIR/"
cp gradle/ "$PACKAGE_DIR/" 2>/dev/null || true

# Copy documentation
echo "📚 Copying documentation..."
cp README.md "$PACKAGE_DIR/"
cp QUICK_START.md "$PACKAGE_DIR/" 2>/dev/null || true

# Create build instructions
cat > "$PACKAGE_DIR/BUILD_INSTRUCTIONS.md" << 'EOF'
# Build Instructions for Brack GGUF Android App

## Prerequisites
1. **Android Studio** (Arctic Fox 2020.3.1 or later)
2. **Android SDK** (API 31+, Build Tools 30.0.3+)
3. **Java JDK** (11+)

## Build Steps

### Option 1: Android Studio
1. Extract this package
2. Open Android Studio
3. File → Open → Select the extracted folder
4. Wait for Gradle sync
5. Build → Make Project (Ctrl+F9)
6. Build → Build Bundle(s)/APK(s) → Build APK(s)

### Option 2: Command Line
```bash
cd brack_android_deployment_*
chmod +x gradlew
./gradlew assembleDebug
```

## Install APK
1. Connect your Motorola device
2. Enable USB debugging in Developer Options
3. Copy the APK from `app/build/outputs/apk/debug/app-debug.apk`
4. Install: `adb install -r app-debug.apk`
5. Launch: `adb shell am start -n com.moltar.brack/.GGUFChatActivity`

## Model Setup
The app auto-detects deployed models:
- LFM700M-GGUF (recommended)
- LFM1.2B-GGUF (advanced)
- LFN350 (test model)

Deploy models using the provided scripts before using the app.

## Features
- GGUF model inference
- Chat interface with performance monitoring
- SpaceGhost optimization indicators
- Model auto-detection
- Real-time conversation

EOF

# Create device deployment script
cat > "$PACKAGE_DIR/deploy_to_device.sh" << 'EOF'
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

EOF

chmod +x "$PACKAGE_DIR/deploy_to_device.sh"

# Create package
echo "📦 Creating deployment archive..."
tar -czf "${PACKAGE_DIR}.tar.gz" "$PACKAGE_DIR"

echo ""
echo "🎉 DEPLOYMENT PACKAGE CREATED!"
echo "==============================="
echo "📁 Package: ${PACKAGE_DIR}.tar.gz"
echo "📏 Size: $(ls -lh "${PACKAGE_DIR}.tar.gz" | awk '{print $5}')"
echo ""
echo "📋 Contents:"
echo "  • Complete Android app source code"
echo "  • Gradle build configuration"
echo "  • Build instructions (BUILD_INSTRUCTIONS.md)"
echo "  • Device deployment script (deploy_to_device.sh)"
echo "  • Documentation and README"
echo ""
echo "🚀 NEXT STEPS:"
echo "1. Transfer ${PACKAGE_DIR}.tar.gz to a system with Android Studio"
echo "2. Extract and follow BUILD_INSTRUCTIONS.md"
echo "3. Build the APK and deploy to Motorola device"
echo "4. Open 'Brack GGUF Chat' app and chat with LFN350!"
echo ""
echo "💡 The app will automatically detect and use LFN350 as the default model"