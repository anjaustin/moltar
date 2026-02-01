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

