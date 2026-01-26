#!/bin/bash

# Brack Environment Setup Script
# Sets up development environment for LFN deployment on Motorola devices

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
MOLTAR_ROOT="$PROJECT_ROOT/../.."

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') - $1"
}

# Check system prerequisites
check_prerequisites() {
    log_info "Checking system prerequisites..."

    # Check macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_error "This setup requires macOS"
        exit 1
    fi
    log_success "macOS detected"

    # Check command line tools
    if ! xcode-select -p >/dev/null 2>&1; then
        log_warning "Xcode command line tools not found"
        log_info "Installing Xcode command line tools..."
        xcode-select --install
        log_info "Please complete Xcode installation and re-run this script"
        exit 1
    fi
    log_success "Xcode command line tools available"

    # Check Homebrew
    if ! command -v brew >/dev/null 2>&1; then
        log_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true
    fi
    log_success "Homebrew available"

    # Check Python
    if ! command -v python3 >/dev/null 2>&1; then
        log_info "Installing Python 3..."
        brew install python@3.11
    fi
    log_success "Python 3 available: $(python3 --version)"

    # Check Android Studio or Android CLI tools
    if ! command -v adb >/dev/null 2>&1; then
        log_warning "ADB not found in PATH"
        log_info "Installing Android platform tools..."
        brew install --cask android-platform-tools
    fi
    log_success "Android platform tools available"
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python environment..."

    cd "$PROJECT_ROOT"

    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Created Python virtual environment"
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "Installed Python dependencies"
    else
        log_info "Creating requirements.txt..."
        cat > requirements.txt << EOF
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
executorch>=0.4.0
numpy>=1.24.0
requests>=2.31.0
tqdm>=4.65.0
pyyaml>=6.0
EOF
        pip install -r requirements.txt
        log_success "Created and installed Python dependencies"
    fi
}

# Setup Android development environment
setup_android_env() {
    log_info "Setting up Android development environment..."

    # Check Android Studio
    if [ ! -d "/Applications/Android Studio.app" ]; then
        log_warning "Android Studio not found"
        log_info "Please download Android Studio from:"
        log_info "https://developer.android.com/studio"
        log_info "Or install via Homebrew: brew install --cask android-studio"
        read -p "Press Enter after installing Android Studio..."
    fi

    # Setup Android SDK (if not already done)
    if [ -z "$ANDROID_HOME" ]; then
        ANDROID_HOME="$HOME/Library/Android/sdk"
        if [ ! -d "$ANDROID_HOME" ]; then
            log_warning "Android SDK not found"
            log_info "Android Studio will install SDK automatically on first run"
            log_info "Please complete Android Studio setup and SDK installation"
            read -p "Press Enter after completing Android Studio setup..."
        fi
        echo "export ANDROID_HOME=\"$ANDROID_HOME\"" >> ~/.zshrc
        export ANDROID_HOME="$ANDROID_HOME"
    fi
    log_success "Android SDK configured: $ANDROID_HOME"

    # Setup Android NDK
    if [ -z "$ANDROID_NDK_HOME" ]; then
        NDK_PATH="$ANDROID_HOME/ndk/$(ls $ANDROID_HOME/ndk 2>/dev/null | head -1)"
        if [ -d "$NDK_PATH" ]; then
            export ANDROID_NDK_HOME="$NDK_PATH"
            echo "export ANDROID_NDK_HOME=\"$ANDROID_NDK_HOME\"" >> ~/.zshrc
            log_success "Android NDK configured: $ANDROID_NDK_HOME"
        else
            log_warning "Android NDK not found"
            log_info "Android Studio will install NDK automatically"
            log_info "Or install manually from SDK Manager"
        fi
    fi

    # Verify ADB connection
    log_info "Verifying ADB connection..."
    if adb devices | grep -q "device$"; then
        log_success "ADB device connected"
    else
        log_warning "No ADB device connected"
        log_info "Please connect your Motorola device and enable USB debugging"
        log_info "Run this script again after device connection"
    fi
}

# Setup Liquid.ai integration
setup_liquid_integration() {
    log_info "Setting up Liquid.ai integration..."

    cd "$PROJECT_ROOT"

    # Create models directory
    mkdir -p models

    # Download Liquid.ai SDK (placeholder - actual SDK would be downloaded)
    log_info "Liquid.ai LFM SDK integration"
    log_info "Note: Download LFM SDK from https://docs.liquid.ai/"
    log_info "Place SDK files in models/ directory"

    # Create SDK integration script
    cat > scripts/download_lfm_model.sh << 'EOF'
#!/bin/bash
# Download Liquid.ai LFM model script

MODEL_NAME="${1:-lfm-2b-chat}"
MODEL_DIR="models"

echo "Downloading Liquid.ai $MODEL_NAME model..."
echo "Note: This is a placeholder script"
echo "Actual model downloads require Liquid.ai API access"
echo ""
echo "Manual steps:"
echo "1. Visit https://docs.liquid.ai/lfm/models"
echo "2. Download $MODEL_NAME model files"
echo "3. Extract to $MODEL_DIR/ directory"
echo "4. Update config/lfm_config.json with model paths"
EOF

    chmod +x scripts/download_lfm_model.sh
    log_success "Created LFM model download script"
}

# Setup project structure
setup_project_structure() {
    log_info "Setting up project structure..."

    cd "$PROJECT_ROOT"

    # Create Android project structure
    mkdir -p src/main/{java/com/moltar/brack,res,assets}
    mkdir -p src/main/{cpp,res/{layout,values,drawable}}
    mkdir -p src/androidTest/java

    # Create basic Android manifest
    cat > src/main/AndroidManifest.xml << EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.moltar.brack">

    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/AppTheme">

        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

    </application>

</manifest>
EOF

    # Create basic Gradle build files
    cat > config/build.gradle.kts << 'EOF'
plugins {
    id("com.android.application") version "8.2.0"
    id("org.jetbrains.kotlin.android") version "1.9.10"
    kotlin("plugin.serialization") version "1.9.10"
}

android {
    namespace = "com.moltar.brack"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.moltar.brack"
        minSdk = 31
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.10.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // ExecuTorch for on-device inference
    implementation("org.pytorch:executorch-android:0.4.0")

    // Liquid.ai LFM integration (placeholder - replace with actual SDK)
    // implementation("ai.liquid:lfm-android:latest")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
EOF

    cat > config/settings.gradle.kts << 'EOF'
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // Add Liquid.ai Maven repository when available
        // maven { url = uri("https://maven.liquid.ai/releases") }
    }
}

rootProject.name = "Brack"
include(":app")
EOF

    log_success "Created Android project structure and build files"
}

# Create build and deployment scripts
create_build_scripts() {
    log_info "Creating build and deployment scripts..."

    cd "$PROJECT_ROOT"

    # Build debug script
    cat > scripts/build_debug.sh << 'EOF'
#!/bin/bash
# Build debug APK for Brack

set -e

echo "Building Brack debug APK..."

# Activate Python environment
source ../venv/bin/activate

# Build Android APK
cd src
./gradlew assembleDebug

echo "Debug APK built: app/build/outputs/apk/debug/app-debug.apk"
EOF

    # Deploy script
    cat > scripts/deploy_device.sh << 'EOF'
#!/bin/bash
# Deploy Brack to connected Motorola device

set -e

APK_PATH="src/app/build/outputs/apk/debug/app-debug.apk"

if [ ! -f "$APK_PATH" ]; then
    echo "APK not found. Run build_debug.sh first."
    exit 1
fi

echo "Deploying Brack to device..."

# Install APK
adb install -r "$APK_PATH"

# Grant permissions if needed
adb shell pm grant com.moltar.brack android.permission.INTERNET

echo "Brack deployed successfully!"
echo "Launch the app on your device to test LFN chat functionality."
EOF

    # Model download placeholder
    cat > scripts/download_lfm_model.sh << 'EOF'
#!/bin/bash
# Download Liquid.ai LFM model

MODEL_NAME="${1:-lfm-2b-chat}"

echo "=== Liquid.ai LFM Model Download ==="
echo "Model: $MODEL_NAME"
echo ""
echo "⚠️  MANUAL DOWNLOAD REQUIRED ⚠️"
echo ""
echo "Due to licensing and access restrictions, LFM models must be downloaded manually:"
echo ""
echo "1. Visit: https://docs.liquid.ai/lfm/models"
echo "2. Sign up for Liquid.ai developer access"
echo "3. Download $MODEL_NAME model files"
echo "4. Extract to models/ directory"
echo "5. Update config/lfm_config.json"
echo ""
echo "Supported models:"
echo "  - lfm-2b-chat (recommended for mobile)"
echo "  - lfm-7b-chat"
echo "  - lfm-40b-chat"
echo ""
echo "Note: Models are large (2-40GB) and require significant bandwidth."
EOF

    chmod +x scripts/*.sh
    log_success "Created build and deployment scripts"
}

# Main setup function
main() {
    echo -e "${BLUE}🔬 BRACK ENVIRONMENT SETUP${NC}"
    echo "============================"
    echo ""

    log_info "Setting up Brack research environment for LFN deployment..."

    check_prerequisites
    setup_python_env
    setup_android_env
    setup_liquid_integration
    setup_project_structure
    create_build_scripts

    echo ""
    echo -e "${GREEN}🎉 BRACK ENVIRONMENT SETUP COMPLETE!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Download LFM model: ./scripts/download_lfm_model.sh"
    echo "2. Build Android app: ./scripts/build_debug.sh"
    echo "3. Deploy to device: ./scripts/deploy_device.sh"
    echo ""
    echo "For development:"
    echo "- Open src/ in Android Studio"
    echo "- Connect Motorola device via ADB"
    echo "- Run app and test LFN chat functionality"
    echo ""
    echo -e "${BLUE}Happy researching with Brack! 🔬${NC}"
}

main