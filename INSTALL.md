# Installation Guide

Complete installation and setup instructions for the Moltar research platform.

## Table of Contents

- [System Requirements](#system-requirements)
- [One-Click Installation](#one-click-installation)
- [Manual Installation](#manual-installation)
- [Device Setup](#device-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **Operating System**: macOS 12+, Ubuntu 20.04+, Windows 10+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Network**: Stable internet connection for model downloads

### Recommended Requirements
- **Operating System**: macOS 13+, Ubuntu 22.04+
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD for model storage
- **Network**: High-speed internet (100Mbps+) for large model downloads

### Device Requirements
- **Motorola Device**: moto g power 5G or compatible Motorola device
- **Android Version**: Android 12+ (API level 31+)
- **Storage**: 4GB+ free space on device
- **USB Debugging**: Must be enabled

---

## One-Click Installation

### Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-org/moltar.git
cd moltar

# Run the one-click installer
./moltar_setup.sh

# Follow the interactive prompts:
# 1. Select installation type (Complete/Research/Minimal)
# 2. Choose device type (Motorola/Generic Android)
# 3. Grant USB permissions when prompted
# 4. Wait for automated setup completion
```

**What gets installed:**
- ✅ Python environment with dependencies
- ✅ Android SDK and platform tools
- ✅ Device drivers and USB configuration
- ✅ Research directory structure
- ✅ Basic AI models for testing

### Quick Setup (For Experienced Users)
```bash
# Minimal setup for existing environments
./moltar_setup.sh --quick

# Or use the command launcher
./moltar setup --quick
```

---

## Manual Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/moltar.git
cd moltar
```

### Step 2: Install System Dependencies

#### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install python@3.11 git cmake ninja

# Install Android platform tools
brew install --cask android-platform-tools
```

#### Ubuntu/Debian
```bash
# Update package index
sudo apt update

# Install required packages
sudo apt install -y python3 python3-pip git cmake ninja-build \
                    android-tools-adb android-tools-fastboot

# Install Python virtual environment
sudo apt install -y python3-venv
```

#### Windows
```bash
# Install Chocolatey if not present
# Download from https://chocolatey.org/

# Install required packages
choco install python git cmake ninja androidstudio

# Add to PATH:
# - Python Scripts directory
# - Android SDK platform-tools
```

### Step 3: Python Environment Setup
```bash
# Create virtual environment
python3 -m venv moltar_env

# Activate environment
source moltar_env/bin/activate  # Linux/macOS
# moltar_env\Scripts\activate   # Windows

# Install Python dependencies
pip install -r requirements.txt

# Install additional research dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate huggingface-hub
pip install numpy pandas matplotlib seaborn
```

### Step 4: Android Development Setup

#### Install Android SDK
```bash
# Download Android Studio
# https://developer.android.com/studio

# Or install command-line tools only
# Linux/macOS:
wget https://dl.google.com/android/repository/commandlinetools-linux-10406996_latest.zip
unzip commandlinetools-linux-10406996_latest.zip
mkdir -p ~/Android/Sdk/cmdline-tools/latest
mv cmdline-tools/* ~/Android/Sdk/cmdline-tools/latest/

# Windows: Download from Android developer site
```

#### Configure Environment Variables
```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/emulator
export PATH=$PATH:$ANDROID_HOME/platform-tools
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin

# Reload profile
source ~/.bashrc  # or ~/.zshrc
```

#### Accept Android SDK Licenses
```bash
# Accept all licenses
yes | sdkmanager --licenses
```

### Step 5: Device Connection Setup

#### Enable USB Debugging on Device
1. Go to **Settings > About Phone**
2. Tap **Build Number** 7 times to enable Developer Options
3. Go to **Settings > Developer Options**
4. Enable **USB Debugging**
5. Enable **OEM Unlocking** (optional, for advanced research)

#### Configure USB Connection
```bash
# List connected devices
adb devices

# If device not found, try:
# 1. Different USB cable
# 2. Different USB port
# 3. Restart device and computer
# 4. Check USB debugging authorization on device

# Authorize computer (when prompted on device)
adb devices  # Should now show device as authorized
```

### Step 6: Research Environment Setup
```bash
# Initialize research directories
./scripts/setup_research_environment.sh

# Configure device for research
./scripts/device/setup_research_device.sh

# Test complete setup
./scripts/test_setup.sh
```

---

## Device Setup

### Motorola Device Preparation

#### 1. Initial Device Check
```bash
# Get device information
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.product.cpu.abi

# Expected output for moto g power 5G:
# moto g power 5G - 2023
# 14
# arm64-v8a
```

#### 2. Enable Advanced Features (Optional)
```bash
# Enable wireless ADB (for cable-free development)
adb tcpip 5555

# Enable root access if available (advanced users only)
# WARNING: Root access can void warranty and cause issues
./scripts/device/enable_root.sh
```

#### 3. Storage Preparation
```bash
# Check available storage
adb shell df -h /data

# Clear cache if needed
adb shell pm clear com.android.providers.downloads
adb shell rm -rf /data/local/tmp/*
```

### Model and Data Setup

#### Download Base Models
```bash
# Download test model (fast, for verification)
./research/brack/scripts/download_lfm_model.sh LiquidAI/LFM2-350M

# Download recommended model (balanced performance)
./research/brack/scripts/download_lfm_model.sh LiquidAI/LFM2-700M
```

#### Configure Model Storage
```bash
# Set up model directories
./scripts/setup_model_storage.sh

# Verify model integrity
./scripts/verify_models.sh
```

---

## Verification

### Basic Installation Test
```bash
# Test command launcher
./moltar --version
./moltar help

# Expected output:
# Moltar Research Platform v1.0.0
# Available commands: setup, device, research, help
```

### Device Connection Test
```bash
# Test device connection
./moltar device info

# Expected output includes:
# 📱 Device: [Your Motorola model]
# 🔧 Android: [Version]
# ✅ Connection: Ready for research
```

### AI Functionality Test
```bash
# Test basic AI functionality
./research/brack/scripts/test_lfn350_philosophical.sh

# Expected output:
# ✅ Model loaded successfully
# 🤖 AI Response: [Philosophical analysis]
# 📊 Performance: [Timing metrics]
```

### Full System Test
```bash
# Run comprehensive test suite
./scripts/test_full_system.sh

# Tests include:
# - Environment configuration
# - Device communication
# - Model loading and inference
# - Performance benchmarks
# - Research environment validation
```

---

## Troubleshooting

### Common Installation Issues

#### Python Environment Issues
```bash
# If pip install fails
pip install --upgrade pip setuptools wheel

# If virtual environment issues
python3 -m venv --clear moltar_env
source moltar_env/bin/activate
pip install -r requirements.txt
```

#### Android SDK Issues
```bash
# If SDK tools not found
sdkmanager "platform-tools" "platforms;android-33"

# If ADB not working
adb kill-server
adb start-server
adb devices
```

#### Device Connection Issues
```bash
# If device not recognized
# 1. Check USB cable and port
# 2. Restart device and computer
# 3. Revoke USB debugging authorizations
# 4. Re-enable USB debugging on device

# Check device state
adb devices  # Should show device with authorization
```

#### Permission Issues
```bash
# Fix script permissions
chmod +x *.sh
chmod +x scripts/*.sh
chmod +x research/brack/scripts/*.sh

# Fix directory permissions
find . -type d -exec chmod 755 {} \;
find . -name "*.sh" -exec chmod 755 {} \;
```

### Getting Help

#### Diagnostic Tools
```bash
# Run installation diagnostics
./scripts/diagnose_installation.sh

# Generate troubleshooting report
./scripts/generate_troubleshoot_report.sh
```

#### Support Resources
- **Quick Start Guide**: `QUICK_START.md`
- **Troubleshooting**: `docs/troubleshooting.md`
- **Device Compatibility**: `HARDWARE_COMPATIBILITY.md`
- **Community Support**: GitHub Issues

---

## Post-Installation

### Environment Activation
```bash
# Always activate the environment before use
cd moltar
source moltar_env/bin/activate
./moltar help
```

### Regular Maintenance
```bash
# Update the platform
./moltar update

# Clean temporary files
./moltar clean

# Check system health
./moltar health
```

### Backup and Recovery
```bash
# Backup your setup
./scripts/backup_setup.sh

# Restore from backup
./scripts/restore_setup.sh
```

---

## Advanced Configuration

### Custom Python Environment
```bash
# Use conda instead of venv
conda create -n moltar python=3.11
conda activate moltar
pip install -r requirements.txt
```

### Custom Android SDK Location
```bash
# Use custom SDK location
export ANDROID_HOME=/opt/android-sdk
export PATH=$PATH:$ANDROID_HOME/platform-tools
```

### Development Environment Setup
```bash
# Set up for development
./scripts/setup_development_environment.sh

# Install development tools
pip install pytest black flake8 mypy
```

---

*This installation provides a complete research environment for AI on Motorola devices. For issues not covered here, see the troubleshooting guide or open a GitHub issue.*