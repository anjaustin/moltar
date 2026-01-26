# Installation Guide

Complete installation and setup guide for the moltar research repository.

## Prerequisites

### System Requirements
- **Operating System**: macOS 11.0+ (recommended), Linux, or Windows with WSL2
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for tools and research data
- **Network**: Stable internet connection for model downloads

### Required Software
- **Git**: Version control system
- **Python**: 3.8+ with pip package manager
- **Android Studio**: For Android development (optional, but recommended)
- **Android SDK/NDK**: Platform tools for device interaction

## Quick Installation

### One-Command Setup (Recommended)
```bash
# Clone repository
git clone https://github.com/anjaustin/moltar.git
cd moltar

# Run automated setup
./moltar_setup.sh
```

This will:
- ✅ Install system dependencies
- ✅ Set up Python environment
- ✅ Configure Android development tools
- ✅ Download and configure research frameworks
- ✅ Validate installation

### Manual Installation

#### Step 1: Clone Repository
```bash
git clone https://github.com/anjaustin/moltar.git
cd moltar
```

#### Step 2: Install System Dependencies
```bash
# macOS with Homebrew
brew install python@3.11 git android-platform-tools

# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip git android-tools-adb android-tools-fastboot

# Windows (WSL2 recommended)
# Install Windows Terminal, WSL2, then:
sudo apt update
sudo apt install python3 python3-pip git android-tools-adb android-tools-fastboot
```

#### Step 3: Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 4: Android Development Setup (Optional)
```bash
# Install Android Studio
# Download from: https://developer.android.com/studio

# Or install command-line tools
# Download SDK from: https://developer.android.com/studio#command-line-tools-only
# Extract to ~/Android/sdk
export ANDROID_HOME=~/Android/sdk
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools
```

## Component-Specific Installation

### Research Frameworks
```bash
# Install research methodology tools
pip install jupyterlab pandas numpy scipy matplotlib

# Install Android research tools
pip install ppadb pure-python-adb
```

### Brack LFN Deployment
```bash
cd research/brack

# Setup environment
./scripts/setup_environment.sh

# Download LFN model
./scripts/download_lfm_model.sh LiquidAI/LFM2-350M
```

## Device Setup

### Motorola Device Preparation
```bash
# Connect device via USB
./scripts/device/connect_device.sh

# Setup research environment on device
./scripts/device/setup_research_device.sh
```

### USB Debugging Setup
1. Enable Developer Options on device
2. Enable USB Debugging
3. Accept RSA key authorization
4. Verify connection: `adb devices`

## Verification

### System Verification
```bash
# Check Python environment
python3 --version
pip --version

# Check Android tools
adb version
fastboot --version

# Check Git
git --version
```

### Repository Verification
```bash
# Check repository status
git status

# Run basic tests
./moltar_setup.sh --help

# Check documentation
ls *.md docs/
```

### Device Verification
```bash
# Check device connection
./scripts/device/connect_device.sh check

# Verify research environment
./scripts/device/setup_research_device.sh verify
```

## Troubleshooting

### Common Installation Issues

#### Python Virtual Environment Issues
```bash
# If venv creation fails
python3 -m pip install --user virtualenv
python3 -m virtualenv venv

# Activate environment
source venv/bin/activate
```

#### Android SDK Issues
```bash
# If Android tools not found
export ANDROID_HOME=~/Android/Sdk
echo 'export ANDROID_HOME=~/Android/Sdk' >> ~/.bashrc
source ~/.bashrc

# Add to PATH
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools
```

#### Device Connection Issues
```bash
# Restart ADB
adb kill-server
adb start-server

# Check USB connection
lsusb  # Linux
system_profiler SPUSBDataType  # macOS

# Try different USB port/cable
# Reboot device and computer
```

#### Permission Issues
```bash
# macOS ADB permissions
sudo chown -R $USER:admin /usr/local/bin/adb
sudo chmod +x /usr/local/bin/adb

# Linux udev rules for Android devices
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="22b8", MODE="0666", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/51-android.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Environment Configuration

### Shell Configuration
Add to `~/.bashrc`, `~/.zshrc`, or `~/.profile`:
```bash
# Moltar environment
export MOLTA_ROOT="/path/to/moltar"
export PATH="$PATH:$MOLTA_ROOT"

# Python virtual environment
export VIRTUAL_ENV="$MOLTA_ROOT/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Android development
export ANDROID_HOME="$HOME/Android/Sdk"
export PATH="$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools"
```

### IDE Configuration

#### VS Code
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "android.sdkPath": "~/Android/Sdk"
}
```

#### Android Studio
- Import `research/brack/src/` as Android project
- Set SDK location in preferences
- Install required SDK components

## Updating Installation

### Update Repository
```bash
git pull origin main
```

### Update Dependencies
```bash
# Update Python packages
pip install -r requirements.txt --upgrade

# Update Android SDK
sdkmanager --update
```

### Update Device Environment
```bash
# Update device research environment
./scripts/device/setup_research_device.sh
```

## Advanced Configuration

### Custom Model Installation
```bash
# Install custom LFN models
cd research/brack
./scripts/download_lfm_model.sh your-custom-model

# Update configuration
nano config/lfm_config.json
```

### Development Environment
```bash
# Enable development mode
export MOLTA_DEV=true

# Enable verbose logging
export MOLTA_LOG_LEVEL=DEBUG

# Custom research directories
export MOLTA_DATA_DIR="/custom/data/path"
```

## Getting Help

### Documentation Resources
- **Main Documentation**: `README.md`
- **Research Methodology**: `docs/methodology/RESEARCH_METHODOLOGY.md`
- **Brack Deployment**: `research/brack/docs/DEPLOYMENT_GUIDE.md`
- **Troubleshooting**: Check logs in `logs/` directory

### Community Support
- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for general questions
- **Documentation**: Wiki pages for detailed guides

---

*This installation guide ensures smooth setup of the moltar research environment across different systems and use cases.*