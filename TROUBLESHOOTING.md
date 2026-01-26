# Troubleshooting Guide

Common issues and solutions for the moltar research repository.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Device Connection Problems](#device-connection-problems)
- [Build and Compilation Errors](#build-and-compilation-errors)
- [Model Download Issues](#model-download-issues)
- [Performance Problems](#performance-problems)
- [Research Environment Issues](#research-environment-issues)

## Installation Issues

### Python Virtual Environment Problems

**Error**: `python3: command not found`
```bash
# Check Python installation
which python3
python3 --version

# Install Python (macOS)
brew install python@3.11

# Install Python (Ubuntu/Debian)
sudo apt install python3 python3-pip python3-venv
```

**Error**: `virtualenv: command not found`
```bash
# Install virtualenv
pip3 install virtualenv

# Or use built-in venv
python3 -m venv venv
```

### Permission Denied Errors

**Error**: `Permission denied` when running scripts
```bash
# Make scripts executable
chmod +x *.sh
chmod +x scripts/**/*.sh

# Or run with bash
bash ./moltar_setup.sh
```

### Android SDK Not Found

**Error**: `Android SDK not found`
```bash
# Set Android SDK path
export ANDROID_HOME="$HOME/Android/Sdk"
export PATH="$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools"

# Add to shell profile
echo 'export ANDROID_HOME="$HOME/Android/Sdk"' >> ~/.zshrc
echo 'export PATH="$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools"' >> ~/.zshrc
```

## Device Connection Problems

### ADB Device Not Found

**Error**: `no devices/emulators found`
```bash
# Check USB connection
lsusb  # Linux
system_profiler SPUSBDataType  # macOS

# Restart ADB
adb kill-server
adb start-server
adb devices

# Check device USB mode
# Settings → Developer Options → Default USB Configuration → File Transfer
```

### Device Unauthorized

**Error**: `device unauthorized`
```bash
# On device: Accept USB debugging authorization dialog
# Check "Always allow from this computer"

# Revoke and re-authorize
adb kill-server
# Unplug/replug device
adb devices
# Accept new authorization dialog
```

### Motorola-Specific Issues

**Motorola device not recognized**
```bash
# Enable OEM Unlocking (if bootloader locked)
# Settings → Developer Options → OEM Unlocking

# Try different USB ports/cables
# Restart device in safe mode
# Check device storage space (>1GB free)
```

### Root Access Issues

**Root check fails**
```bash
# Verify Magisk installation
adb shell su -c "whoami"

# Reboot device after Magisk installation
# Check Magisk Manager app
# Verify Magisk version compatibility
```

## Build and Compilation Errors

### Gradle Build Failures

**Error**: `Gradle build failed`
```bash
# Clean and rebuild
cd research/brack/src
./gradlew clean
./gradlew build

# Check Java version
java -version
javac -version

# Update Gradle wrapper
./gradlew wrapper --gradle-version 8.2
```

### Android Manifest Issues

**Error**: `AndroidManifest.xml not found`
```bash
# Verify file structure
ls -la research/brack/src/main/

# Create missing manifest
mkdir -p research/brack/src/main
# Copy from template or recreate
```

### ExecuTorch Integration Problems

**Error**: `ExecuTorch library not found`
```bash
# Check Gradle dependencies
grep "executorch" research/brack/config/build.gradle.kts

# Verify Maven repository
# Ensure internet connection for dependency download
# Clear Gradle cache: ./gradlew cleanBuildCache
```

## Model Download Issues

### HuggingFace Access Problems

**Error**: `Repository not found`
```bash
# Verify model name
./scripts/download_lfm_model.sh  # Shows available models

# Check internet connection
ping huggingface.co

# Login to HuggingFace (if required)
huggingface-cli login
```

### Disk Space Issues

**Error**: `No space left on device`
```bash
# Check available space
df -h

# Clean up space
rm -rf ~/Downloads/*.tmp
docker system prune -a  # If using Docker

# Use smaller model
./scripts/download_lfm_model.sh LiquidAI/LFM2-350M  # Smaller than 2B models
```

### Git LFS Issues

**Error**: `git lfs: command not found`
```bash
# Install Git LFS
brew install git-lfs  # macOS
sudo apt install git-lfs  # Ubuntu

# Initialize LFS
git lfs install
```

## Performance Problems

### High Latency Issues

**Symptom**: Response time >200ms
```bash
# Check device specifications
adb shell getprop ro.product.model
adb shell getprop ro.build.version.sdk

# Verify DSP acceleration
# Check Snapdragon 480 status
# Monitor CPU usage during inference
```

### Memory Usage Problems

**Symptom**: App crashes with OOM
```bash
# Monitor memory usage
adb shell dumpsys meminfo com.moltar.brack

# Reduce model size
# Use quantization in config
# Check available RAM: adb shell cat /proc/meminfo
```

### Battery Drain Issues

**Symptom**: Excessive battery usage
```bash
# Monitor battery stats
adb shell dumpsys battery

# Check background processes
adb shell ps | grep brack

# Optimize inference frequency
# Use DSP acceleration for power efficiency
```

## Research Environment Issues

### Script Execution Problems

**Error**: `command not found`
```bash
# Source environment
cd /path/to/moltar
source venv/bin/activate

# Add to PATH
export PATH="$PWD:$PATH"
```

### Configuration File Issues

**Error**: `config file not found`
```bash
# Verify file locations
ls -la config/
ls -la research/brack/config/

# Recreate missing configs
cp config/lfm_config.json research/brack/config/
```

### Log File Problems

**Symptom**: No logs generated
```bash
# Check log directory
ls -la logs/

# Enable logging in config
# Check file permissions
chmod 755 logs/
```

## Advanced Troubleshooting

### Debug Mode

Enable verbose logging:
```bash
export MOLTA_LOG_LEVEL=DEBUG
export MOLTA_DEBUG=true

# Re-run with debugging
./moltar_setup.sh
```

### System Diagnostics

Run comprehensive diagnostics:
```bash
# System information
uname -a
sw_vers  # macOS
lsb_release -a  # Linux

# Python diagnostics
python3 -c "import sys; print(sys.version)"
pip list | grep -E "(torch|executorch|huggingface)"

# Android diagnostics
adb version
fastboot --version
```

### Network Diagnostics

Check connectivity issues:
```bash
# Test basic connectivity
ping 8.8.8.8

# Test HuggingFace access
curl -I https://huggingface.co

# Test GitHub access
curl -I https://github.com
```

### Log Analysis

Analyze error logs:
```bash
# Check setup logs
tail -50 logs/setup_*.log

# Check device logs
adb logcat | grep -i brack

# Check system logs
dmesg | tail -20  # Linux
log show --last 1h | grep -i adb  # macOS
```

## Getting Help

### Quick Diagnosis
Run the diagnostic script:
```bash
./scripts/device/connect_device.sh check
./research/brack/scripts/test_brack_deployment.sh
```

### Support Resources
- **Documentation**: Check `docs/` directory
- **Issues**: GitHub Issues for bugs
- **Discussions**: GitHub Discussions for questions
- **Logs**: Check `logs/` directory for error details

### Emergency Recovery
Reset to clean state:
```bash
# Remove problematic installations
rm -rf venv/
rm -rf research/brack/models/
rm -rf research/brack/src/app/build/

# Reinstall
./moltar_setup.sh
```

---

*This troubleshooting guide addresses the most common issues encountered during moltar setup and usage. For issues not covered here, please check the logs and create a GitHub issue with detailed information.*