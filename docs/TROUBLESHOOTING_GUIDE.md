# Troubleshooting Guide: LFN Deployment Issues

## Overview

This guide provides solutions for common issues encountered during Liquid AI Foundation Model (LFN) deployment with SpaceGhost optimizations on Motorola Snapdragon 480 devices.

## Quick Diagnosis

### Run Diagnostic Script
```bash
cd research/brack
./scripts/diagnose_deployment.sh
```

This automated diagnostic checks:
- Device connectivity and authorization
- SpaceGhost optimization status
- Model file integrity
- Performance metrics
- System resource usage

## Device Connection Issues

### Problem: Device Not Detected
```
$ adb devices
List of devices attached
(no devices)
```

**Solutions:**
```bash
# 1. Check USB connection
ls /dev/tty.usb*  # macOS
ls /dev/ttyACM*   # Linux

# 2. Restart ADB
adb kill-server
adb start-server

# 3. Check device USB debugging
# Settings > Developer Options > USB Debugging (enable)

# 4. Try different USB ports/cables
# Some cables are charging-only

# 5. Check device screen unlock
# Device must be unlocked for ADB authorization
```

### Problem: Device Unauthorized
```
$ adb devices
List of devices attached
XXXXXXX unauthorized
```

**Solutions:**
```bash
# 1. Revoke USB debugging authorizations
# Device: Settings > Developer Options > Revoke USB debugging authorizations

# 2. Reconnect device
adb kill-server && adb start-server

# 3. Accept authorization prompt on device
# Look for "Allow USB debugging?" dialog

# 4. Check device screen is unlocked
# Authorization prompt only appears on unlocked device
```

### Problem: Device Connection Drops
```
$ adb shell ls
error: device offline
```

**Solutions:**
```bash
# 1. Check USB connection stability
adb devices  # Should show "device" not "offline"

# 2. Restart device USB service
adb usb

# 3. Reconnect USB cable
# Try different ports if issue persists

# 4. Check device battery level
# Low battery can cause connection instability

# 5. Disable battery optimization for ADB
# Settings > Apps > Android System > Don't optimize
```

## SpaceGhost Optimization Issues

### Problem: SpaceGhost Optimizations Not Applied
```
$ adb shell /data/local/tmp/spaceghost_demo/show_achievements.sh
# Missing or incorrect output
```

**Symptoms:**
- Performance matches baseline, not optimized
- No DSP delegation in logs
- Latency >200ms for LFM2-350M

**Solutions:**
```bash
# 1. Verify cleanup pass is available
cd research/spaceghost
python -c "from patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline; print('✅ Available')"

# 2. Check model export pipeline
cd research/brack
./scripts/test_brack_deployment.sh --verbose

# 3. Verify quantization applied
python -c "
import torch
from torch.export import export
from research.brack.src.main.python.model import create_lfm_model

model = create_lfm_model('LFM2-350M')
sample_input = torch.randn(1, 512)  # Adjust for model input
exported = export(model, (sample_input,))
print('✅ Model exports successfully')
"

# 4. Check SpaceGhost integration
./scripts/build_debug_spaceghost.sh --check-integrity
```

### Problem: Quantization Chain Issues
```
Exception: RuntimeError: Didn't find engine for operation quantized::linear_prepack
```

**Symptoms:**
- REQ-XNN-002 validation fails
- Model export fails during quantization

**Solutions:**
```bash
# 1. Check PyTorch quantization backend
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('Quantization engines:', torch.backends.quantized.supported_engines())
"

# 2. Use CPU quantization fallback
export PYTORCH_QUANTIZATION_ENGINE=fbgemm
# or
export PYTORCH_QUANTIZATION_ENGINE=qnnpack

# 3. Test quantization separately
python -c "
import torch
from torch.ao.quantization import quantize_dynamic

model = torch.nn.Linear(10, 5)
quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print('✅ Quantization works')
"
```

### Problem: Ghost Partition Bug Returns
```
Partitioning completed but no delegate operations created
```

**Symptoms:**
- MaxPool2d operations not delegated to DSP
- XNNPack partitioner creates empty delegate nodes

**Solutions:**
```bash
# 1. Verify ExecuTorch version compatibility
python -c "
import executorch
print('ExecuTorch version:', executorch.__version__)
# Should be 0.4.0+ for SpaceGhost compatibility
"

# 2. Check XNNPack backend availability
python -c "
from executorch.backends.xnnpack import XnnpackPartitioner
print('✅ XNNPack partitioner available')
"

# 3. Test partitioning directly
cd research/spaceghost
python -c "
from executorch.backends.xnnpack import XnnpackPartitioner
from torch.fx import GraphModule
import torch

# Create simple test graph
graph = torch.fx.Graph()
input_node = graph.placeholder('x')
output_node = graph.output(input_node)
gm = GraphModule({}, graph)

# Try partitioning
partitioner = XnnpackPartitioner()
try:
    partitioned = gm.to_backend(partitioner)
    print('✅ Partitioning works')
except Exception as e:
    print('❌ Partitioning failed:', e)
"
```

## Model Deployment Issues

### Problem: Model Download Fails
```
Error: Repository LiquidAI/LFM2-350M not found
```

**Solutions:**
```bash
# 1. Check HuggingFace access
huggingface-cli whoami  # Should show logged-in user

# 2. Login to HuggingFace
huggingface-cli login

# 3. Check model availability
curl -s https://huggingface.co/LiquidAI/LFM2-350M | head -20

# 4. Use alternative download method
cd research/brack
./scripts/download_lfm_model.sh LiquidAI/LFM2-350M --force --direct
```

### Problem: Model Loading Fails on Device
```
Error: Failed to load LFM model
```

**Symptoms:**
- App crashes on startup
- Model file corrupted or incompatible

**Solutions:**
```bash
# 1. Verify model file integrity
adb shell ls -lh /data/data/com.moltar.brack/files/models/*.pte

# 2. Check file permissions
adb shell ls -la /data/data/com.moltar.brack/files/models/

# 3. Re-download model
cd research/brack
rm -rf models/LFM2-350M/
./scripts/download_lfm_model.sh LiquidAI/LFM2-350M --force

# 4. Check available storage
adb shell df /data

# 5. Clear app data and reinstall
adb shell pm clear com.moltar.brack
./scripts/deploy_device_spaceghost.sh
```

### Problem: Memory Issues During Inference
```
Error: RuntimeException: Out of memory during inference
```

**Symptoms:**
- App crashes during model execution
- Performance degrades over time

**Solutions:**
```bash
# 1. Check device memory
adb shell dumpsys meminfo com.moltar.brack

# 2. Monitor memory usage during inference
adb shell dumpsys meminfo | grep -A 10 "com.moltar.brack"

# 3. Reduce model size (use LFM2-350M instead of larger models)
cd research/brack
./scripts/download_lfm_model.sh LiquidAI/LFM2-350M --replace

# 4. Enable memory optimization
# In app config, set memory_optimization=true

# 5. Clear device cache
adb shell pm trim-caches 256M
```

## Performance Issues

### Problem: Latency Above Targets
```
Measured: 350ms, Target: <200ms
```

**Solutions:**
```bash
# 1. Verify SpaceGhost optimizations active
adb shell /data/local/tmp/spaceghost_demo/show_achievements.sh

# 2. Check DSP utilization
adb logcat | grep -i "xnnpack\|delegate" | tail -20

# 3. Monitor CPU usage
adb shell dumpsys cpuinfo | grep -A 5 "com.moltar.brack"

# 4. Test with smaller model
cd research/brack
./scripts/download_lfm_model.sh LiquidAI/LFM2-350M --replace

# 5. Check thermal throttling
adb shell dumpsys battery | grep temperature
```

### Problem: Battery Drain Excessive
```
Battery usage: 12%/hour, Target: <5%/hour
```

**Solutions:**
```bash
# 1. Check background activity
adb shell dumpsys batterystats | grep -A 10 "com.moltar.brack"

# 2. Monitor wakelocks
adb shell dumpsys power | grep -i wake

# 3. Reduce inference frequency
# Adjust app settings for less frequent model calls

# 4. Enable battery optimization
# Settings > Apps > Brack > Battery > Optimize battery usage (disable)

# 5. Check for thermal issues causing higher power draw
adb shell dumpsys battery | grep -E "(temperature|voltage)"
```

## Build and Development Issues

### Problem: Gradle Build Fails
```
Error: Could not find method compile() for arguments
```

**Solutions:**
```bash
# 1. Check Gradle version compatibility
cd research/brack
./gradlew --version

# 2. Clean and rebuild
./gradlew clean
rm -rf .gradle/ build/

# 3. Update dependencies
./gradlew dependencyUpdates

# 4. Check Android SDK/NDK versions
echo "Android SDK: $ANDROID_SDK_ROOT"
echo "Android NDK: $ANDROID_NDK_ROOT"
ls $ANDROID_SDK_ROOT/platforms/  # Should have android-33+
```

### Problem: ExecuTorch Compilation Fails
```
Error: Undefined symbols for architecture arm64
```

**Solutions:**
```bash
# 1. Verify NDK version compatibility
export ANDROID_NDK_ROOT=/path/to/android-ndk-r25+
echo "NDK Version: $(cat $ANDROID_NDK_ROOT/source.properties | grep Pkg.Revision)"

# 2. Clean ExecuTorch build cache
cd research/spaceghost/executorch
rm -rf build/ cmake-out/

# 3. Rebuild with correct ABI
export ANDROID_ABI=arm64-v8a
python setup.py build_ext --inplace

# 4. Check CMake configuration
cmake --version  # Should be 3.19+
```

### Problem: Python Import Errors
```
ModuleNotFoundError: No module named 'executorch'
```

**Solutions:**
```bash
# 1. Check Python environment
python --version
which python
pip list | grep executorch

# 2. Reinstall dependencies
cd research/brack
pip install -r requirements.txt --force-reinstall

# 3. Check virtual environment
source venv/bin/activate  # or conda activate moltar
python -c "import executorch; print('✅ Working')"

# 4. Update PATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/research/spaceghost"
```

## System-Level Issues

### Problem: Sandbox/Restriction Errors
```
Error: Operation not permitted (sandbox restriction)
```

**Solutions:**
```bash
# 1. Check if running in restricted environment
echo "User: $(whoami)"
echo "Groups: $(groups)"

# 2. Request elevated permissions if needed
sudo -l  # Check sudo permissions

# 3. Use alternative deployment method
# Deploy via local device connection instead of sandboxed environment

# 4. Check file system permissions
ls -la /dev/ | grep usb
```

### Problem: Network/Connectivity Issues
```
Error: Failed to download model (network timeout)
```

**Solutions:**
```bash
# 1. Check network connectivity
ping -c 3 huggingface.co

# 2. Try with different network
# Switch WiFi networks or use mobile data

# 3. Use proxy if required
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# 4. Download manually and copy
# Download model on different machine, transfer via USB
```

## Advanced Debugging

### Collect Comprehensive Logs
```bash
# 1. Device logs
adb logcat -d > device_logs_$(date +%Y%m%d_%H%M%S).txt

# 2. System information
adb shell getprop > device_properties.txt
adb shell dumpsys battery > battery_info.txt

# 3. App-specific logs
adb logcat | grep -i brack > app_logs.txt

# 4. Performance metrics
./scripts/falsify_performance_claims.sh > performance_report.txt
```

### Run Extended Diagnostics
```bash
# Complete system diagnostic
cd research/brack
./scripts/diagnose_deployment.sh --full

# SpaceGhost optimization check
cd ../spaceghost
python falsification_req_xnn_002.py

# Model integrity check
python -c "
import torch
model_path = 'models/LFM2-350M/model.pte'
try:
    # Load and validate model
    with open(model_path, 'rb') as f:
        data = f.read()
    print(f'✅ Model file valid: {len(data)} bytes')
except Exception as e:
    print(f'❌ Model file corrupted: {e}')
"
```

### Performance Profiling
```bash
# CPU profiling
adb shell dumpsys cpuinfo | grep -A 20 "com.moltar.brack"

# Memory profiling
adb shell dumpsys meminfo com.moltar.brack

# Network profiling (if applicable)
adb shell dumpsys netstats | grep -A 10 "com.moltar.brack"

# Battery impact
adb shell dumpsys batterystats | grep -A 5 "com.moltar.brack"
```

## Getting Help

### Quick Reference
```bash
# Most common fixes
./moltar_setup.sh          # One-click environment reset
adb kill-server && adb start-server  # ADB reset
./gradlew clean           # Clean build
rm -rf models/ && ./scripts/download_lfm_model.sh LiquidAI/LFM2-350M  # Model reset
```

### Issue Reporting Template
When filing issues, include:
1. **Full error message and traceback**
2. **Device model and Android version**
3. **Complete command sequence that failed**
4. **Output of diagnostic script**: `./scripts/diagnose_deployment.sh`
5. **Relevant log files**: `adb logcat -d | grep -i error`

### Prevention Best Practices
- Always run diagnostic script before troubleshooting
- Keep device battery >20% during deployment
- Use original USB cables and ports
- Update Android SDK/NDK regularly
- Test on clean device state (clear app data)

---

## Summary of Most Common Issues

| Issue | Frequency | Quick Fix |
|-------|-----------|-----------|
| Device unauthorized | Very Common | Accept USB debugging prompt |
| SpaceGhost not applied | Common | Check optimization pipeline |
| Model download fails | Common | Login to HuggingFace, check network |
| Memory issues | Common | Use smaller model, clear cache |
| ADB connection drops | Common | Restart ADB, check USB |
| Build failures | Moderate | Clean build, update dependencies |
| Performance issues | Moderate | Verify optimizations, check thermal state |

**Remember**: 90% of deployment issues are resolved by running the diagnostic script and following the device connection troubleshooting steps.