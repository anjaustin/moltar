# Brack LFN Deployment Guide

Complete guide for deploying Liquid.ai's LFN chat models on Motorola devices.

## Overview

Brack enables research and deployment of Liquid Foundation Models for conversational AI on mobile devices. This guide covers the complete setup process from development environment to device deployment.

## Prerequisites

### Hardware
- **Motorola 5G Play (2023)** with Snapdragon 480
- **USB-C cable** for device connection
- **macOS development machine** (11.0+ recommended)

### Software
- **Android Studio**: Arctic Fox (2020.3.1) or later
- **Android SDK**: API 33+ with Android 13 support
- **Android NDK**: r25+ for native compilation
- **Python 3.8+** with virtual environment support

### Access Requirements
- **Liquid.ai Developer Account**: For LFN model access
- **USB Debugging**: Enabled on Motorola device
- **ADB Authorization**: Device recognizes development machine

## Quick Start

### 1. Environment Setup
```bash
# Navigate to brack project
cd research/brack

# Run complete environment setup
./scripts/setup_environment.sh
```

This script will:
- ✅ Install system dependencies (Homebrew, Python, Android tools)
- ✅ Create Python virtual environment with required packages
- ✅ Set up Android development environment
- ✅ Create Android project structure
- ✅ Generate build and deployment scripts

### 2. Model Acquisition
```bash
# Download LFN model from HuggingFace
./scripts/download_lfm_model.sh LiquidAI/LFM2-350M

# Available models on HuggingFace:
# LiquidAI/LFM2-350M      ⭐ RECOMMENDED for mobile/Snapdragon 480
# liquid-ai/LFM-2B-Chat   (conversational model)
# liquid-ai/LFM-7B-Chat   (advanced conversational model)
# liquid-ai/LFM-40B-Chat  (high-capability model)

# Model files will be automatically downloaded to models/ directory:
# - model files (ExecuTorch .pte or other formats)
# - tokenizer.json (tokenizer configuration)
# - config.json (model hyperparameters)

## LFM2-350M: Mobile-Optimized Model

**Why LFM2-350M for Motorola/Snapdragon 480:**

| Specification | LFM2-350M | vs LFM-2B |
|---------------|-----------|-----------|
| **Parameters** | 350M | 2B (6x larger) |
| **Storage** | ~500MB | ~2GB (4x smaller) |
| **Memory** | <256MB | <512MB (2x efficient) |
| **Latency** | <200ms | <500ms (2.5x faster) |
| **Battery** | <5% drain | <10% drain (2x efficient) |

**Key Advantages:**
- **Ultra-compact**: Specifically designed for edge/mobile deployment
- **Multi-language**: English, Arabic, Chinese, French, German, Japanese, Korean, Spanish
- **Fast inference**: 2x faster than competitors on CPU
- **Hardware optimized**: Runs efficiently on CPU, GPU, and NPU
- **Agentic tasks**: Perfect for conversational AI and RAG applications
```

### 3. Application Build
```bash
# Build debug APK
./scripts/build_debug.sh

# Verify APK creation
ls -la src/app/build/outputs/apk/debug/
```

### 4. Device Deployment
```bash
# Deploy to connected Motorola device
./scripts/deploy_device.sh
```

## Detailed Setup Process

### Environment Configuration

#### Android Studio Setup
1. Download and install Android Studio
2. Complete initial setup wizard
3. Install required SDK components:
   - Android SDK API 33+
   - Android NDK r25+
   - CMake 3.22+
   - LLDB (for debugging)

#### SDK Environment Variables
```bash
# Add to ~/.zshrc or ~/.bashrc
export ANDROID_HOME="$HOME/Library/Android/sdk"
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/$(ls $ANDROID_HOME/ndk | head -1)"
export PATH="$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools"
```

#### Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Model Integration

#### LFN Model Files Structure
```
models/
├── lfm-2b-chat.pte      # ExecuTorch model file
├── tokenizer.json       # Tokenizer configuration
├── config.json          # Model hyperparameters
└── metadata.json        # Model metadata
```

#### Configuration Setup
Edit `config/lfm_config.json`:
```json
{
  "model": {
    "name": "lfm-2b-chat",
    "backend": "executorch",
    "quantization": "4bit"
  },
  "inference": {
    "max_tokens": 2048,
    "temperature": 0.7,
    "device": "cpu"
  }
}
```

### Android Application Development

#### Project Structure
```
src/
├── main/
│   ├── AndroidManifest.xml
│   ├── java/com/moltar/brack/
│   │   ├── MainActivity.kt     # Main chat interface
│   │   └── LFMModule.kt        # LFM integration wrapper
│   ├── res/
│   │   ├── layout/
│   │   │   └── activity_main.xml
│   │   └── values/
│   │       └── strings.xml
│   └── cpp/                    # Native code (optional)
├── androidTest/                # Instrumentation tests
└── test/                       # Unit tests
```

#### Key Components

**MainActivity.kt**: Chat interface with LFM integration
```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var lfmModule: LLMModule

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initializeLFM()
    }

    private fun sendMessage(message: String) {
        lifecycleScope.launch {
            val response = lfmModule.generate(message, config)
            displayResponse(response)
        }
    }
}
```

**LFM Integration**: Uses ExecuTorch for on-device inference
```kotlin
val config = LLMModule.LLMConfig().apply {
    maxSeqLen = 2048
    temperature = 0.7f
    useKVCache = true
}

val result: LLMResult = lfmModule.generate(prompt, config)
```

### Build Configuration

#### Gradle Setup
**settings.gradle.kts**:
```kotlin
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        // Liquid.ai repository when available
    }
}
```

**build.gradle.kts**:
```kotlin
dependencies {
    // ExecuTorch for LFN inference
    implementation("org.pytorch:executorch-android:0.4.0")

    // Android UI components
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")

    // Coroutines for async operations
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

### Device-Specific Optimization

#### Snapdragon 480 Configuration
```json
{
  "performance": {
    "target_latency_ms": 500,
    "use_dsp": true,
    "use_gpu": false,
    "quantization": "4bit"
  }
}
```

#### Memory Management
- **4-bit quantization** for reduced memory footprint
- **KV cache optimization** for efficient inference
- **Batch size 1** for mobile constraints
- **Memory monitoring** built into application

## Testing and Validation

### Unit Tests
```bash
./gradlew test
```

### Instrumentation Tests
```bash
./gradlew connectedAndroidTest
```

### Performance Testing
```bash
# Run performance benchmarks
./scripts/run_performance_tests.sh

# Monitor device resources
./scripts/monitor_device.sh
```

### LFN Model Validation
```bash
# Test model loading
./scripts/test_model_loading.sh

# Validate inference accuracy
./scripts/validate_inference.sh
```

## Deployment Pipeline

### Development Deployment
```bash
# Quick development build and deploy
./scripts/build_deploy.sh
```

### Production Deployment
```bash
# Release build
./scripts/build_release.sh

# Deploy to test devices
./scripts/deploy_release.sh

# Generate deployment report
./scripts/generate_report.sh
```

## Troubleshooting

### Common Issues

#### Build Failures
```
ERROR: NDK not found
```
**Solution**: Verify ANDROID_NDK_HOME environment variable

#### Model Loading Errors
```
ERROR: Failed to load LFM model
```
**Solutions**:
- Verify model files are in `models/` directory
- Check file permissions
- Ensure sufficient device storage (>2GB free)

#### Performance Issues
```
WARNING: Inference latency >500ms
```
**Solutions**:
- Enable DSP acceleration
- Reduce max_tokens in configuration
- Use model quantization

#### Device Connection Problems
```
ERROR: device unauthorized
```
**Solutions**:
- Accept USB debugging authorization dialog
- Revoke and re-grant USB debugging permissions
- Try different USB cable/port

### Debug Tools

#### Android Debug Bridge
```bash
# Monitor device logs
adb logcat | grep brack

# Check device storage
adb shell df /data

# Monitor app performance
adb shell dumpsys meminfo com.moltar.brack
```

#### Application Logging
```bash
# View app-specific logs
adb logcat -s BrackApp

# Monitor LFM inference
adb logcat | grep -i lfm
```

## Performance Optimization

### Mobile-Specific Optimizations
- **Model quantization** (4-bit recommended)
- **DSP offloading** for Snapdragon devices
- **Memory-efficient caching**
- **Background processing** for heavy computations

### Benchmarking
```bash
# Run comprehensive benchmarks
./scripts/benchmark_inference.sh

# Compare different configurations
./scripts/compare_configs.sh
```

### Profiling
```bash
# Memory profiling
./scripts/profile_memory.sh

# CPU usage analysis
./scripts/profile_cpu.sh
```

## Security Considerations

### Model Security
- **Model encryption** for sensitive deployments
- **Secure key management** for model access
- **Input sanitization** to prevent prompt injection
- **Output filtering** for safe responses

### Device Security
- **Permission management** (minimal required permissions)
- **Data isolation** (no sensitive data storage)
- **Network security** (encrypted communications)
- **Update mechanisms** (secure model updates)

## Research Applications

### Use Cases
- **Conversational AI research** on mobile devices
- **Edge computing** performance analysis
- **Liquid Neural Networks** behavior studies
- **Human-AI interaction** experiments

### Data Collection
- **Performance metrics** (latency, memory, battery)
- **User interaction logs** (with privacy controls)
- **Model behavior analysis**
- **Device capability assessments**

## Future Enhancements

### Planned Features
- **Multi-modal inputs** (text + images)
- **Streaming responses** for real-time chat
- **Model switching** between different LFN variants
- **Offline mode** for disconnected operation
- **Custom model fine-tuning** capabilities

### Research Directions
- **Energy efficiency** optimization
- **Privacy-preserving** inference techniques
- **Federated learning** integration
- **Cross-device** model synchronization

---

*This deployment guide provides comprehensive instructions for setting up and deploying Liquid.ai's LFN models on Motorola devices for research and development purposes.*