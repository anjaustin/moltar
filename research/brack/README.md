# Brack: Liquid Foundation Model Chat Deployment

Research project for deploying Liquid.ai's LFN (Liquid Foundation Model) for chat applications on Motorola devices using ExecuTorch.

## Overview

This project implements end-to-end deployment of Liquid.ai's Liquid Foundation Models optimized for conversational AI, specifically targeting the Motorola 5G Play (Snapdragon 480) for on-device inference.

## Architecture

```
brack/
├── src/                    # Android application source
│   ├── main/
│   │   ├── java/          # Java/Kotlin source
│   │   ├── res/           # Android resources
│   │   └── cpp/           # Native C++ code
│   └── androidTest/       # Android tests
├── models/                # LFN model files and configs
├── config/                # Build and deployment configs
├── scripts/               # Build and deployment scripts
├── deployment/            # Deployment artifacts and bundles
├── test/                  # Testing infrastructure
└── docs/                  # Project documentation
```

## Prerequisites

### System Requirements
- **Android Studio**: Arctic Fox or later (2021.3.1+)
- **Android SDK**: API 33+ (Android 13)
- **Android NDK**: r25+ for native compilation
- **Python**: 3.8+ for build scripts

### Device Requirements
- **Motorola 5G Play (2023)** with Snapdragon 480
- **Android 12+** with API level 31+
- **USB Debugging** enabled
- **Root access** (optional, recommended for full access)

### Dependencies
- **ExecuTorch**: 0.4.0+ (Android AAR)
- **Liquid.ai LFM SDK**: Latest release
- **Android Gradle Plugin**: 8.0+

## Quick Start

### 1. Environment Setup
```bash
# Clone and setup moltar environment
cd /path/to/moltar
./moltar_setup.sh

# Navigate to brack project
cd research/brack

# Run environment setup
./scripts/setup_environment.sh
```

### 2. Download LFN Model
```bash
# Download LFN model from HuggingFace
./scripts/download_lfm_model.sh liquid-ai/LFM-2B-Chat

# Available models:
# liquid-ai/LFM-2B-Chat   (recommended for mobile)
# liquid-ai/LFM-7B-Chat   (advanced conversational)
# liquid-ai/LFM-40B-Chat  (high-capability model)
```

### 3. Build Android App
```bash
# Build debug APK
./scripts/build_debug.sh

# Or use Android Studio
# Open src/ directory as Android project
```

### 4. Deploy to Device
```bash
# Deploy to connected Motorola device
./scripts/deploy_device.sh
```

## Model Configuration

### Supported LFN Models
- **LFM-2B-Chat**: 2B parameter conversational model
- **LFM-7B-Chat**: 7B parameter advanced conversational model
- **LFM-40B-Chat**: 40B parameter high-capability model

### Performance Targets (Snapdragon 480)
- **Latency**: <500ms response time
- **Memory**: <512MB RAM usage
- **Battery**: <10% additional drain
- **Storage**: <2GB model size

## Development Workflow

### Local Development
```bash
# Start development server
./scripts/dev_server.sh

# Run unit tests
./scripts/run_tests.sh

# Build and deploy
./scripts/build_deploy.sh
```

### Device Testing
```bash
# Connect device
../../scripts/device/connect_device.sh

# Deploy test build
./scripts/deploy_test.sh

# Monitor performance
./scripts/monitor_performance.sh
```

## Configuration Files

### Build Configuration (`config/build.gradle.kts`)
```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    compileSdk = 34
    defaultConfig {
        applicationId = "ai.moltar.brack"
        minSdk = 31
        targetSdk = 34
    }
}

dependencies {
    implementation("org.pytorch:executorch-android:0.4.0")
    implementation("ai.liquid:lfm-android:latest")
}
```

### Model Configuration (`config/lfm_config.json`)
```json
{
  "model": "lfm-2b-chat",
  "backend": "executorch",
  "quantization": "4bit",
  "max_tokens": 2048,
  "temperature": 0.7,
  "device": "cpu"
}
```

## API Usage

### Basic Chat Interface
```kotlin
class BrackChatActivity : AppCompatActivity() {
    private lateinit var lfmModel: LFMModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize LFM model
        lfmModel = LFMModel.loadFromAsset(this, "lfm-2b-chat.pte")

        // Setup chat interface
        setupChatInterface()
    }

    private fun sendMessage(message: String) {
        lifecycleScope.launch {
            val response = lfmModel.generate(message)
            displayResponse(response)
        }
    }
}
```

### Native Integration (C++)
```cpp
#include <executorch/extension/llm/model.h>
#include <ai/liquid/lfm/model.h>

// Load LFN model
auto model = liquid::lfm::Model::load_from_file("lfm-2b-chat.pte");

// Generate response
std::string response = model->generate("Hello, how are you?", {
    .max_tokens = 100,
    .temperature = 0.7f
});
```

## Performance Optimization

### Snapdragon 480 Specific
- **DSP Acceleration**: Leverage Hexagon DSP for inference
- **GPU Offloading**: Use Adreno GPU for matrix operations
- **Memory Optimization**: 4-bit quantization for mobile constraints
- **Caching Strategy**: Hybrid cache for Liquid architecture

### Monitoring
```bash
# Performance monitoring
./scripts/monitor_performance.sh

# Memory profiling
./scripts/profile_memory.sh

# Battery impact analysis
./scripts/analyze_battery.sh
```

## Testing

### Unit Tests
```bash
./scripts/run_unit_tests.sh
```

### Integration Tests
```bash
./scripts/run_integration_tests.sh
```

### Device Tests
```bash
./scripts/run_device_tests.sh
```

## Deployment

### Debug Build
```bash
./scripts/build_debug.sh
./scripts/deploy_debug.sh
```

### Release Build
```bash
./scripts/build_release.sh
./scripts/deploy_release.sh
```

### CI/CD Integration
```bash
./scripts/ci_build.sh
./scripts/ci_deploy.sh
```

## Troubleshooting

### Common Issues

#### Model Loading Failures
```
Error: Failed to load LFM model
```
**Solutions:**
- Verify model file exists in `models/` directory
- Check file permissions
- Ensure sufficient storage space (>2GB free)

#### Performance Issues
```
Warning: Inference latency >500ms
```
**Solutions:**
- Enable DSP acceleration in config
- Reduce model size with quantization
- Optimize prompt length

#### Device Compatibility
```
Error: Unsupported Android version
```
**Solutions:**
- Update device to Android 12+ (API 31+)
- Check Snapdragon 480 compatibility
- Verify USB debugging is enabled

## Contributing

### Code Style
- **Kotlin**: Follow Android Kotlin style guide
- **C++**: Follow Google C++ style guide
- **Documentation**: Clear, concise, and complete

### Commit Messages
```
feat: add chat message persistence
fix: resolve memory leak in LFM inference
docs: update deployment guide for Snapdragon 480
```

### Pull Requests
- **Title**: Clear, descriptive summary
- **Description**: Detailed explanation of changes
- **Testing**: Include test results and device testing
- **Documentation**: Update relevant docs

## License

This research project is part of the moltar repository and follows the same licensing terms.

## Contact

For questions about the brack project:
- **Technical Issues**: Check `docs/troubleshooting.md`
- **Research Discussion**: See moltar repository issues
- **Performance Data**: Review `deployment/benchmark_results.json`

---

*Brack: Bringing Liquid AI to Motorola devices through rigorous research and optimization.*