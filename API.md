# API Reference

Complete API reference for moltar research tools and interfaces.

## Command Line Interface

### moltar Command

The main command-line interface for moltar operations.

```bash
moltar [command] [options]
```

#### Commands

##### `moltar setup`
Interactive device setup wizard.

```bash
moltar setup                    # Full interactive setup
moltar setup --quick           # Quick connect (existing setup)
moltar setup:non-interactive   # Automated setup (no prompts)
```

##### `moltar connect`
Device connection management.

```bash
moltar connect                  # Establish device connection
```

##### `moltar research`
Research environment setup.

```bash
moltar research                 # Configure research environment
```

##### `moltar device`
Device-specific operations.

```bash
moltar device info             # Display device information
moltar device test             # Test device connectivity
```

##### `moltar help`
Display help information.

```bash
moltar help                    # Show command reference
```

## Device Connection API

### connect_device.sh

Primary device connection script.

#### Functions

##### `check_prerequisites()`
Verify system requirements for device connection.

**Returns**: 0 on success, 1 on failure

**Checks**:
- Android platform tools availability
- Device connectivity
- Root access status

##### `check_device_connection()`
Verify device connection and compatibility.

**Returns**: Device information object

**Validates**:
- ADB device detection
- Motorola device identification
- Android API level compatibility

##### `setup_device()`
Prepare device for research operations.

**Creates**:
- `/data/local/tmp/moltar-research/` directory
- Basic file permissions
- Research environment structure

#### Usage Examples

```bash
# Basic connection check
./scripts/device/connect_device.sh check

# Full connection with setup
./scripts/device/connect_device.sh

# Device information only
./scripts/device/connect_device.sh info
```

## Research Environment API

### setup_research_device.sh

Device research environment configuration.

#### Functions

##### `install_research_tools()`
Install essential research utilities on device.

**Installs**:
- BusyBox (if available)
- Basic shell utilities
- Research directory structure

##### `configure_research_environment()`
Set up research environment variables and scripts.

**Configures**:
- Environment variables
- Shell aliases
- Log rotation scripts
- Performance monitoring

##### `setup_logging()`
Configure automated logging infrastructure.

**Creates**:
- Log directories
- Rotation policies
- Compression settings

##### `setup_performance_monitoring()`
Enable performance data collection.

**Monitors**:
- CPU usage
- Memory consumption
- Battery drain
- Network activity

#### Configuration Files

##### `research_env.sh`
Device-side environment configuration.

```bash
# Environment variables
export BRACK_HOME="/data/local/tmp/brack"
export PATH="$BRACK_HOME/bin:$PATH"

# Aliases
alias research-logs="cd $BRACK_HOME/logs"
alias research-data="cd $BRACK_HOME/data"
```

## LFN Model API

### download_lfm_model.sh

Liquid Foundation Model download and management.

#### Parameters

- `model_name`: HuggingFace model identifier
  - Default: `LiquidAI/LFM2-350M`
  - Examples: `liquid-ai/LFM-2B-Chat`, `liquid-ai/LFM-7B-Chat`

#### Functions

##### `check_prerequisites()`
Verify download requirements.

**Validates**:
- Python 3.8+ availability
- `huggingface_hub` package
- Git LFS installation
- Disk space availability

##### `download_model()`
Download model from HuggingFace.

**Process**:
1. Create model directory
2. Authenticate with HuggingFace
3. Download model files
4. Verify integrity
5. Update configuration

**Outputs**:
- Model files in `models/{model_name}/`
- Updated `config/lfm_config.json`
- Download verification logs

#### Model File Structure

```
models/LFM2-350M/
├── config.json          # Model hyperparameters
├── tokenizer.json       # Tokenizer configuration
├── model.pte           # ExecuTorch model file
├── generation_config.json  # Generation parameters
└── special_tokens_map.json # Special token mappings
```

## Android Application API

### Brack Application Interface

#### MainActivity.kt

Primary chat interface component.

##### Properties

- `lfmModule: LLMModule` - ExecuTorch LFM inference engine
- `chatHistory: StringBuilder` - Conversation history buffer

##### Methods

###### `initializeLFM()`
Initialize LFM model for inference.

```kotlin
private suspend fun initializeLFM() {
    withContext(Dispatchers.IO) {
        // Load model from assets
        val modelFile = assets.open("lfm-2b-chat.pte")
        lfmModule = LLMModule(modelFile)
    }
}
```

###### `sendMessage(message: String)`
Send user message and generate response.

```kotlin
private fun sendMessage(message: String) {
    appendToChat(message, "You")
    lifecycleScope.launch {
        generateResponse(message)
    }
}
```

###### `generateResponse(userMessage: String)`
Generate AI response using LFM model.

```kotlin
private suspend fun generateResponse(userMessage: String) {
    val config = LLMModule.LLMConfig().apply {
        maxSeqLen = 2048
        temperature = 0.7f
        useKVCache = true
    }

    val response = withContext(Dispatchers.IO) {
        lfmModule.generate(userMessage, config)
    }

    appendToChat(response.text, "Assistant")
}
```

#### Configuration

##### `lfm_config.json`
Model and inference configuration.

```json
{
  "model": {
    "name": "lfm2-350m",
    "parameters": 354483968,
    "quantization": "bfloat16"
  },
  "inference": {
    "max_tokens": 2048,
    "temperature": 0.7,
    "device": "cpu"
  },
  "performance": {
    "target_latency_ms": 200,
    "max_memory_mb": 700
  }
}
```

## Testing API

### test_brack_deployment.sh

Comprehensive deployment testing framework.

#### Test Categories

##### Environment Tests
- Python dependencies verification
- Android tools availability
- File system permissions
- Network connectivity

##### Model Tests
- Model file presence and integrity
- Configuration file validation
- Download verification

##### Build Tests
- Gradle build execution
- APK generation
- Android manifest validation

##### Deployment Tests
- Device connectivity verification
- APK installation testing
- Permission validation

#### Test Results

Returns structured test results:
```bash
{
  "tests_run": 15,
  "tests_passed": 12,
  "tests_failed": 3,
  "failures": ["missing_model", "build_failure", "device_unavailable"]
}
```

### falsify_performance_claims.sh

Scientific performance claim falsification.

#### Claim Categories

- **Model Size**: Storage requirements validation
- **Memory Usage**: RAM consumption verification
- **Latency**: Response time measurement
- **Battery Drain**: Power consumption analysis
- **Compatibility**: Platform support validation

#### Falsification Process

1. **Hypothesis Formation**: Define falsifiable performance claims
2. **Evidence Collection**: Gather empirical performance data
3. **Statistical Analysis**: Apply statistical tests to results
4. **Conclusion Formation**: Accept/reject claims based on evidence

#### Output Format

```bash
CLAIM 1: Model Size Claim - SUPPORTED
  Evidence: Model is 450MB (<500MB target)

CLAIM 2: Memory Usage Claim - FALSIFIED
  Counter-evidence: Model parameters alone require 676MB

CLAIM 3: Latency Claim - SUPPORTED
  Evidence: Snapdragon 480 capable of <200ms response
```

## Build System API

### Gradle Build Configuration

#### build.gradle.kts

Android application build specification.

```kotlin
plugins {
    id("com.android.application") version "8.2.0"
    id("org.jetbrains.kotlin.android") version "1.9.10"
}

android {
    compileSdk = 34
    defaultConfig {
        applicationId = "com.moltar.brack"
        minSdk = 31
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }
}

dependencies {
    implementation("org.pytorch:executorch-android:0.4.0")
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
}
```

#### Build Tasks

##### `./gradlew assembleDebug`
Build debug APK for development.

**Outputs**: `app/build/outputs/apk/debug/app-debug.apk`

##### `./gradlew assembleRelease`
Build release APK for production.

**Outputs**: `app/build/outputs/apk/release/app-release.apk`

##### `./gradlew installDebug`
Build and install debug APK on connected device.

**Requires**: Connected Android device with USB debugging enabled.

## Error Handling

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| 0 | Success | No action required |
| 1 | Device not found | Check USB connection |
| 2 | Model download failed | Verify internet connection |
| 3 | Build failure | Check Gradle configuration |
| 4 | Permission denied | Run with appropriate privileges |
| 5 | Insufficient resources | Check disk space/RAM |

### Exception Handling

#### Kotlin/Android Exceptions

```kotlin
try {
    lfmModule.generate(message, config)
} catch (e: IOException) {
    // Model file not found
    showError("Model file missing")
} catch (e: IllegalArgumentException) {
    // Invalid configuration
    showError("Configuration error")
} catch (e: RuntimeException) {
    // Inference failure
    showError("Inference failed")
}
```

#### Shell Script Error Handling

```bash
set -e  # Exit on any error

trap 'echo "Script failed at line $LINENO"' ERR

# Safe command execution
if ! adb devices >/dev/null 2>&1; then
    echo "ADB command failed"
    exit 1
fi
```

## Performance Monitoring API

### Device Performance Metrics

#### CPU Monitoring
```bash
# Get CPU usage
adb shell cat /proc/stat

# Monitor specific process
adb shell top -p $(adb shell pidof com.moltar.brack)
```

#### Memory Monitoring
```bash
# Application memory info
adb shell dumpsys meminfo com.moltar.brack

# System memory info
adb shell cat /proc/meminfo
```

#### Battery Monitoring
```bash
# Battery status
adb shell dumpsys battery

# Power consumption
adb shell dumpsys power
```

## Logging API

### Log Categories

#### Application Logs
- **INFO**: General information and status updates
- **WARNING**: Non-critical issues that don't stop execution
- **ERROR**: Critical errors that may affect functionality
- **DEBUG**: Detailed debugging information

#### Performance Logs
- **LATENCY**: Response time measurements
- **MEMORY**: RAM usage statistics
- **BATTERY**: Power consumption data
- **NETWORK**: Data transfer metrics

### Log Format

```
[timestamp] [level] [component] message
```

**Example**:
```
2024-01-26 14:30:15 INFO MainActivity Model initialized successfully
2024-01-26 14:30:16 DEBUG InferenceEngine Processing input: "Hello"
2024-01-26 14:30:17 INFO PerformanceMonitor Latency: 145ms
```

### Log Rotation

Automatic log rotation prevents disk space issues:

```bash
# Rotate logs when >1MB
# Keep last 10 rotated logs
# Compress old logs
```

## Security API

### Model Security

#### Input Sanitization
```kotlin
fun sanitizeInput(input: String): String {
    return input
        .trim()
        .take(1000) // Limit length
        .replace(Regex("[\\x00-\\x1F\\x7F-\\x9F]"), "") // Remove control chars
}
```

#### Output Filtering
```kotlin
fun filterOutput(output: String): String {
    // Remove potentially harmful content
    return output
        .replace(Regex("password|token|key", RegexOption.IGNORE_CASE), "[REDACTED]")
}
```

### Device Security

#### Permission Management
Required Android permissions:
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

#### Secure Storage
```kotlin
// Use EncryptedSharedPreferences for sensitive data
val masterKey = MasterKey.Builder(context)
    .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
    .build()

val sharedPreferences = EncryptedSharedPreferences.create(
    context,
    "secure_prefs",
    masterKey,
    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
)
```

---

*This API reference provides comprehensive documentation for all moltar interfaces, functions, and configuration options.*