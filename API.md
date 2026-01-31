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

## SpaceGhost Optimization API

### Snapdragon 480 Hardware Detection

#### `snapdragon_480_caps_t` Structure
Hardware capability detection results.

**Members:**
- `bool has_dotprod` - ARMv8.2-A +dotprod support
- `bool has_fp16` - Half-precision floating point support
- `bool has_sve` - Scalable Vector Extension support
- `uint32_t l3_cache_size_kb` - L3 cache size in KB
- `uint32_t big_core_count` - Number of big cores (Cortex-A76)
- `uint32_t little_core_count` - Number of little cores (Cortex-A55)
- `uint32_t max_frequency_mhz` - Maximum CPU frequency

#### `detect_snapdragon_480_capabilities()`
Detect Snapdragon 480 hardware capabilities at runtime.

```c
#include "snapdragon_480_optimization.h"

snapdragon_480_caps_t caps = detect_snapdragon_480_capabilities();
if (caps.has_dotprod) {
    printf("Dot product support detected\n");
}
```

**Returns:** `snapdragon_480_caps_t` structure with detected capabilities

**Note:** Uses `getauxval(AT_HWCAP)` for runtime CPU feature detection

#### `is_snapdragon_480_with_dotprod()`
Check if running on Snapdragon 480 with dot product support.

```c
if (is_snapdragon_480_with_dotprod()) {
    enable_snapdragon_480_optimizations();
}
```

**Returns:** `true` if Snapdragon 480 with dot product support detected

### Thread Pool Optimization

#### `pin_thread_to_big_cores(pthread_t thread, uint32_t thread_index)`
Pin thread to Snapdragon 480 big cores (Cortex-A76).

```c
#include <pthread.h>
#include "snapdragon_480_optimization.h"

pthread_t thread;
pthread_create(&thread, NULL, worker_function, NULL);
pin_thread_to_big_cores(thread, 0); // Pin to core 0
```

**Parameters:**
- `thread`: Thread to pin
- `thread_index`: Index for core assignment (0-1 for Snapdragon 480)

**Returns:** 0 on success, errno on failure

#### `get_optimal_thread_count_snapdragon_480()`
Get optimal thread count for Snapdragon 480 (big cores only).

```c
uint32_t optimal_threads = get_optimal_thread_count_snapdragon_480();
// Returns 2 for Snapdragon 480 (big cores only)
```

**Returns:** Recommended thread count (2 for Snapdragon 480)

### Cache Optimization

#### `prefetch_snapdragon_l3(const void* data, size_t size, int locality)`
Prefetch data into Snapdragon 480 L3 cache.

```c
#include "cache_optimization_snapdragon.h"

// Prefetch model weights for high locality
prefetch_snapdragon_l3(weights, weight_size, PREFETCH_LOCALITY_HIGH);
```

**Parameters:**
- `data`: Pointer to data to prefetch
- `size`: Size of data in bytes
- `locality`: Temporal locality hint (0-2)

#### `optimize_memory_layout_snapdragon(const void* data, size_t size, size_t element_size)`
Optimize memory layout for Snapdragon 480 cache hierarchy.

```c
void* optimized_data = optimize_memory_layout_snapdragon(
    input_data, data_size, sizeof(float)
);
if (optimized_data) {
    // Use optimized_data (cache-aligned)
    free(optimized_data);
}
```

**Returns:** Cache-aligned memory pointer, or original pointer if allocation fails

### LFN XNNPack Cleanup Pass

#### `LFNXNNPackCleanupPass` Class
ExecuTorch ExportPass for cleaning graphs before XNNPack partitioning.

**Implements:** `REQ-XNN-001` (MaxPool2d delegation) and `REQ-XNN-002` (quantization optimization)

```python
from executorch.exir.pass_base import ExportPass
from research.spaceghost.patches.xnnpack.lfn_xnnpack_cleanup_pass import LFNXNNPackCleanupPass

class LFNXNNPackCleanupPass(ExportPass):
    def call(self, graph_module):
        """
        Apply cleanup transformations for XNNPack compatibility.

        Args:
            graph_module: FX GraphModule to transform

        Returns:
            PassResult with transformed graph
        """
        # Fixes REQ-XNN-001: MaxPool2d tuple output
        # Fixes REQ-XNN-002: Redundant Q/DQ chains
        # Prepares for REQ-XNN-003: Snapdragon optimization
        pass
```

**Methods:**
- `call(graph_module)`: Apply cleanup transformations
- `remove_unused_maxpool_indices(graph_module)`: Strip MaxPool2d indices
- `fuse_redundant_quantization(graph_module)`: Remove Q/DQ chains

### Dot Product Kernels

#### GEMM Microkernels
Optimized GEMM kernels using ARM dot product instructions.

```c
// 1x8 microkernel for Snapdragon 480
void xnn_qs8_gemm_minmax_ukernel_1x8__snapdragon480_dotprod_ld128(
    size_t mr, size_t nr, size_t k,
    const int8_t* a, size_t a_stride,
    const int8_t* w, size_t w_stride,
    const float* bias, float* c, size_t c_stride,
    const union xnn_qs8_conv_minmax_params params[restrict static 1]
);
```

**Features:**
- ARMv8.2-A +dotprod instruction utilization
- Cortex-A76 microarchitecture optimization
- 128-bit load optimization
- Quantized matrix multiplication acceleration

### Validation and Testing API

#### Falsification Testing Framework

##### `falsification_req_xnn_001.py`
Test REQ-XNN-001 MaxPool2d delegation implementation.

```bash
python research/spaceghost/falsification_req_xnn_001.py
```

**Tests:**
- MaxPool2d tuple output transformation
- XNNPack partitioner delegation
- Graph structure preservation

##### `falsification_req_xnn_002.py`
Test REQ-XNN-002 quantization chain optimization.

```bash
python research/spaceghost/falsification_req_xnn_002.py
```

**Tests:**
- Q/DQ chain detection and fusion
- Graph simplification validation
- Performance overhead reduction

##### `falsification_req_xnn_003.py`
Test REQ-XNN-003 Snapdragon 480 optimizations.

```bash
python research/spaceghost/falsification_req_xnn_003.py
```

**Tests:**
- Hardware capability detection
- Dot product kernel performance
- Thread optimization effectiveness
- Cache optimization validation

#### Performance Monitoring

##### `snapdragon_480_metrics_t` Structure
Performance metrics for Snapdragon 480 optimizations.

**Members:**
- `uint64_t dotprod_instructions` - Dot product instructions executed
- `uint64_t l3_cache_accesses` - L3 cache access count
- `uint64_t big_core_time_us` - Time spent on big cores (microseconds)
- `double dotprod_utilization` - Percentage of ops using dot product
- `double cache_hit_rate` - L3 cache hit rate
- `double big_core_utilization` - Percentage time on big cores

##### `collect_snapdragon_480_metrics(snapdragon_480_metrics_t* metrics)`
Collect performance metrics for Snapdragon optimizations.

```c
snapdragon_480_metrics_t metrics;
collect_snapdragon_480_metrics(&metrics);
print_snapdragon_480_metrics(&metrics);
```

### Build System Integration

#### CMake Configuration
```cmake
# Enable Snapdragon 480 optimizations
if(ANDROID AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod")
    add_definitions(-DXNN_ARCH_ARM64_DOTPROD=1)
    add_definitions(-DSNAPDRAGON_480_OPTIMIZATIONS=1)
endif()
```

#### Makefile Integration
```makefile
# Snapdragon 480 specific flags
SNAPDRAGON_CFLAGS = -march=armv8.2-a+dotprod -DSNAPDRAGON_480_OPTIMIZATIONS=1
SNAPDRAGON_LDFLAGS = -latomic

# Build optimized libraries
snapdragon: CFLAGS += $(SNAPDRAGON_CFLAGS) -O3 -flto
snapdragon: $(SNAPDRAGON_OBJS)
    $(AR) rcs libxnnpack_snapdragon.a $(SNAPDRAGON_OBJS)
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

*This API reference provides comprehensive documentation for all moltar interfaces, functions, and configuration options, including the new SpaceGhost optimization APIs.*