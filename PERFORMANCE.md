# Performance Guide

Performance characteristics, benchmarks, and optimization guidelines for the moltar research platform.

## Overview

Performance is critical for on-device AI research. This guide covers performance targets, measurement methodologies, and optimization strategies.

## Performance Targets

### LFM2-350M on Snapdragon 480 (Primary Target)

| Metric | Target | Status | Notes |
|--------|--------|--------|-------|
| **Latency** | <200ms | ✅ Validated | Chat response time |
| **Memory** | <256MB | ⚠️ Under review | Runtime memory usage |
| **Storage** | ~500MB | ✅ Validated | Model + app size |
| **Battery** | <5%/hour | ✅ Validated | Additional drain |
| **CPU** | <20% | ✅ Validated | Background usage |

### Comparative Performance

| Model | Device | Latency | Memory | Battery | Status |
|-------|--------|---------|--------|---------|--------|
| LFM2-350M | Snapdragon 480 | <200ms | <256MB | <5% | ✅ Production |
| LFM-2B | Snapdragon 480 | <500ms | <512MB | <10% | ✅ Compatible |
| Baseline | Snapdragon 480 | <50ms | <128MB | <2% | ✅ Reference |

## Measurement Methodology

### Latency Measurement

```python
import time

def measure_latency(model, input_text):
    start_time = time.perf_counter()

    # Model inference
    response = model.generate(input_text)

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    return latency_ms, response
```

**Factors affecting latency:**
- Model size and complexity
- Input length and processing requirements
- Hardware acceleration utilization
- Memory pressure and caching
- Background system load

### Memory Measurement

```bash
# Android memory profiling
adb shell dumpsys meminfo com.moltar.brack

# Process-specific memory
adb shell ps -p $(adb shell pidof com.moltar.brack) -o rss,vsz
```

**Memory components:**
- Model weights and parameters
- KV cache for attention mechanisms
- Input/output processing buffers
- Runtime overhead and allocations

### Battery Measurement

```bash
# Battery drain monitoring
adb shell dumpsys battery

# Power consumption tracking
# Note: Requires additional battery monitoring tools
```

**Battery impact factors:**
- CPU utilization during inference
- DSP/GPU power consumption
- Memory access patterns
- Screen and system state

### CPU Utilization

```bash
# CPU usage monitoring
adb shell top -p $(adb shell pidof com.moltar.brack) -o %CPU

# System-wide CPU stats
adb shell cat /proc/stat
```

## Benchmarking Results

### LFM2-350M Chat Performance

#### Latency Distribution (100 inferences)
```
Min:  85ms
Max: 245ms
Mean: 156ms
P95: 198ms
P99: 235ms
```

#### Memory Usage Over Time
```
Initial load: 180MB
Steady state: 210MB
Peak usage: 240MB
Post-GC: 195MB
```

#### Battery Impact (1-hour continuous chat)
```
Base drain: 2.1%/hour
With LFN: 4.8%/hour
Additional: 2.7%/hour (acceptable)
```

## Optimization Strategies

### Model Optimization

#### Quantization
```python
# 4-bit quantization for mobile
model = quantize_model(model, bits=4)
# Reduces size by ~75%, minimal accuracy loss
```

#### Pruning
```python
# Remove redundant parameters
pruned_model = prune_model(model, sparsity=0.3)
# Reduces size by ~30%, slight accuracy trade-off
```

### Hardware Acceleration

#### DSP Utilization
```cpp
// Enable Hexagon DSP acceleration
runtime_config.enable_dsp = true;
runtime_config.dsp_threads = 4;
```

#### GPU Offloading
```cpp
// Use Adreno GPU for matrix operations
runtime_config.use_gpu = true;
runtime_config.gpu_precision = FP16;
```

### Memory Optimization

#### Efficient Caching
```cpp
// Optimize KV cache management
cache_config.max_sequence_length = 2048;
cache_config.attention_optimization = true;
cache_config.memory_efficient_attention = true;
```

#### Memory Pool Management
```cpp
// Pre-allocate memory pools
memory_config.use_memory_pool = true;
memory_config.pool_size_mb = 128;
memory_config.enable_garbage_collection = true;
```

### Power Optimization

#### Adaptive Frequency
```cpp
// Reduce frequency during idle periods
power_config.adaptive_frequency = true;
power_config.idle_timeout_ms = 5000;
power_config.low_power_mode = true;
```

#### Batch Processing
```cpp
// Process multiple requests efficiently
inference_config.batch_size = 4;
inference_config.dynamic_batching = true;
```

## Performance Monitoring

### Built-in Monitoring

```kotlin
// Enable performance tracking
LFMConfig.enablePerformanceMonitoring = true
LFMConfig.performanceLogLevel = LogLevel.DEBUG

// Monitor specific metrics
val metrics = lfmModel.getPerformanceMetrics()
println("Latency: ${metrics.averageLatency}ms")
println("Memory: ${metrics.peakMemoryUsage}MB")
println("CPU: ${metrics.cpuUtilization}%")
```

### External Monitoring

```bash
# Continuous performance logging
./scripts/monitor_performance.sh com.moltar.brack

# Generate performance reports
./scripts/generate_performance_report.sh
```

## Troubleshooting Performance Issues

### High Latency

**Symptoms:**
- Response times >500ms
- UI freezing during inference

**Solutions:**
1. Check hardware acceleration is enabled
2. Reduce input length
3. Use smaller model variant
4. Close background applications
5. Ensure device is cool (thermal throttling)

### High Memory Usage

**Symptoms:**
- App crashes with OOM errors
- System becomes unresponsive

**Solutions:**
1. Enable memory optimization flags
2. Reduce model size with quantization
3. Clear app cache regularly
4. Monitor memory usage patterns
5. Consider model offloading strategies

### Battery Drain

**Symptoms:**
- Rapid battery depletion
- Device heating during use

**Solutions:**
1. Enable power optimization modes
2. Reduce inference frequency
3. Use lower-precision models
4. Implement idle timeouts
5. Monitor background processing

### CPU Overload

**Symptoms:**
- Device becomes slow/unresponsive
- High CPU temperatures

**Solutions:**
1. Enable DSP acceleration
2. Reduce concurrent operations
3. Optimize inference parameters
4. Use CPU affinity settings
5. Implement request throttling

## Comparative Analysis

### Performance vs Model Size

| Model Size | Latency | Memory | Battery | Quality |
|------------|---------|--------|---------|---------|
| 350M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 2B | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 7B | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| 40B | ⭐ | ⭐ | ❌ | ⭐⭐⭐⭐⭐ |

### Device Compatibility

| Device | Latency | Memory | Battery | Recommendation |
|--------|---------|--------|---------|----------------|
| Snapdragon 480+ | ✅ Excellent | ✅ Excellent | ✅ Good | ⭐ Primary target |
| Snapdragon 6/7/8 Gen 1 | ✅ Good | ✅ Good | ⚠️ Fair | ⭐ Compatible |
| Snapdragon 8 Gen 2+ | ✅ Excellent | ✅ Excellent | ✅ Excellent | ⭐ Optimal |
| A-series/M-series | ⚠️ Variable | ✅ Good | ✅ Good | ⚠️ Test required |
| Older devices | ❌ Poor | ❌ Limited | ❌ High drain | ❌ Not recommended |

## Future Performance Improvements

### Short-term (3-6 months)
- **Model optimization**: Improved quantization techniques
- **Runtime optimization**: Better memory management
- **Hardware utilization**: Enhanced DSP/GPU usage
- **Caching improvements**: More efficient KV cache

### Medium-term (6-12 months)
- **Custom kernels**: Device-specific optimized operations
- **Dynamic adaptation**: Runtime performance adjustment
- **Multi-threading**: Parallel inference processing
- **Edge TPU integration**: Additional accelerator support

### Long-term (1+ years)
- **Neuromorphic computing**: Brain-inspired processing
- **Quantum acceleration**: Quantum-enhanced inference
- **Federated learning**: Distributed model optimization
- **Self-optimizing systems**: Automatic performance tuning

## Contributing Performance Improvements

### Performance Testing
```bash
# Run performance test suite
./scripts/run_performance_tests.sh

# Submit benchmark results
./scripts/submit_benchmark_results.sh
```

### Optimization Proposals
1. **Document the optimization** with before/after metrics
2. **Provide reproducible benchmarks**
3. **Consider backward compatibility**
4. **Test on multiple device types**
5. **Include performance regression tests**

### Reporting Issues
When reporting performance issues:
- Include device specifications
- Provide benchmark results
- Describe expected vs actual performance
- Include system logs and metrics
- Specify test conditions and environment

---

*Performance optimization is an ongoing process. This guide is updated as new optimizations and benchmarks become available.*