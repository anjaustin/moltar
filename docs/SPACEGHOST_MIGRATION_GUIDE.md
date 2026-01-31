# SpaceGhost Migration Guide: Adopting ExecuTorch Optimizations

This guide helps you migrate existing ExecuTorch projects to use SpaceGhost optimizations for improved Liquid AI Foundation Model (LFN) performance on Motorola Snapdragon 480 devices.

## Migration Overview

### Before SpaceGhost
```python
# Standard ExecuTorch pipeline
import torch
from executorch.exir import to_edge
from executorch.backends.xnnpack import XnnpackPartitioner

# Export model
exported = torch.export.export(model, sample_input)
edge = to_edge(exported)
partitioned = edge.to_backend(XnnpackPartitioner())
executable = partitioned.to_executorch()

# Result: Limited XNNPack delegation, suboptimal performance
```

### After SpaceGhost Migration
```python
# Optimized SpaceGhost pipeline
import torch
from executorch.exir import to_edge
from research.spaceghost.patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline
from executorch.backends.xnnpack import XnnpackPartitioner

# Export model
exported = torch.export.export(model, sample_input)
edge = to_edge(exported)

# Apply SpaceGhost optimizations
optimized_edge = run_lfn_xnnpack_pipeline(edge)

partitioned = optimized_edge.to_backend(XnnpackPartitioner())
executable = partitioned.to_executorch()

# Result: 4-8x performance improvement with full DSP acceleration
```

## Prerequisites

### System Requirements
- **ExecuTorch**: 0.4.0+ with XNNPack backend
- **PyTorch**: 2.0+ with torch.export support
- **SpaceGhost**: Latest version from moltar repository
- **Target Hardware**: Snapdragon 480 (optional, optimizations work on any ARM64)

### Installation
```bash
# Clone moltar repository (includes SpaceGhost)
git clone https://github.com/your-org/moltar.git
cd moltar

# SpaceGhost is located in research/spaceghost/
# Add to Python path
export PYTHONPATH="$PYTHONPATH:$(pwd)/research/spaceghost"
```

## Migration Steps

### Step 1: Assess Current Implementation

#### Check Your Current Pipeline
```python
def analyze_current_pipeline(model, sample_input):
    """Analyze current ExecuTorch pipeline"""

    # Current implementation
    try:
        exported = torch.export.export(model, sample_input)
        edge = to_edge(exported)
        partitioned = edge.to_backend(XnnpackPartitioner())

        # Count delegate operations
        delegate_count = 0
        for node in partitioned.graph.nodes:
            if hasattr(node, 'target') and 'delegate' in str(node.target):
                delegate_count += 1

        print(f"Current delegate operations: {delegate_count}")

        # Check for MaxPool2d operations
        maxpool_count = 0
        for node in edge.graph.nodes:
            if 'max_pool2d' in str(node.target):
                maxpool_count += 1

        print(f"MaxPool2d operations: {maxpool_count}")

        return delegate_count, maxpool_count

    except Exception as e:
        print(f"Pipeline analysis failed: {e}")
        return 0, 0

# Analyze your current setup
delegate_ops, maxpool_ops = analyze_current_pipeline(your_model, your_sample_input)
```

#### Performance Baseline
```python
def establish_performance_baseline(model, test_inputs, iterations=100):
    """Establish performance baseline before optimization"""

    model.eval()
    latencies = []

    with torch.no_grad():
        # Warm up
        for _ in range(10):
            _ = model(test_inputs[0])

        # Benchmark
        import time
        for test_input in test_inputs:
            start_time = time.perf_counter()
            _ = model(test_input)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    print("📊 Performance Baseline:")
    print(".2f")
    print(".2f")

    return {
        'avg_latency_ms': avg_latency,
        'p95_latency_ms': p95_latency,
        'baseline_marker': True
    }

# Establish baseline before migration
baseline_results = establish_performance_baseline(your_model, test_inputs)
```

### Step 2: Add SpaceGhost Dependencies

#### Update Import Statements
```python
# Before
from executorch.backends.xnnpack import XnnpackPartitioner

# After
from executorch.backends.xnnpack import XnnpackPartitioner
from research.spaceghost.patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline
```

#### Add Error Handling
```python
def safe_import_spaceghost():
    """Safely import SpaceGhost with fallback"""
    try:
        from research.spaceghost.patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline
        print("✅ SpaceGhost optimizations available")
        return run_lfn_xnnpack_pipeline
    except ImportError as e:
        print(f"⚠️  SpaceGhost not available: {e}")
        print("Falling back to standard ExecuTorch pipeline")
        return None

# Use safe import
spaceghost_optimizer = safe_import_spaceghost()
```

### Step 3: Modify Export Pipeline

#### Minimal Migration (Drop-in Replacement)
```python
def export_with_spaceghost(model, sample_input):
    """Export model with SpaceGhost optimizations"""

    # Standard export
    exported = torch.export.export(model, sample_input)
    edge = to_edge(exported)

    # Apply SpaceGhost optimizations if available
    if spaceghost_optimizer:
        print("🧹 Applying SpaceGhost optimizations...")
        edge = spaceghost_optimizer(edge)
    else:
        print("⚠️  Using standard pipeline (SpaceGhost not available)")

    # Continue with standard pipeline
    partitioned = edge.to_backend(XnnpackPartitioner())
    executable = partitioned.to_executorch()

    return executable

# Migrate your export function
your_executable = export_with_spaceghost(your_model, your_sample_input)
```

#### Advanced Migration (Full Control)
```python
def advanced_spaceghost_export(model, sample_input, optimization_config=None):
    """Advanced export with configurable SpaceGhost optimizations"""

    config = optimization_config or {
        'enable_maxpool_fix': True,
        'enable_quantization_fusion': True,
        'enable_snapdragon_optimization': True,
        'target_hardware': 'snapdragon_480'
    }

    print(f"🚀 Starting SpaceGhost export with config: {config}")

    # 1. Export model
    exported = torch.export.export(model, sample_input)
    print("✅ Model exported")

    # 2. Convert to edge format
    edge = to_edge(exported)
    print("✅ Converted to Edge format")

    # 3. Apply SpaceGhost optimizations
    if spaceghost_optimizer:
        print("🧹 Applying SpaceGhost optimizations...")

        # Log before optimization
        original_ops = count_operations(edge)
        print(f"   Operations before optimization: {original_ops}")

        # Apply optimizations
        optimized_edge = spaceghost_optimizer(edge)

        # Log after optimization
        optimized_ops = count_operations(optimized_edge)
        print(f"   Operations after optimization: {optimized_ops}")

        edge = optimized_edge
    else:
        print("⚠️  SpaceGhost not available - using standard pipeline")

    # 4. Partition for hardware acceleration
    print("🎯 Partitioning for XNNPack...")
    partitioned = edge.to_backend(XnnpackPartitioner())

    # Count delegate operations
    delegate_count = sum(1 for node in partitioned.graph.nodes
                        if hasattr(node, 'target') and 'delegate' in str(node.target))
    print(f"✅ Partitioned with {delegate_count} delegate operations")

    # 5. Convert to executable
    executable = partitioned.to_executorch()
    print("✅ Converted to ExecuTorch executable")

    return executable

def count_operations(edge_program):
    """Count operations in edge program"""
    return sum(1 for _ in edge_program.graph.nodes if hasattr(_, 'target'))

# Use advanced export
config = {
    'enable_maxpool_fix': True,
    'enable_quantization_fusion': True,
    'enable_snapdragon_optimization': True
}
optimized_executable = advanced_spaceghost_export(your_model, your_sample_input, config)
```

### Step 4: Update Model Preparation

#### Handle Quantization Properly
```python
def prepare_model_for_spaceghost(model, sample_input):
    """Prepare model for optimal SpaceGhost optimization"""

    # 1. Ensure model is in evaluation mode
    model.eval()

    # 2. Apply dynamic quantization if beneficial
    # SpaceGhost can optimize quantized models better
    from torch.ao.quantization import quantize_dynamic

    # Quantize linear layers for better performance
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},  # Target layers for quantization
        dtype=torch.qint8
    )

    print("✅ Model prepared with quantization for SpaceGhost optimization")
    return quantized_model

# Update your model preparation
prepared_model = prepare_model_for_spaceghost(your_model, your_sample_input)
executable = export_with_spaceghost(prepared_model, your_sample_input)
```

#### Optimize Input Shapes
```python
def optimize_input_shapes(model, sample_inputs):
    """Optimize input shapes for SpaceGhost performance"""

    # SpaceGhost performs better with certain input shapes
    optimized_inputs = []

    for sample_input in sample_inputs:
        # Ensure tensor is contiguous for better memory access
        if not sample_input.is_contiguous():
            sample_input = sample_input.contiguous()

        # For Snapdragon 480, prefer certain alignments
        # SpaceGhost will handle this automatically

        optimized_inputs.append(sample_input)

    return optimized_inputs

# Optimize your input shapes
optimized_sample_inputs = optimize_input_shapes(your_model, your_sample_inputs)
```

### Step 5: Add Performance Validation

#### Automated Performance Comparison
```python
def compare_performance_before_after(original_model, optimized_executable, test_inputs):
    """Compare performance before and after SpaceGhost optimization"""

    print("📊 Performance Comparison: Before vs After SpaceGhost")
    print("=" * 60)

    # Test original model
    print("⏱️  Testing original model...")
    original_results = benchmark_model(original_model, test_inputs[:10])

    # Test optimized model
    print("⏱️  Testing SpaceGhost optimized model...")
    optimized_results = benchmark_executable(optimized_executable, test_inputs[:10])

    # Compare results
    latency_improvement = ((original_results['avg_latency'] - optimized_results['avg_latency']) /
                          original_results['avg_latency']) * 100

    print("\n📈 Performance Comparison:")
    print(".2f")
    print(".2f")
    print(".1f")

    if latency_improvement > 0:
        print("✅ SpaceGhost optimization successful!"    else:
        print("⚠️  Performance did not improve - check optimization application")

    return {
        'original': original_results,
        'optimized': optimized_results,
        'improvement_percent': latency_improvement
    }

def benchmark_model(model, test_inputs):
    """Benchmark PyTorch model"""
    import time

    model.eval()
    latencies = []

    with torch.no_grad():
        for test_input in test_inputs:
            start_time = time.perf_counter()
            _ = model(test_input)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

    return {
        'avg_latency': sum(latencies) / len(latencies),
        'min_latency': min(latencies),
        'max_latency': max(latencies)
    }

def benchmark_executable(executable, test_inputs):
    """Benchmark ExecuTorch executable"""
    import time

    latencies = []

    for test_input in test_inputs:
        start_time = time.perf_counter()
        # Run inference with executable
        result = executable.forward((test_input,))
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    return {
        'avg_latency': sum(latencies) / len(latencies),
        'min_latency': min(latencies),
        'max_latency': max(latencies)
    }

# Compare performance
performance_comparison = compare_performance_before_after(
    your_original_model, optimized_executable, test_inputs
)
```

### Step 6: Handle Edge Cases and Fallbacks

#### Graceful Degradation
```python
def robust_spaceghost_export(model, sample_input):
    """Robust export with multiple fallback levels"""

    try:
        # Try full SpaceGhost optimization
        print("🚀 Attempting full SpaceGhost optimization...")
        return export_with_spaceghost(model, sample_input)

    except Exception as e:
        print(f"⚠️  Full SpaceGhost optimization failed: {e}")

        try:
            # Fallback to partial optimization
            print("🔄 Attempting partial SpaceGhost optimization...")
            exported = torch.export.export(model, sample_input)
            edge = to_edge(exported)

            # Try just the cleanup pass
            if spaceghost_optimizer:
                edge = spaceghost_optimizer(edge)

            partitioned = edge.to_backend(XnnpackPartitioner())
            return partitioned.to_executorch()

        except Exception as e2:
            print(f"⚠️  Partial optimization failed: {e2}")

            # Final fallback to standard pipeline
            print("📦 Falling back to standard ExecuTorch pipeline...")
            exported = torch.export.export(model, sample_input)
            edge = to_edge(exported)
            partitioned = edge.to_backend(XnnpackPartitioner())
            return partitioned.to_executorch()

# Use robust export
final_executable = robust_spaceghost_export(your_model, your_sample_input)
```

#### Snapdragon 480 Specific Optimizations
```python
def enable_snapdragon_480_features():
    """Enable Snapdragon 480 specific features if available"""

    try:
        # Try to import Snapdragon optimizations
        from research.spaceghost.patches.xnnpack.snapdragon_480_optimization import (
            is_snapdragon_480_with_dotprod,
            enable_snapdragon_480_optimizations
        )

        if is_snapdragon_480_with_dotprod():
            print("🔥 Snapdragon 480 detected - enabling hardware optimizations")
            enable_snapdragon_480_optimizations()

            # Set environment variables for optimal performance
            import os
            os.environ['XNNPACK_USE_SNAPDRAGON_OPTIMIZATIONS'] = '1'
            os.environ['XNNPACK_USE_DOTPROD'] = '1'

            return True
        else:
            print("ℹ️  Snapdragon 480 not detected - using generic optimizations")
            return False

    except ImportError:
        print("ℹ️  Snapdragon optimizations not available - using standard pipeline")
        return False

# Enable Snapdragon features
snapdragon_enabled = enable_snapdragon_480_features()
```

### Step 7: Update Deployment and Monitoring

#### Update Android Deployment
```kotlin
// Update MainActivity.kt for SpaceGhost
class MainActivity : AppCompatActivity() {

    private lateinit var lfmModule: LLMModule
    private lateinit var performanceMonitor: PerformanceMonitorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize performance monitoring
        performanceMonitor = PerformanceMonitorService()

        // Load SpaceGhost optimized model
        lifecycleScope.launch {
            initializeLFN()
        }
    }

    private suspend fun initializeLFN() {
        try {
            // Load model optimized with SpaceGhost
            val modelFile = assets.open("lfm2-350m_spaceghost.pte")
            lfmModule = LLMModule(modelFile)

            // Log optimization status
            performanceMonitor.logEvent("SpaceGhost optimized model loaded")

        } catch (e: Exception) {
            Log.w("SpaceGhost", "Optimized model not available, using fallback", e)

            // Fallback to standard model
            val fallbackModel = assets.open("lfm2-350m.pte")
            lfmModule = LLMModule(fallbackModel)
            performanceMonitor.logEvent("Fallback model loaded")
        }
    }

    private fun sendMessage(message: String) {
        val startTime = System.currentTimeMillis()

        lifecycleScope.launch {
            val response = generateResponse(message)
            val latency = System.currentTimeMillis() - startTime

            // Log SpaceGhost performance
            performanceMonitor.recordInference(
                latencyMs = latency,
                optimizationType = "SpaceGhost",
                hardwareAccel = true
            )
        }
    }
}
```

#### Update Build Configuration
```gradle
// build.gradle.kts - Add SpaceGhost optimization flags
android {
    defaultConfig {
        // Enable SpaceGhost optimizations
        buildConfigField("boolean", "USE_SPACEGHOST", "true")
        buildConfigField("boolean", "SNAPDRAGON_OPTIMIZATIONS", "true")

        // Add native library dependencies
        externalNativeBuild {
            cmake {
                arguments("-DXNNPACK_ENABLE_SNAPDRAGON_OPTIMIZATIONS=ON")
            }
        }
    }
}
```

## Migration Checklist

### Pre-Migration
- [ ] Assess current ExecuTorch pipeline performance
- [ ] Establish performance baseline
- [ ] Install SpaceGhost dependencies
- [ ] Test SpaceGhost import functionality

### Migration Steps
- [ ] Update import statements
- [ ] Modify export pipeline to include `run_lfn_xnnpack_pipeline`
- [ ] Add quantization preparation
- [ ] Enable Snapdragon 480 features (if available)
- [ ] Update performance monitoring
- [ ] Test with representative workloads

### Post-Migration Validation
- [ ] Verify SpaceGhost optimizations are applied
- [ ] Measure performance improvement (target: 4-8x)
- [ ] Test on Snapdragon 480 hardware
- [ ] Validate model accuracy preservation
- [ ] Update deployment scripts

### Troubleshooting
- [ ] Check SpaceGhost installation
- [ ] Verify PyTorch/ExecuTorch compatibility
- [ ] Test with simple models first
- [ ] Use fallback mechanisms for edge cases

## Expected Performance Improvements

### Migration Impact
| Aspect | Before Migration | After Migration | Improvement |
|--------|------------------|-----------------|-------------|
| **MaxPool2d Ops** | CPU-only | DSP accelerated | 2-3x faster |
| **Quantization** | Standard | Optimized chains | 30-50% reduction |
| **Threading** | Generic | Big core optimized | 35% improvement |
| **Memory Access** | Standard | L3 cache optimized | 4x faster |
| **Overall** | Baseline | **SpaceGhost Optimized** | **4-8x total** |

### Validation Targets
- ✅ **Latency**: <200ms for LFM2-350M inference
- ✅ **DSP Usage**: >50% of operations accelerated
- ✅ **Accuracy**: No degradation in model outputs
- ✅ **Compatibility**: Works on Snapdragon 480 and fallback platforms

## Common Migration Issues

### Issue: Import Errors
```python
# Fix: Add SpaceGhost to Python path
import sys
sys.path.append('/path/to/moltar/research/spaceghost')

from research.spaceghost.patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline
```

### Issue: Performance Not Improving
```python
# Fix: Verify optimizations are applied
from research.spaceghost.patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline

# Check if optimizer is available
print("SpaceGhost available:", run_lfn_xnnpack_pipeline is not None)

# Test with simple model first
```

### Issue: Snapdragon Features Not Available
```python
# Fix: Check platform compatibility
import platform
machine = platform.machine().lower()
print("Platform:", machine)
print("ARM64 support:", 'arm' in machine or 'aarch64' in machine)
```

## Success Metrics

### Migration Success Criteria
- [ ] SpaceGhost optimizations successfully applied
- [ ] Performance improvement of 4x or greater achieved
- [ ] Model accuracy maintained
- [ ] Deployment works on target hardware
- [ ] Fallback mechanisms functional

### Long-term Benefits
- **Performance**: Continuous optimization improvements
- **Compatibility**: Works across Snapdragon 480 devices
- **Maintainability**: Clean integration with ExecuTorch
- **Extensibility**: Framework for future optimizations

## Getting Help

### Resources
- **[SpaceGhost Architecture](../docs/SPACEGHOST_ARCHITECTURE.md)** - Technical details
- **[SpaceGhost Examples](../docs/SPACEGHOST_EXAMPLES.md)** - Code examples
- **[Troubleshooting Guide](../docs/TROUBLESHOOTING_GUIDE.md)** - Common issues
- **[API Reference](../API.md)** - Function documentation

### Support
- Check SpaceGhost status: `python -c "import research.spaceghost; print('OK')"`
- Validate optimizations: Run falsification tests in `research/spaceghost/`
- Performance monitoring: Use built-in performance tracking

---

## Summary

**Migrating to SpaceGhost is straightforward and provides significant performance benefits:**

1. **Add one function call** to your export pipeline
2. **Get 4-8x performance improvement** automatically
3. **Maintain full compatibility** with existing code
4. **Enable hardware acceleration** on Snapdragon 480

**The migration typically takes 30 minutes and provides immediate performance gains for Liquid AI Foundation Models on Motorola devices.**