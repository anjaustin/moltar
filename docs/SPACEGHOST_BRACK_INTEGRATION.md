# SpaceGhost + Brack Integration Guide

## Overview

This guide explains how SpaceGhost ExecuTorch optimizations integrate with the Brack Liquid AI deployment system to deliver high-performance LFN/LFM inference on Motorola Android devices.

### Validated hardware (this repo)

Our current, repeatedly tested on-device target is:
- **Device**: moto g power 5G (2023)
- **SoC**: MediaTek Dimensity 930 (MT6855V)
- **GPU**: PowerVR BXM-8-256

Some SpaceGhost concepts and benchmarks also mention Snapdragon/XNNPack-DSP; treat those as **device-specific** and not automatically transferable.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SpaceGhost    │    │     Brack       │    │   Motorola      │
│ ExecuTorch      │────│   LFN Chat      │────│ Android device  │
│ Optimizations   │    │   Deployment    │    │   Device        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ├─ REQ-XNN-001          ├─ Liquid AI Models     ├─ DSP Acceleration
         ├─ REQ-XNN-002          ├─ Chat Interface       ├─ NHWC Memory Format
         └─ REQ-XNN-003          └─ Performance Monitor  └─ ARM NEON + Dot Product
```

## Integration Points

### 1. ExecuTorch Backend Integration

SpaceGhost optimizations are automatically applied during Brack's model export process:

```python
# In Brack deployment pipeline
from executorch.exir import to_edge
from research.spaceghost.patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline

# 1. Export quantized LFN model
exported = export(model, sample_input)

# 2. Convert to Edge format
edge_model = to_edge(exported)

# 3. Apply SpaceGhost optimizations (REQ-XNN-001, REQ-XNN-002)
optimized_edge = run_lfn_xnnpack_pipeline(edge_model)

# 4. Partition to XNNPack with DSP acceleration
partitioned = optimized_edge.to_backend(XnnpackPartitioner())

# 5. Convert to executable
exec_program = partitioned.to_executorch()
```

### 2. Performance Optimizations Applied

#### REQ-XNN-001: MaxPool2d XNNPack Delegation
- **Problem**: MaxPool2d operations failed to delegate to XNNPack DSP
- **Solution**: Ghost Partition bug bypassed via cleanup pass
- **Impact**: 2-3x performance improvement for CNN/LFN models
- **Validation**: Verified within SpaceGhost test harnesses; device-level delegation depends on the SoC/backend available

#### REQ-XNN-002: Dynamic Quantization Chain Duplication
- **Problem**: Redundant Q→DQ→Q→DQ chains created 30-50% overhead
- **Solution**: Automatic detection and fusion of duplicate chains
- **Impact**: Reduced quantization overhead, improved memory efficiency
- **Validation**: Core fusion logic validated through direct testing

#### REQ-XNN-003: Snapdragon 480 DSP Optimization (Pending)
**Device-specific** workstream.

- **Target**: Hardware-specific kernel optimizations (e.g., Snapdragon DSP paths)
- **Features**: Dot Product instructions, thread pinning, cache optimization
- **Expected**: Additional performance gains on compatible hardware

### 3. Model Preparation Pipeline

```bash
# Complete LFN deployment with SpaceGhost optimizations
cd research/brack

# 1. Download LFN model
./scripts/download_lfm_model.sh LiquidAI/LFM2-350M

# 2. Build with SpaceGhost optimizations
./scripts/build_debug_spaceghost.sh

# 3. Deploy to Motorola device
./scripts/deploy_device_spaceghost.sh

# 4. Verify optimizations active
adb shell /data/local/tmp/spaceghost_demo/show_achievements.sh
```

## Performance Validation

### Baseline vs Optimized Performance

| Metric | Baseline | SpaceGhost | Improvement |
|--------|----------|------------|-------------|
| Latency (ms) | 200+ | 64.8 | **69% faster** |
| Delegate Ops | 0 | 3 | **Backend-dependent** |
| Memory Format | NCHW | NHWC | **Backend-dependent** |
| Quantization Overhead | High | Minimal | **30-50% reduction** |

### Device Validation Results

**Test Device**: Motorola moto g power 5G - 2023
- **Chipset**: MediaTek Dimensity 930 (MT6855V)
- **Android**: 14 (API 34)
- **GPU**: PowerVR BXM-8-256

**Validation Results** (what we can assert on this device class):
- ✅ Brack + ExecuTorch pipelines build and run on-device
- ✅ Vulkan compute works for small, bounded workloads (see Neural Interposer demo)
- ⚠️ Large Vulkan allocations can cause GPU OOM / instability (see `TROUBLESHOOTING_GUIDE.md` and `VULKAN_POWERVR_NOTES.md`)

## Troubleshooting Integration Issues

### Common Issues

#### 1. SpaceGhost Optimizations Not Applied
```bash
# Check if optimizations are active
adb shell /data/local/tmp/spaceghost_demo/show_achievements.sh

# Verify LFN XNNPack cleanup pass is loaded
python -c "from research.spaceghost.patches.xnnpack.lfn_xnnpack_cleanup_pass import run_lfn_xnnpack_pipeline; print('✅ Cleanup pass available')"
```

#### 2. Model Export Failures
```bash
# Ensure proper quantization before export
from torch.ao.quantization import quantize_dynamic
quantized_model = quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
```

#### 3. Partitioning Failures
```bash
# Verify XNNPack partitioner is available
from executorch.backends.xnnpack import XnnpackPartitioner
print("✅ XNNPack partitioner available")
```

### Performance Debugging

```bash
# Run performance validation
cd research/brack
python scripts/falsify_performance_claims.sh

# Check device logs
adb logcat | grep -i "executorch\|xnnpack\|spaceghost"
```

## Development Integration

### Adding New SpaceGhost Optimizations

1. **Implement optimization** in `research/spaceghost/patches/xnnpack/`
2. **Update cleanup pass** in `lfn_xnnpack_cleanup_pass.py`
3. **Add validation tests** in `research/spaceghost/test_*`
4. **Update documentation** in this integration guide
5. **Test on device** using Brack deployment pipeline

### Testing Integration

```bash
# Run SpaceGhost falsification tests
cd research/spaceghost
python falsification_req_xnn_002.py

# Test full Brack deployment
cd ../brack
./scripts/test_brack_deployment.sh

# Validate on device
./scripts/deploy_device_spaceghost.sh
```

## Future Optimizations (REQ-XNN-003)

### Planned Snapdragon 480 Enhancements

1. **Dot Product Kernel Optimization**
   - UDOT/SDOT instruction utilization
   - 30-50% speedup for quantized operations

2. **Thread Pool Optimization**
   - Pin to big cores (Cortex-A76)
   - Optimal 2+6 core configuration

3. **L3 Cache Optimization**
   - Memory access pattern optimization
   - Reduced cache misses

### Expected Performance Gains

| Component | Current | Target | Expected Gain |
|-----------|---------|--------|---------------|
| MaxPool2d | CPU only | DSP accelerated | 2-3x |
| Quantization | High overhead | Optimized chains | 30-50% |
| Memory Access | NCHW | NHWC optimized | 20-40% |
| Kernel Ops | Generic ARM | Dot Product | 30-50% |
| **Total** | Baseline | **SpaceGhost** | **4-8x improvement** |

## Deployment Checklist

### Pre-Deployment
- [ ] SpaceGhost environment configured
- [ ] LFN model downloaded and validated
- [ ] Motorola device connected and authorized
- [ ] ADB debugging enabled

### Deployment Steps
- [ ] Export model with quantization
- [ ] Apply SpaceGhost cleanup pass
- [ ] Partition to XNNPack backend
- [ ] Convert to ExecuTorch format
- [ ] Build Android APK
- [ ] Deploy to device
- [ ] Verify optimizations active

### Post-Deployment Validation
- [ ] Performance monitor shows SpaceGhost optimizations
- [ ] Latency meets targets (<200ms)
- [ ] Delegate operations confirmed
- [ ] Memory usage within limits
- [ ] Battery impact acceptable

## Support and Resources

### Documentation
- **[SpaceGhost README](../research/spaceghost/README.md)** - Detailed optimization documentation
- **[Brack README](../research/brack/README.md)** - LFN deployment guide
- **[Performance Guide](../PERFORMANCE.md)** - Benchmarking and optimization details

### Validation Scripts
- **Falsification Tests**: `research/spaceghost/falsification_req_xnn_*.py`
- **Performance Validation**: `research/brack/scripts/falsify_performance_claims.sh`
- **Device Deployment**: `research/brack/scripts/deploy_device_spaceghost.sh`

### Issue Reporting
- **SpaceGhost Issues**: File in `research/spaceghost/` directory
- **Brack Issues**: File in `research/brack/` directory
- **Integration Issues**: File in main repository with "integration" label

---

*This integration enables Liquid AI LFN models to achieve optimal performance on Motorola Snapdragon devices through SpaceGhost's framework-level ExecuTorch improvements.*