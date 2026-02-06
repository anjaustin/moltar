# GPU Ternary GEMM + Spline Activation Results

**Date**: 2026-02-04
**Probe**: `research/trix_fabric/probes/gpu_ternary_gemm_probe.c`
**Shader**: `research/trix_fabric/probes/gpu_ternary_gemm.comp`

## Hypothesis

GPU with 8 GB/s dedicated DRAM path could outperform CPU for ternary GEMM when weights stream through GPU's buffer path, freeing CPU for attention.

## Results

### Test 1: FFN-sized (4096 x 1024)
```
CPU Ternary GEMM:  519 us  (16.16 GOP/s, 3.06 GB/s effective)
GPU Ternary GEMM: 3268 us  ( 2.57 GOP/s, 0.40 GB/s effective)

CPU is 6.3x FASTER
```

### Test 2: Larger (8192 x 2048)  
```
CPU Ternary GEMM:  2008 us  (16.71 GOP/s, 3.15 GB/s effective)
GPU Ternary GEMM: 27543 us  ( 1.22 GOP/s, 0.19 GB/s effective)

CPU is 13.7x FASTER (worse at larger sizes!)
```

## Analysis

### Why GPU Lost

1. **Shader inefficiency**: The compute shader has a loop decoding 16 trits per thread, with many branches. PowerVR's SIMD architecture penalizes divergent branches heavily.

2. **Memory access pattern**: The shader reads packed weights in a way that doesn't coalesce well - each thread reads from different memory locations.

3. **Host-visible memory**: All buffers use host-visible/coherent memory for easy mapping. This goes through a slow path on mobile GPUs.

4. **No subgroup operations**: The reduction uses shared memory barriers instead of subgroup shuffle/reduce, adding latency.

5. **FP16 unpacking overhead**: The shader unpacks FP16 pairs manually instead of using native FP16 compute.

### Why CPU Won

1. **NEON TBL instruction**: Single instruction decodes 16 trits via table lookup - exactly what ternary needs.

2. **Dot product intrinsic**: `vdotq_s32` does 4x4 dot products in one instruction.

3. **Sequential access**: CPU reads weights sequentially, L1 prefetcher keeps up.

4. **Cache efficiency**: 99% L1 hit rate measured in earlier profiling.

## Conclusion: GPU Ternary GEMM is NOT viable

The Cortex-A78's NEON unit with TBL+DOT instructions is purpose-built for this workload. The PowerVR GPU lacks equivalent primitives and its general-purpose compute model adds too much overhead.

## What Would Change This?

1. **Subgroup ternary decode**: If PowerVR had a native 2-bit decode instruction (like NEON TBL)
2. **Device-local memory**: Using dedicated GPU memory instead of host-visible
3. **Batched inference**: Multiple tokens at once (but this is rare in autoregressive generation)
4. **Different quantization**: Maybe INT8 with texture sampling would be better on GPU

## Next Steps

Given this result, the winning strategy is:
- **CPU does everything** for single-token inference
- **GPU only for batched vision encoding** (SigLIP patches)
- Focus optimization efforts on CPU kernels

The GPU's 8 GB/s dedicated path is real but doesn't help for compute-bound kernels.
