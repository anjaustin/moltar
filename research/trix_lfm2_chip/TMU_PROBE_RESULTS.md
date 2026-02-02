# TMU Activation Probe Results: PowerVR BXM-8-256

**Date:** February 2, 2026  
**Probe:** `tests/vk_tmu_probe.c`  
**Device:** Motorola moto g power 5G (2023), MediaTek MT6855 / Dimensity 930  
**GPU:** PowerVR BXM-8-256, Vulkan 1.1.170  

---

## Raw Results

| Test | Mean (us) | Min (us) | Max (us) | Max Error |
|------|-----------|----------|----------|-----------|
| NOP dispatch (1 invocation) | 386.2 | 182.2 | 727.7 | — |
| ALU sigmoid (SFU exp) | 428.7 | 223.3 | 877.5 | 1.79e-7 |
| Buffer LUT + manual lerp | 423.4 | 194.1 | 725.5 | 2.90e-4 |
| TMU texture (R16_SFLOAT + LINEAR) | 409.2 | 198.2 | 749.3 | 9.69e-4 |

Configuration: 1024 elements, 50 warmup, 500 timed iterations, 256-entry LUT.
Input: uniform distribution over [-10, +10].

---

## Analysis

### Dispatch Latency: The Dominant Cost

The NOP shader (single invocation, writes one constant) takes **386 us mean / 182 us min**.
This is the floor — the cost of `vkQueueSubmit` → `vkWaitForFences` round-trip on this
driver/hardware combination.

The actual compute cost of each strategy, isolated from dispatch overhead:

| Strategy | Compute cost (mean - NOP mean) | Per-element |
|----------|-------------------------------|-------------|
| ALU sigmoid | ~43 us | ~42 ns |
| Buffer LUT + lerp | ~37 us | ~36 ns |
| TMU texture | ~23 us | ~22 ns |

TMU is ~1.9x faster than ALU for the compute portion. But the compute portion is
**10-17x smaller** than dispatch overhead.

### Accuracy

| Strategy | Max Error | Notes |
|----------|-----------|-------|
| ALU sigmoid | 1.79e-7 | Full FP32 precision, SFU or ALU exp |
| Buffer LUT 256 | 2.90e-4 | Linear interpolation quantization |
| TMU R16_SFLOAT 256 | 9.69e-4 | FP16 quantization + linear interp |

All three are within Q4_0 quantization noise (~0.01 for 4-bit weights). The 9.69e-4
TMU error is acceptable for inference.

### Impact on Fabric Architecture

**Per-layer GPU offload is NOT viable.**

Arithmetic:
- 32 dispatches per token (SiLU + softmax + elementwise across 16 layers)
- 32 × 386 us = **12.4 ms dispatch overhead per token**
- At 40 tok/s target = 25 ms/tok budget
- Dispatch alone consumes **50% of budget** with zero useful compute

Even at minimum dispatch latency (182 us):
- 32 × 182 = 5.8 ms = **23% of budget** — still too high for the marginal
  benefit of offloading ~1 ms of activation compute.

### When GPU IS Viable

The GPU becomes worthwhile when:

1. **Fused multi-op kernels**: One dispatch doing RMSNorm + SiLU + mul + residual
   for an entire layer. ~4000+ elements of work per dispatch. Amortized dispatch
   cost drops to ~10% of compute.

2. **Matvec offload**: D_MODEL × FFN_HIDDEN = 1024 × 4608 = 4.7M multiply-adds.
   Even at 386 us dispatch, this is viable IF the GPU can compute the matvec
   faster than NEON. (Unlikely for Q4_0 due to 16KB shared memory constraint —
   only 28 weight rows fit in shared memory.)

3. **Batch > 1**: Multiple tokens processed simultaneously. Elementwise dimension
   grows from 1024 to batch×1024. At batch=16, the GPU's SIMD width advantage
   materializes.

4. **Async pipeline with pre-recorded command buffers**: Pre-record N command
   buffers, submit all at once, overlap GPU compute with CPU compute on different
   layers. This turns the dispatch into a one-time cost rather than per-layer.

### Decision

**CPU-first architecture with NEON optimization is the correct path for batch=1
single-token generation on this hardware.**

The GPU's value on this specific SoC is limited by:
- 386 us dispatch latency (MediaTek PowerVR driver overhead)
- 16 KB shared memory (too small for weight tiling)
- Shared LPDDR4X bus (GPU and CPU compete for bandwidth)

The GPU's strengths (128-wide subgroups, TMU linear filtering, native FP16) become
assets only in batched or fused-kernel scenarios.

### Revised Strategy

1. **Phase 1**: NEON-optimized Q4_0 matvec (the 80% of compute)
2. **Phase 2**: NEON-optimized activations (Yinsen LUT+lerp pattern via NEON, not GPU)
3. **Phase 3**: Operator fusion (RMSNorm+matvec, SiLU+mul in SwiGLU)
4. **Phase 4**: Big/little core scheduling (A78 for matvec, A55 for misc)
5. **Phase 5 (stretch)**: GPU fused-layer kernel with pre-recorded command buffers

---

## Probe Source

```
tests/vk_tmu_probe.c    — C probe with embedded SPIR-V
shaders/dispatch_nop.comp — NOP shader for dispatch latency
shaders/sigmoid_alu.comp  — ALU sigmoid via SFU exp
shaders/sigmoid_lut_buf.comp — Buffer LUT + manual lerp
shaders/sigmoid_tmu.comp — TMU texture LUT with hardware LINEAR filter
```

Build and run:
```bash
NDK=~/Library/Android/sdk/ndk/28.2.13676358
CC=$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang
glslc --target-env=vulkan1.1 -o shaders/dispatch_nop.spv shaders/dispatch_nop.comp
glslc --target-env=vulkan1.1 -o shaders/sigmoid_alu.spv shaders/sigmoid_alu.comp
glslc --target-env=vulkan1.1 -o shaders/sigmoid_lut_buf.spv shaders/sigmoid_lut_buf.comp
glslc --target-env=vulkan1.1 -o shaders/sigmoid_tmu.spv shaders/sigmoid_tmu.comp
# Generate embedded headers
xxd -i shaders/dispatch_nop.spv > shaders/dispatch_nop_spv.h
xxd -i shaders/sigmoid_alu.spv > shaders/sigmoid_alu_spv.h
xxd -i shaders/sigmoid_lut_buf.spv > shaders/sigmoid_lut_buf_spv.h
xxd -i shaders/sigmoid_tmu.spv > shaders/sigmoid_tmu_spv.h
# Compile
$CC -O2 -Ishaders -o vk_tmu_probe tests/vk_tmu_probe.c -lvulkan -lm
adb push vk_tmu_probe /data/local/tmp/
adb shell /data/local/tmp/vk_tmu_probe
```
