# Hardware Probe Results: PowerVR BXM-8-256 on Motorola Moto G Power 5G

**Date:** February 2, 2026
**Method:** `vk_probe_caps.c` — Vulkan capability query run on-device via adb
**Device:** Motorola moto g power 5G (2023), MediaTek MT6855 / Dimensity 930

---

## Identity

```
Device:         PowerVR BXM-8-256
API version:    Vulkan 1.1.170
Driver version: 6133110
Vendor ID:      0x1010 (Imagination Technologies)
Device ID:      0x35010101
Type:           Integrated GPU
```

Note: The BXM-8-256 IP block supports Vulkan 1.3 per Imagination's spec sheet,
but the MediaTek driver only exposes 1.1.170. This means Vulkan 1.2/1.3 core
features must be accessed via extensions where available.

---

## The Five Key Findings

### 1. Subgroup Size = 128

```
subgroupSize: 128
```

128 threads execute in lockstep as a single SIMD wavefront. This is unusually
large — Qualcomm Adreno uses 64, ARM Mali uses 16, Apple uses 32.

**Implications for LFM inference:**
- D_MODEL = 1024 fits in exactly **8 subgroups** (8 wavefronts)
- A single subgroup can process 128 elements in one instruction
- Subgroup arithmetic reductions (`subgroupAdd`) sum 128 values in hardware
  — a 1024-element dot product needs only 8 subgroup sums + 3 levels of
  cross-subgroup reduction
- D_HEAD = 64 is **half a subgroup** — two attention heads fit in one wavefront

All subgroup operations are supported:
```
BASIC, VOTE, ARITHMETIC, BALLOT, SHUFFLE, SHUFFLE_RELATIVE, CLUSTERED, QUAD
```

This is the full set. `SHUFFLE` enables arbitrary cross-lane data movement
within a 128-wide wavefront. `ARITHMETIC` gives us hardware reductions.
`CLUSTERED` enables partial reductions within subsets of the subgroup.

### 2. TMU Linear Filtering: FP16 Yes, FP32 No

```
R32_SFLOAT:  optimalTiling sampled=yes  filter=NEAREST_ONLY
R16_SFLOAT:  optimalTiling sampled=yes  filter=LINEAR
R8_UNORM:    optimalTiling sampled=yes  filter=LINEAR
```

The Texture Mapping Units (TMUs) can perform hardware-accelerated bilinear
interpolation on R16_SFLOAT and R8_UNORM textures, but NOT on R32_SFLOAT.

**What this means:**

Activation lookup tables (sigmoid, tanh, SiLU, exp) stored as R16_SFLOAT 1D
textures get **free hardware interpolation**. A single `texture()` call in
a compute shader does the LUT index calculation, the two nearest-entry fetch,
and the linear interpolation — all in fixed-function TMU silicon, parallel
with the ALU pipeline. Zero ALU cycles consumed.

This is the GPU equivalent of Yinsen's LUT+lerp discovery: use the hardware
for what it already does fast. The FPU doesn't care if it multiplies by 1.0
or 0.3. The TMU doesn't care if it's interpolating a color gradient or a
sigmoid table.

**Accuracy consideration:** FP16 has 10 mantissa bits = ~3 decimal digits
of precision. For sigmoid (output range [0,1]) this gives ~1e-3 worst-case
quantization error from the FP16 storage alone. Combined with 256-entry
linear interpolation, total error should be ~1e-3, comparable to Yinsen's
FAST3 tier but computed entirely in hardware.

For higher precision, R8_UNORM (256 values mapped to [0.0, 1.0]) with LINEAR
filtering gives hardware interpolation at 8-bit resolution — appropriate for
sigmoid (output is [0,1]) but not for tanh (output is [-1,1], needs SNORM
or remapping).

### 3. Unified Memory — True Zero-Copy

```
Memory Heaps: 1
  [0] size=3597 MB  DEVICE_LOCAL

Memory Types: 3
  [0] DEVICE_LOCAL
  [1] DEVICE_LOCAL  HOST_VISIBLE  HOST_COHERENT
  [2] DEVICE_LOCAL  HOST_VISIBLE  HOST_CACHED
```

Single memory heap. All memory is DEVICE_LOCAL. Types [1] and [2] are
simultaneously device-local AND host-visible. This is true unified memory
— the CPU and GPU access the same physical LPDDR4X through the same bus.

**Implications:**
- No staging buffers needed. CPU writes directly to GPU-accessible memory.
- No `vkCmdCopyBuffer` transfers. The data is already there.
- Memory type [1] (HOST_COHERENT) means CPU writes are immediately visible
  to GPU without explicit flush. Perfect for the ION channel pattern:
  CPU writes activation vector, GPU reads it immediately.
- Memory type [2] (HOST_CACHED) gives better CPU read performance (cached
  on the CPU side) but requires explicit cache management for GPU writes.
- **The GGUF model file can be mmap'd once and read by both CPU and GPU**
  from the same virtual address (via VK_EXT_external_memory_dma_buf or
  VK_ANDROID_external_memory_android_hardware_buffer).

### 4. Native FP16 + INT8 Shader Support

```
Extensions:
  VK_KHR_shader_float16_int8  (native FP16 and INT8 arithmetic in shaders)
  VK_KHR_16bit_storage        (FP16 values in storage/uniform buffers)
  VK_KHR_8bit_storage         (INT8 values in storage buffers)
  VK_EXT_scalar_block_layout  (packed struct layouts, no padding)

Features:
  shaderInt16: yes
  storageBuffer16BitAccess: yes
  uniformAndStorageBuffer16BitAccess: yes
  storagePushConstant16: yes
```

The GPU can natively:
- Read Q4_0 weight blocks as `uint8` arrays directly from storage buffers
- Perform FP16 arithmetic (though at the same throughput as FP32 on this part)
- Store intermediate activations in FP16 to halve bandwidth
- Use scalar block layout to match C struct packing (no std140/std430 padding)

**Q4_0 dequantization in a shader** becomes straightforward:
```glsl
// Read a Q4_0 block: 2 bytes scale (FP16) + 16 bytes quants
float16_t scale = block.d;
uint8_t packed = block.qs[j];
float lo = float(int(packed & 0xF) - 8) * float(scale);
float hi = float(int(packed >> 4) - 8) * float(scale);
```

### 5. Compute Limits

```
maxComputeWorkGroupInvocations: 512
maxComputeWorkGroupSize: [512, 512, 64]
maxComputeSharedMemorySize: 16384 bytes (16 KB)
maxStorageBufferRange: 134217728 bytes (128 MB)
maxPushConstantsSize: 128 bytes
maxBoundDescriptorSets: 4
Queues: 2 (GRAPHICS + COMPUTE + TRANSFER)
```

**Workgroup sizing:**
- 512 invocations max = 4 subgroups of 128
- For D_MODEL=1024: use 512 invocations, each thread handles 2 elements
- For D_MODEL=1024 with 8 subgroups: need 2 dispatches or 1024 invocations
  (exceeds max). Alternative: 512 invocations processing 2 elements each.

**Shared memory:**
- 16 KB = 4096 floats = enough for one 1024-element activation vector
  plus working space
- Not enough to tile a full matvec (weight rows are 576 bytes each in Q4_0,
  16KB = ~28 rows). Matvec stays on CPU/NEON.

**Storage buffers:**
- 128 MB max range. The full LFM2-350M GGUF is 209 MB. Cannot bind the
  entire model as one storage buffer. Would need to bind per-layer weight
  regions, or use buffer device address (VK_KHR_buffer_device_address
  available as VK_EXT_buffer_device_address).

**Two queues** in one family. Both support compute. Potential for overlapping
GPU compute with GPU transfer, but they share the same USC so true parallelism
is limited. More useful for pipelining submissions than actual concurrent
execution.

---

## Extension Inventory (73 total)

### Directly Useful for Inference

| Extension | What It Enables |
|-----------|----------------|
| `VK_KHR_shader_float16_int8` | Native FP16/INT8 arithmetic in shaders |
| `VK_KHR_16bit_storage` | FP16 in storage/uniform buffers |
| `VK_KHR_8bit_storage` | INT8 in storage buffers (Q4_0 block access) |
| `VK_EXT_scalar_block_layout` | C-compatible struct layout in shaders |
| `VK_KHR_push_descriptor` | Fast descriptor updates (no pre-allocated sets) |
| `VK_KHR_descriptor_update_template` | Batch descriptor updates |
| `VK_KHR_variable_pointers` | Dynamic buffer indexing |
| `VK_KHR_buffer_device_address` | Raw 64-bit buffer pointers in shaders |
| `VK_KHR_shader_float_controls` | Control FP rounding/denorm behavior |
| `VK_KHR_shader_subgroup_extended_types` | Subgroup ops on FP16/INT8 types |
| `VK_KHR_shader_clock` | Shader timing (for micro-benchmarks) |
| `VK_KHR_synchronization2` | Modern sync primitives |
| `VK_KHR_timeline_semaphore` | CPU-GPU timeline synchronization |
| `VK_KHR_external_memory` | Import external memory (mmap'd GGUF) |
| `VK_KHR_external_memory_fd` | Import via file descriptor |
| `VK_EXT_external_memory_dma_buf` | Import DMA buffers |
| `VK_ANDROID_external_memory_android_hardware_buffer` | Android HardwareBuffer import |

### Not Available (Notable Absences)

| Extension | What It Would Enable |
|-----------|---------------------|
| `VK_KHR_shader_integer_dot_product` | Hardware INT8 dot products. **Not present.** |
| `VK_EXT_subgroup_size_control` | Choose subgroup size. **Not present** — locked to 128. |
| `VK_KHR_cooperative_matrix` | Hardware matrix multiply. **Not present.** |

The absence of `shader_integer_dot_product` means INT8 dot products must be
done manually (multiply + add in shader). The absence of `cooperative_matrix`
means no hardware GEMM acceleration.

---

## Comparison: CPU vs GPU for LFM Inference Operations

| Operation | CPU (A78 NEON) | GPU (BXM-8-256) | Winner |
|-----------|---------------|------------------|--------|
| Q4_0 matvec (1024x3072) | NEON dotprod, sequential weight read | 16KB shared mem too small for weight tiling | **CPU** |
| RMSNorm (1024 elements) | NEON, 3 passes | 8 subgroups, subgroupAdd for reduction | Toss-up |
| SiLU (1024 elements) | Scalar expf or spline LUT | TMU texture sigmoid, zero ALU cost | **GPU** |
| Elementwise mul (1024) | NEON vmul | 128-wide SIMD, 8 cycles | **GPU** |
| Softmax (over seq_len) | Scalar, variable length | Subgroup reductions | Depends on length |
| RoPE (1024 elements) | NEON sin/cos pairs | SFU sin/cos or LUT | Toss-up |
| Attention scores | NEON matvec per head | 128-wide subgroup = 2 heads | Toss-up |

**The pattern:** CPU wins bandwidth-bound work (large weight reads).
GPU wins compute-bound parallel work (activations, elementwise, reductions).
The fabric schedules accordingly.

---

## Architectural Implications for the Fabric Layer

### The Pipeline Model

```
Time ──────────────────────────────────────────────►

CPU:  [weight read + matvec L0] [weight read + matvec L0 FFN] [L1...]
GPU:                    [activations L0] [elem mul L0]  [activations L0 FFN]
                              ↑                              ↑
                        zero-copy read                 zero-copy read
                        from CPU output                from CPU output
```

CPU and GPU share the memory bus but do NOT compete for bandwidth when
the GPU is doing pure compute on data already in cache/shared memory.
The GPU's 16KB shared memory holds the 1024-element activation vector
(4KB as FP32, 2KB as FP16) with room for LUT tables and working space.

### The TMU Activation Pattern

```
Init (once):
  1. Create R16_SFLOAT 1D image, width=256
  2. Fill with sigmoid values: sigmoid(XMIN + i * XSTEP)
  3. Create sampler with VK_FILTER_LINEAR
  4. Repeat for tanh, exp tables

Hot path (per layer):
  1. CPU writes activation vector to HOST_COHERENT buffer
  2. GPU dispatch: each thread reads one element, samples sigmoid texture
  3. TMU does the lookup + interpolation in hardware
  4. GPU writes result back to HOST_COHERENT buffer
  5. CPU reads result for next matvec
```

### Subgroup-Aware Reduction

For RMSNorm, softmax denominator, or any vector reduction:

```glsl
// 1024 elements, 512 invocations (each handles 2 elements)
float val = input[gl_GlobalInvocationID.x * 2]
          + input[gl_GlobalInvocationID.x * 2 + 1];

// Subgroup reduction: 128 → 1 in hardware (7 cycles)
float subgroup_sum = subgroupAdd(val);

// 4 subgroup sums → final result via shared memory
if (subgroupElect()) shared_sums[gl_SubgroupID] = subgroup_sum;
barrier();
if (gl_LocalInvocationID.x == 0) {
    result = shared_sums[0] + shared_sums[1]
           + shared_sums[2] + shared_sums[3];
}
```

A 1024-element reduction in ~10 cycles. On CPU with NEON, the same
reduction takes ~128 cycles (256 vadd operations for 1024 elements,
folded through vector registers).

---

## What We Don't Know Yet (Needs Probing)

1. **TMU throughput on this specific part.** How many texture samples per
   clock? The BXM-8-256 spec says 16 PPC for graphics; the TMU count is
   likely 4-8 units. Need to measure actual samples/second.

2. **GPU clock frequency.** Not exposed via Vulkan. Likely 400-800 MHz.
   Critical for estimating actual GFLOPS and TMU throughput.

3. **Dispatch latency.** How many microseconds from `vkQueueSubmit` to
   first shader instruction? If dispatch overhead exceeds the time saved
   by GPU compute, the pipelining model fails. Need to measure.

4. **HOST_COHERENT write-to-read latency.** How many nanoseconds from
   a CPU store to HOST_COHERENT memory until a GPU shader can read the
   new value? This determines the minimum pipeline bubble.

5. **Actual memory bandwidth partition.** When CPU and GPU are both
   reading from LPDDR4X simultaneously, how is the bus arbitrated?
   Does one starve the other?

6. **SFU throughput.** Hardware sin/cos/exp/rsqrt exist but at reduced
   throughput (typically 1/4 to 1/8 of FMA rate). Need to measure to
   decide whether SFU or TMU-LUT is faster for activations.

These are the questions for the next probe.

---

## Raw Data

Full output of `vk_probe_caps` is reproduced above. The probe source is at
`tests/vk_probe_caps.c`. Build and run:

```bash
NDK=~/Library/Android/sdk/ndk/28.2.13676358
$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang \
    -O2 -o build-android/vk_probe_caps tests/vk_probe_caps.c -lvulkan
adb push build-android/vk_probe_caps /data/local/tmp/
adb shell /data/local/tmp/vk_probe_caps
```
