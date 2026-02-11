# Moltar: MT6855 GPU Compute & MDLA Hardware Research

## Project Overview

Moltar is a hardware research project targeting the MediaTek Dimensity 930 (MT6855) SoC on a Motorola Moto G Power 5G 2023 (codename: devonn). The project had two phases:

1. **Phase 1 (MDLA Unlock)** — Attempted to power on the hidden MDLA 2.1 (MediaTek Deep Learning Accelerator). Failed due to SPM firmware lockout — documented here as a definitive reference.

2. **Phase 2 (GPU Compute)** — Comprehensive enumeration and benchmarking of the PowerVR BXM-8-256 GPU's compute capabilities via OpenCL 3.0, Vulkan 1.1, and NNAPI/NeuroPilot.

## Device

| Property | Value |
|----------|-------|
| Device | Motorola Moto G Power 5G 2023 |
| Codename | devonn |
| SoC | MT6855 / Dimensity 930 |
| CPU | 2x Cortex-A78 @ 2.2 GHz + 6x Cortex-A55 @ 2.0 GHz |
| GPU | PowerVR BXM-8-256 (Imagination Volcanic B-Series) |
| RAM | ~3.5 GB LPDDR4X |
| Android | 14 (SDK 34) |
| Kernel | 5.10.205 (CFI_CLANG + MODVERSIONS enabled) |
| Root | Magisk (unlocked bootloader) |

---

## Phase 1: MDLA/APU Unlock Attempt (FAILED)

### What We Proved

The APU (containing MDLA 2.1 + VPU) is disabled at **three independent hardware levels**, making software-only unlock impossible:

**1. Device Tree** — No APU nodes exist. No `apu_top`, `apusys`, `apu_conn`, or MDLA device tree entries. The `mt6363_vsram_apu` regulator has no `regulator-always-on` property.

**2. SPM Firmware (MD32 RISC core)** — The System Power Manager runs embedded firmware on an MD32 core that does NOT contain an APU wake handler. Setting `WAKEUP_APU` (bit 0) in `SPM_CROSS_WAKE_M01_REQ` (SPM+0x670) causes the MD32 to stay in its idle loop at PC=0x540. `OTHER_PWR_STATUS` bit 5 never asserts. This firmware cannot be modified at runtime.

**3. SPM Resource Masks** — All 5 APU resource request lines (APSRC, DDR, INFRA, SRCCLK, VRF18) are masked out in `SPM_SRC_MASK_0` (bits 5-9 = 0).

**Result**: The entire APU address space (0x19000000 - 0x190F2000) reads all zeros. Every register in APU_RPC, APU_CONN, APU_VCORE, MDLA0, APU_AO, APU_MBOX — all zero. The MTCMOS power switches are physically open and controlled exclusively by SPM firmware.

### What We Tried

- Direct SPM register writes (buck isolation release, SPM2APU_CON, CROSS_WAKE)
- ATF SMC calls (`MTK_SIP_APUSYS_CONTROL` = 0xC200051E, ops AFC_EN/WAKEUP_RPC/CG_EN) — ATF returned success but did nothing
- Unmasking APU resource requests in `SPM_SRC_MASK_0`
- RPC direct initialization (writes vanish — hardware unpowered)
- Full SPM `PWR_CON` space scan (0xE00-0xF48) — no `APU_PWR_CON` register exists
- Custom kernel module loading with CFI bypass (working pipeline, described below)

### Key APU Hardware Facts

| Item | Value |
|------|-------|
| VAPU buck | `mt6363_vbuck3` (ON at 750mV, always-on) |
| VSRAM_APU | `mt6363_vsram_apu` (regulator.23, DISABLED, 800mV) |
| SPM base | `0x1C001000` (not 0x10006000 like older SoCs) |
| APU RPC base | `0x190F0000` |
| APU power model | Callback-based via `register_apu_callback()` + cross-wake to SPM firmware |

### Kernel Module Loading Pipeline (Reusable)

We built a complete pipeline to compile, patch, and load custom kernel modules on this CFI-enabled kernel with MODVERSIONS. This is fully solved and reusable for any future kernel module work on this device.

**Challenges solved:**
- `CONFIG_CFI_CLANG=y` (strict, non-permissive) — solved with `cfi_stub.S` providing `__cfi_check`, `__cfi_check_fail`, and data pointers for `__cfi_jt_init_module`/`__cfi_jt_cleanup_module`
- `CONFIG_MODVERSIONS=y` — solved with `gen_versions.py` that generates correct CRC values extracted from vendor `.ko` files
- Vermagic mismatch (source=5.10.168, device=5.10.205) — solved with `objcopy` patching
- Must use `aarch64-linux-gnu-objcopy` (GNU), NOT `llvm-objcopy` (corrupts extended symbol index)

**Build command:**
```bash
make -C /home/ztflynn/001/moltar/kernel-mtk M=/home/ztflynn/001/moltar/mdla_probe \
  ARCH=arm64 CC=clang CROSS_COMPILE=aarch64-linux-gnu- CLANG_TRIPLE=aarch64-linux-gnu- \
  KCFLAGS="-msign-return-address=all -mbranch-protection=pac-ret+leaf -ffixed-x18 \
           -fsanitize=shadow-call-stack" \
  clean modules
```

**Patch pipeline:**
```bash
SYMS=$(aarch64-linux-gnu-objdump -t mdla_read.ko | grep "UND" | awk '{print $NF}' | tr '\n' ' ')
python3 gen_versions.py versions.bin module_layout $SYMS
cp mdla_read.ko mdla_read_patched.ko
aarch64-linux-gnu-objcopy --strip-debug mdla_read_patched.ko
aarch64-linux-gnu-objcopy --remove-section=__versions mdla_read_patched.ko
aarch64-linux-gnu-objcopy --add-section __versions=versions.bin \
  --set-section-flags __versions=alloc,contents,load,readonly mdla_read_patched.ko
aarch64-linux-gnu-objcopy --dump-section .modinfo=modinfo_orig.bin mdla_read_patched.ko
# Patch vermagic string: 5.10.168 -> 5.10.205 in modinfo_orig.bin -> modinfo_new.bin
aarch64-linux-gnu-objcopy --update-section .modinfo=modinfo_new.bin mdla_read_patched.ko
adb push mdla_read_patched.ko /data/local/tmp/mdla_read.ko
adb shell "su -c 'insmod /data/local/tmp/mdla_read.ko'"
```

---

## Phase 2: GPU Compute

### GPU Hardware

| Property | Value |
|----------|-------|
| Architecture | Imagination Volcanic B-Series |
| Model | PowerVR BXM-8-256 |
| Shader clusters | 8 clusters x 32 ALUs = 256 ALUs |
| Frequency | 390 MHz (base) - 950 MHz (boost), 37 OPP levels |
| Power | 289 mW - 1208 mW |
| Memory | Unified with CPU (~1.8 GB visible to GPU) |
| Max allocation | 449 MB |
| Local memory | 28 KB (OpenCL) / 16 KB (Vulkan) |
| Cache | 128 KB, 128-byte lines |
| Driver module | `pvrsrvkm_mt6855_sec` (1.46 MB, loaded) |
| GPUEB | Dedicated GPU firmware coprocessor for DVFS |
| Device node | `/dev/pvr_sync` (world accessible) |

### API Support

| API | Version | Library | Notes |
|-----|---------|---------|-------|
| OpenCL | **3.0** | `/vendor/lib64/mt6855/libPVROCL.so` (33.6 MB) | Full OpenCL C 3.0, EMBEDDED_PROFILE |
| Vulkan | **1.1.170** | `/vendor/lib64/hw/mt6855/vulkan.mtk.so` (843 KB) | Compute shaders, FP16/Int8 |
| OpenGL ES | 3.2 | `/vendor/lib64/egl/mt6855/libGLESv2_mtk.so` (1.9 MB) | Compute shaders |
| NNAPI | Feature Level 7 | `libneuron_adapter_mgvi.so` + runtime | GPU + Neuron backends |

### OpenCL Capabilities

```
Platform:    PowerVR (Imagination Technologies)
Profile:     EMBEDDED_PROFILE
Device:      PowerVR BXM-8-256
Compute:     1 CU (256 ALUs internally), max workgroup 1024
FP16:        Supported (cl_khr_fp16) — InfNaN, RoundZero, FMA
FP32:        Full — InfNaN, RoundZero, FMA
FP64:        Not supported
Int64:       Supported (cles_khr_int64)
SVM:         Coarse-grain buffer (OpenCL 2.0+)
Images:      8192x8192 2D, 8192x8192x2048 3D
Subgroups:   Yes (cl_khr_subgroups), size 128 (Vulkan)
SPIR/SPIR-V: Yes (cl_khr_spir, cl_img_spirv)
DMA-buf:     Yes (cl_arm_import_memory_dma_buf) — zero-copy
Gralloc:     Yes (cl_img_use_gralloc_ptr_v2) — zero-copy camera frames
```

**Full extension list:**
```
cl_khr_icd                    cl_khr_fp16
cl_img_spirv                  cles_khr_int64
cl_img_yuv_image              cl_khr_device_uuid
cl_khr_depth_images           cl_khr_priority_hints
cl_img_generate_mipmap        cl_khr_3d_image_writes
cl_img_cached_allocations     cl_khr_create_command_queue
cl_img_mem_properties         cl_img_mem_properties_relax_alloc_requirements
cl_khr_extended_versioning    cl_khr_image2d_from_buffer
cl_khr_byte_addressable_store cl_khr_local_int32_base_atomics
cl_khr_global_int32_base_atomics
cl_khr_local_int32_extended_atomics
cl_khr_global_int32_extended_atomics
cl_khr_subgroups              cl_khr_spir
cl_arm_import_memory          cl_arm_import_memory_dma_buf
cl_img_use_gralloc_ptr        cl_img_protected_content
cl_img_use_gralloc_ptr_v2
```

### Vulkan Compute Capabilities

```
Driver:         PowerVR B-Series Vulkan Driver v1.15@6133110
API:            Vulkan 1.1.170
Compute:        512 max invocations, 512x512x64 max size, 16 KB shared
Subgroup size:  128
FP16 shaders:   Yes (VK_KHR_shader_float16_int8)
Int8 shaders:   Yes
Int16/Int64:    Yes
16-bit storage: Full (buffers, push constants, I/O)
Geometry:       Yes
Tessellation:   Yes
ASTC/ETC2:      Yes
DMA-buf import: Yes (VK_EXT_external_memory_dma_buf)
```

### NNAPI / NeuroPilot

Three accelerator backends are registered:

| Backend | Role | INT8 Support | FP32 Support |
|---------|------|-------------|-------------|
| `mtk-neuron_shim` | NeuroPilot runtime (GPU-routed since APU is dead) | Yes (100% delegated) | Yes (100% delegated) |
| `mtk-gpu_shim` | Direct GPU compute | No (falls back to CPU) | Yes (100% delegated) |
| `nnapi-reference` | CPU reference implementation | Yes | Yes |

**Key libraries:**
```
/vendor/lib64/mt6855/libneuron_adapter_mgvi.so  (4.8 MB)
/vendor/lib64/mt6855/libneuron_runtime.5.so     (574 KB)
/vendor/lib64/libtflite_mtk.so                  (6.7 MB)
/vendor/lib64/libneuron_graph_delegate.mtk.so
```

---

## Benchmark Results

### Raw Compute

| Benchmark | Time | Throughput |
|-----------|------|-----------|
| SGEMM Tiled (FP32) 512^3 | 25.1 ms | **10.7 GFLOPS** |
| SGEMM Tiled (FP32) 1024^3 | 199.1 ms | **10.8 GFLOPS** |
| HGEMM Tiled (FP16) 512^3 | 17.1 ms | **15.7 GFLOPS** |
| FP16 Dot Product 16M (half8) | 5.34 ms | **12.6 GB/s** |
| Vector Add 48 MB (FP32) | 5.82 ms | **8.6 GB/s** |

### ML Inference (MobileNet v2 1.0 224)

| Backend | Model | Latency | FPS |
|---------|-------|---------|-----|
| CPU 1T (XNNPACK) | INT8 | 44.2 ms | 23 |
| CPU 4T (XNNPACK) | INT8 | 15.7 ms | 64 |
| CPU 4T (XNNPACK) | FP32 | 19.2 ms | 52 |
| **NNAPI auto** | **INT8** | **13.6 ms** | **73** |
| NNAPI neuron_shim | FP32 | 23.3 ms | 43 |
| TFLite GPU delegate (OpenCL) | FP32 | 20.5 ms | 49 |

### Practical Workloads

| Workload | GPU Time | GPU Rate | CPU 1T Time | GPU Speedup |
|----------|----------|----------|-------------|-------------|
| 3x3 Gaussian Blur 1080p | 4.25 ms | 488 MP/s, 235 FPS | 2.8 ms | 0.66x (CPU wins) |
| 5x5 Gaussian Blur 1080p | 9.11 ms | 228 MP/s, 110 FPS | ~7 ms | ~0.8x (tie) |
| Sobel Edge Detection 1080p | 5.50 ms | 377 MP/s, 182 FPS | 2.6 ms | 0.47x (CPU wins) |
| RGB to YUV 4K | 11.22 ms | 739 MP/s, 89 FPS | 25.6 ms | **2.3x GPU** |
| Histogram 256-bin 1080p | 4.77 ms | 210 FPS | — | — |
| SHA-256 batch (256K x 64B) | 24.90 ms | 10.5 MHash/s | 27.5 ms | **1.1x GPU** |
| FFT 1M-point complex | 32.61 ms | 32.2 MS/s | — | — |
| Bilateral Filter 720p | 95.10 ms | 9.7 MP/s, 11 FPS | — | — |
| SGEMM 512^3 (FP32) | 25.1 ms | 10.7 GFLOPS | 389 ms | **15.5x GPU** |
| HGEMM 512^3 (FP16) | 17.1 ms | 15.7 GFLOPS | — | **22.7x vs CPU FP32** |

### Analysis

**GPU wins decisively on:**
- Matrix multiply / linear algebra (15-23x speedup)
- ML inference via NNAPI (73 FPS INT8 MobileNet v2)
- Large-frame color conversion (4K+ resolution, 2.3x)
- FFT / signal processing (32M samples/s)

**CPU wins on:**
- Simple image filters at 1080p or below (NEON auto-vectorization is fast, GPU transfer overhead isn't amortized)
- SHA-256 (register-heavy, no intra-hash parallelism)

**Crossover point:** The GPU becomes faster than CPU for image processing at roughly 4K resolution or 5x5+ convolution kernels. For compute-heavy workloads (GEMM, FFT), GPU always wins.

---

## Where the GPU Excels (Practical Applications)

### Tier 1: Ready to Build

**Real-time ML inference** — Camera to MobileNet/YOLO to overlay. All infrastructure works: NNAPI delegates fully functional, 50-73 FPS sustained inference, INT8 quantization preferred.

**Custom camera processing pipeline** — Chain OpenCL kernels: capture raw, denoise (bilateral), color convert, classify. Zero-copy via `cl_arm_import_memory_dma_buf` and Gralloc integration.

### Tier 2: Viable with Effort

**On-device LLM inference** — FP16 GEMM at 15.7 GFLOPS enables small quantized models (TinyLlama, Phi-2 INT4/INT8). Limited by 449 MB max alloc and ~1.8 GB total GPU memory. Token generation would be slow but functional.

**Audio / SDR processing** — FFT at 32M samples/s supports wideband FM demodulation, real-time spectrograms, filter bank processing.

### Tier 3: Possible but Limited

**Compute offload server** — Phone as headless GPU compute node over ADB/SSH. Accept OpenCL kernels, execute, return results.

**Hash computation** — 10.5 MHash/s SHA-256. Not competitive for mining but usable for security research workloads.

---

## File Structure

```
moltar/
├── MOLTAR.md                  # This document
├── boot.img                   # Magisk-patched boot image
├── dtbo_backup.img            # DTBO backup
│
├── gpu_probe/                 # Phase 2: GPU compute tools
│   ├── clinfo.c               # OpenCL capability enumeration (dlopen-based)
│   ├── clinfo                 # Compiled aarch64 binary
│   ├── gpu_bench.c            # SGEMM/HGEMM/vecadd/reduction benchmarks
│   ├── gpu_bench              # Compiled aarch64 binary
│   ├── workloads.c            # Practical workload suite (10 kernels)
│   ├── workloads              # Compiled aarch64 binary
│   ├── cpu_baseline.c         # CPU comparison benchmarks
│   ├── cpu_baseline           # Compiled aarch64 binary
│   ├── benchmark_model        # TFLite benchmark tool (prebuilt)
│   ├── mobilenet_v2_quant.tflite  # MobileNet v2 INT8 (3.6 MB)
│   ├── mobilenet_v2_fp32.tflite   # MobileNet v2 FP32 (14 MB)
│   └── libOpenCL.so           # Pulled from device for reference
│
├── mdla_probe/                # Phase 1: Kernel module tools
│   ├── cfi_stub.S             # CFI bypass stubs for module loading
│   ├── mdla_read_main.c       # APU probe module (cross-wake, SPM)
│   ├── mdla_read.c            # Original register reader
│   ├── Makefile               # Cross-compilation for arm64
│   ├── gen_versions.py        # MODVERSIONS CRC generator
│   ├── patch_module.py        # Module patching automation
│   ├── mdla_read_patched.ko   # Last built/patched module
│   └── *.bin                  # Intermediate patch artifacts
│
├── mdla_unlock/               # Early MDLA unlock attempts
│   ├── mdla_module.c          # Minimal MDLA driver
│   └── *.md                   # Status documents
│
├── modules/                   # Vendor kernel modules (reference)
│   ├── apusys.ko              # 1.48 MB (unused, APU dead)
│   └── apu_top.ko             # 204 KB (unused, APU dead)
│
├── kernel-mtk/                # Motorola kernel source (5.10.168)
├── kbuild/                    # Kernel build config
│   └── .config                # Device kernel config
├── kernel_headers/            # Extracted kernel headers
└── toolchain/                 # Build toolchain
```

## Build Requirements

**Host tools:**
- Android NDK r27c (`/opt/android-ndk-r27c/`)
- `aarch64-linux-gnu-gcc` 11.4+ (for kernel modules)
- `clang` (for kernel modules with CFI)
- `aarch64-linux-gnu-objcopy` (GNU, NOT llvm — critical)
- Python 3 (for `gen_versions.py`)

**Building GPU probe tools:**
```bash
NDK=/opt/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin
$NDK/aarch64-linux-android34-clang -O2 -lm -o gpu_probe/clinfo gpu_probe/clinfo.c -ldl
$NDK/aarch64-linux-android34-clang -O2 -lm -o gpu_probe/gpu_bench gpu_probe/gpu_bench.c -ldl
$NDK/aarch64-linux-android34-clang -O2 -lm -o gpu_probe/workloads gpu_probe/workloads.c -ldl
$NDK/aarch64-linux-android34-clang -O2 -lm -o gpu_probe/cpu_baseline gpu_probe/cpu_baseline.c
```

**Deploying to device:**
```bash
adb push gpu_probe/clinfo /data/local/tmp/
adb push gpu_probe/gpu_bench /data/local/tmp/
adb push gpu_probe/workloads /data/local/tmp/
adb shell "chmod 755 /data/local/tmp/clinfo /data/local/tmp/gpu_bench /data/local/tmp/workloads"
adb shell "/data/local/tmp/clinfo"
adb shell "/data/local/tmp/gpu_bench"
adb shell "/data/local/tmp/workloads"
```

**TFLite inference benchmarks:**
```bash
adb push gpu_probe/benchmark_model /data/local/tmp/
adb push gpu_probe/mobilenet_v2_quant.tflite /data/local/tmp/
adb push gpu_probe/mobilenet_v2_fp32.tflite /data/local/tmp/
adb shell "chmod 755 /data/local/tmp/benchmark_model"

# CPU baseline (4 threads, INT8)
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v2_quant.tflite --num_threads=4 --num_runs=20"

# NNAPI GPU (best performance for INT8)
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v2_quant.tflite --use_nnapi=true --num_runs=20"

# TFLite GPU delegate (best for FP32)
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v2_fp32.tflite --use_gpu=true --num_runs=20"
```

## Operational Notes

- USB debugging auth gets revoked on every reboot — must re-authorize on phone
- ADB connection drops intermittently — replug USB and tap "Allow"
- SELinux can be set to Permissive: `adb shell "su -c 'setenforce 0'"`
- GPU DVFS ramps from 390 MHz to 950 MHz under sustained load
- OpenCL shader compilation adds ~1 second to first kernel launch (cached thereafter)
- Phone PIN: 1196

## Symbol CRC Table

For kernel module development, the verified CRC values for `__versions`:

```python
SYMBOL_CRCS = {
    "module_layout": 0x7C24B32D,
    "__ioremap": 0x6B4B2933,
    "iounmap": 0xEDC03953,
    "printk": 0xC5850110,
    "proc_create": 0x20FD21C6,
    "remove_proc_entry": 0x3C651057,
    "arm64_use_ng_mappings": 0xAF56600A,
    "__log_post_read_mmio": 0x6980EA4B,
    "__log_read_mmio": 0xCF1211A8,
    "seq_lseek": 0xAD9F2705,
    "seq_read": 0xB9997D36,
    "seq_printf": 0xD740362B,
    "single_open": 0x4F731E2B,
    "single_release": 0xEF42EDDB,
    "__tracepoint_rwmmio_post_read": 0x19EBF04E,
    "__tracepoint_rwmmio_read": 0xA035D76E,
    "__stack_chk_fail": 0x98A9D10C,
    "__stack_chk_guard": 0x8F678B07,
    "kfree": 0x037A0CBA,
    "kmalloc_caches": 0x8900B200,
    "kmem_cache_alloc_trace": 0xB38391E9,
    "vsnprintf": 0x00148653,
    "vmalloc": 0xD6EE688F,
    "vfree": 0x999E8297,
    "__tracepoint_rwmmio_write": 0x95575C33,
    "__log_write_mmio": 0x31DFD5CD,
    "__const_udelay": 0xEAE3DFD6,
    "__arm_smccc_smc": 0xF93AAE46,
}
```
