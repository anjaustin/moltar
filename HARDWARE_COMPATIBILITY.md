# Hardware Compatibility

## Validated Hardware

### Motorola Moto G Power 5G (2023) — Primary Target

| Spec | Value |
|------|-------|
| **Codename** | `devonn` |
| **SKU** | XT2311-4 |
| **SoC** | MediaTek MT6855V / Dimensity 930 |
| **CPU (big)** | 2x Cortex-A78 @ 2.2 GHz |
| **CPU (little)** | 6x Cortex-A55 @ 2.0 GHz |
| **ISA** | ARMv8.2-a |
| **NEON extensions** | DOTPROD (`asimddp`), FP16 (`fphp`, `asimdhp`) |
| **Missing extensions** | I8MM, SVE, SVE2 |
| **RAM** | 6 GB LPDDR4X @ 4266 MHz |
| **DRAM bandwidth** | 15.5 GB/s sustained (2 threads, Android stopped) |
| **L1D cache** | 64 KB per core |
| **L2 cache** | 256 KB per A78 core |
| **L3 cache (DSU)** | ~1 MB shared |
| **Cache line** | 64 bytes |
| **GPU** | PowerVR BXM-8-256 |
| **GPU compute** | OpenCL 3.0, 128-wide SIMD, 28 KB local mem, 950 MHz max |
| **Storage** | 256 GB |
| **Android** | 13 |
| **Root** | Magisk (boot image patched) |
| **ADB serial** | ZY22HWSKXX |

### CPU Features (`/proc/cpuinfo`)

```
Features: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp
          cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp
          sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs paca pacg
          dcpodp svei8mm svebf16 i8mm bf16 dgh rng
```

Key flags for our kernels:
- `asimddp` (DOTPROD) — used by SDOT GEMV kernels
- `fphp` / `asimdhp` — FP16 support
- `asimd` — NEON baseline

Note: despite `i8mm` appearing in cpuinfo, the A78 does NOT have hardware I8MM. The flag comes from the A55 cluster which reports combined features. Our kernels use only DOTPROD.

### CPU Topology

```
Core 0-5: Cortex-A55 (little)  — NOT used for inference
Core 6-7: Cortex-A78 (big)     — ALL inference runs here (taskset c0)
```

Governor: set to `performance` via `/sys/devices/system/cpu/cpu{6,7}/cpufreq/scaling_governor`.

### Memory Hierarchy

```
L1D:  64 KB per core, 4-way, 64B line     — VDB topo fits at N<=256 (8 KB)
L2:   256 KB per A78 core, 8-way           — VDB total fits at N<=512 (48 KB)
L3:   ~1 MB shared (DSU), 16-way           — VDB total fits at N<=4096 (384 KB)
DRAM: 6 GB LPDDR4X, dual-channel           — LLM weights, large VDB
```

### DRAM Bandwidth (Measured)

| Configuration | Bandwidth |
|--------------|-----------|
| 1 thread, Android running | ~8 GB/s |
| 2 threads, Android running | ~11 GB/s |
| 1 thread, Android stopped | ~10 GB/s |
| **2 threads, Android stopped** | **15.5 GB/s** |
| Theoretical max (LPDDR4X 4266 dual) | ~17 GB/s |

Android framework (SurfaceFlinger, SystemUI, etc.) consumes ~4 GB/s of DRAM bandwidth. Stopping it (`su -c 'stop'`) is essential for maximum inference throughput.

### GPU Details (PowerVR BXM-8-256)

Confirmed via OpenCL probe:

| Spec | Value |
|------|-------|
| Vendor | Imagination Technologies |
| Device | PowerVR BXM-8-256 |
| OpenCL version | 3.0 |
| Max compute units | 2 |
| Max work group size | 512 |
| SIMD width | 128 |
| Local memory | 28 KB |
| Global memory | ~5.5 GB (shared with CPU) |
| Max clock | 950 MHz |

GPU is useful for async VDB search during LLM inference, but dispatch overhead (~40 us) makes it slower than CPU for standalone queries.

## Device Setup

### Root Access

Device is rooted via Magisk (boot image patch method). Required for:
- Stopping Android framework (`su -c 'stop'`)
- Setting CPU governors (`su -c 'echo performance > ...'`)
- Magisk service scripts (auto-setup on boot)

### Boot Script

`/data/adb/service.d/moltar_perf.sh`:
```bash
#!/system/bin/sh
# Stop Android framework to free DRAM bandwidth
stop
# Lock big cores to max frequency
echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
```

### Models on Device

Located at `/data/local/tmp/`:

| File | Size |
|------|------|
| LFM2-350M-Q4_0-pure.gguf | 190 MiB |
| LFM2-350M-Q8_0.gguf | 359 MiB |
| LFM2-700M-Q4_0.gguf | 446 MiB |
| LFM2-1.2B-Q4_0.gguf | 696 MiB |
| LFM2.5-1.2B-Thinking-Q4_0.gguf | 696 MiB |

## Toolchain

| Tool | Path / Version |
|------|---------------|
| NDK | `/opt/android-ndk-r27c` |
| NDK clang | `aarch64-linux-android33-clang` |
| Cross-GCC | `aarch64-linux-gnu-gcc` (system) |
| ADB | system install |
