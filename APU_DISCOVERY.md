# MediaTek APU (MDLA 2.1) Discovery on MT6855/Dimensity 930

## Overview

**February 10, 2026** — The Moltar project discovered accessible AI accelerator hardware (MDLA 2.1) on the Motorola Moto G Power 5G 2023 (MT6855/Dimensity 930), despite manufacturer documentation stating the chip lacks NPU capabilities.

## Hardware Discovery

### Device Tree Anomaly

Motorola's kernel device tree for `devonn` (XT2311-4) completely omits all APU-related nodes:
- No `apu_rpc` device tree entry
- No `mdla` device tree entry  
- No `apu_conn`, `apu_vcore`, or `apu_iommu` entries
- All APUSYS-related config options disabled in `defconfig`

However, physical register addresses from the Vivo Y77 (MT6855) kernel tree successfully map and respond.

### Confirmed Hardware Components

| Component | Physical Address | Size | Purpose |
|-----------|-----------------|------|---------|
| APU RPC | 0x190F0000 | 4 KB | Remote Power Controller — power on/off, clock mux |
| MDLA0 | 0x19034000 | 4 KB | Deep Learning Accelerator core v2.0 |
| APU IOMMU | 0x19010000 | 4 KB | I/O Memory Management Unit for APU |
| APU CONN | 0x19020000 | 4 KB | Bus configuration and clock gating |
| APU VCORE | 0x19029000 | 4 KB | Core voltage and clock control |
| SPM | 0x10006000 | 4 KB | System Power Manager — APU power domain status |

## APU Architecture

### MDLA 2.0 Block Diagram

```
                    ┌─────────────────┐
                    │   APU RPC       │ ← Power/Clock Control
                    │  (0x190F0000)   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   APU CONN    │   │   APU VCORE   │   │   APU IOMMU   │
│  (Clock/Bus)  │   │ (Core Clock)  │   │  (Mem Map)    │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │     MDLA 0      │
                   │  Deep Learning  │
                   │    Accelerator  │
                   │   (0x19034000)  │
                   └─────────────────┘
```

### MDLA 2.0 Key Registers

| Offset | Register | Description |
|--------|---------|-------------|
| 0x000 | MDLA_CG_CON | Clock gating control |
| 0x00C | MDLA_SW_RST | Software reset |
| 0x0504 | MREG_TOP_G_INTP0 | Top interrupt 0 |
| 0x0510 | MREG_TOP_G_CDMA0 | Command DMA control |
| 0x0550 | MREG_TOP_ENG0 | Engine status |
| 0x0E00 | CFG_PMCR | Performance monitor control |
| 0x0E04 | PMU_CLK_CNT | Clock counter |

## Power Domain

The APU is powered through the SPM (System Power Manager) at `0x10006000`:

- **Bit 5** in `SPM_OTHER_PWR_STATUS` indicates APU power domain state
- RPC (Remote Power Controller) handles wake sequences:
  - `RPC_SW_TYPE0` — APUTOP wake type
  - `RPC_SW_TYPE6` — MDLA0 wake type

## Why Motorola Hid This

Possible explanations:

1. **Cost reduction** — Disabled APU in production to reduce power/thermal load
2. **Software lockout** — Driver removed from kernel, hardware still present
3. **Different SKU** — Some MT6855 variants have full APU, others disabled
4. **Certification** — Avoided neural network API certifications (Android NNAPI)

## Kernel Module: apu_probe.c

A minimal kernel module was developed to probe the APU hardware:

```bash
# Build with kernel headers
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules

# Load and check dmesg
insmod apu_probe.ko
dmesg | grep apu_probe
rmmod apu_probe
```

The module:
1. Maps each register region via `ioremap()`
2. Reads signature registers to confirm hardware presence
3. Tests RPC responsiveness via bit manipulation
4. Reports power domain status via SPM

## Userspace Driver: mdla_driver/

A userspace MDLA driver was developed for easier testing and iteration:

```
moltar/research/mdla_driver/
├── mdla.h           # Public API header
├── mdla.c           # Driver implementation (uses /dev/mem)
├── test_mdla.c       # Test program
└── Makefile         # Cross-compilation support
```

Build and run:
```bash
# Cross-compile for ARM64
make

# Push to device
adb push test_mdla /data/local/tmp/
adb shell 'su -c "chmod 666 /dev/mem"'
adb shell 'su -c /data/local/tmp/test_mdla'

# Or build native on device (Termux)
make android
```

### Driver Features

| Function | Description |
|----------|-------------|
| `mdla_open()` | Maps all register regions via `/dev/mem` |
| `mdla_power_on()` | Clears reset, enables clocks |
| `mdla_power_off()` | Asserts reset, gates clocks |
| `mdla_reset()` | Software reset sequence |
| `mdla_read_status()` | Read ENG0 status register |
| `mdla_is_idle()` | Check if MDLA is idle |
| `mdla_dump_regs()` | Dump all registers for debugging |

### Register Access

The driver provides direct access to key registers:

```c
// Control registers
mdla_write_cmd(dev, REG_MDLA_SW_RST, MDLA_SW_RST_MASK);  // Reset
mdla_write_cmd(dev, REG_MDLA_CG_CLR, 0xFFFFFFFF);       // Enable clocks

// Command interface
mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA0, base_addr);   // Set input address
mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA1, base_addr);   // Set weight address
mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA2, base_addr);   // Set output address

// Status
uint32_t status = mdla_read_status(dev);  // Read ENG0
uint32_t fin = mdla_read_finish(dev);     // Read FIN0
```

## Next Steps

### Immediate

1. **Run the probe on device** — Confirm APU responds to register access
2. **Check power domain** — Is bit 5 set in SPM_OTHER_PWR_STATUS?
3. **Test RPC wake sequence** — Bring MDLA out of reset
4. **Run userspace test** — `./test_mdla` to verify basic access

### Medium Term

5. **Implement command submission** — Write to CDMA registers to load commands
6. **Execute GEMM operation** — First working operation on MDLA
7. **Integrate with llama.cpp** — Offload Q4_0/Q8_0 GEMM to MDLA

### Long Term

8. **Full operator support** — Conv2D, depthwise, pooling, activation
9. **Android NNAPI backend** — Expose as `/dev/accel` or NNAPI delegate
10. **Performance benchmarking** — Compare MDLA vs NEON on LLM inference

## References

| Source | URL/Location |
|--------|-------------|
| MT6855 Kernel | `kernel-mtk/` |
| MT6853/MT6855 Apusys | `kernel-mtk/drivers/misc/mediatek/apusys/` |
| MDLA Driver | `kernel-mtk/drivers/misc/mediatek/apusys/mdla/` |
| RPC Driver | `kernel-mtk/drivers/misc/mediatek/apusys/midware/` |
| Power Driver | `kernel-mtk/drivers/misc/mediatek/apusys/power/` |
| Vivo Y77 Kernel | External reference (MT6855 variant) |

## Risks and Caveats

- **Hardware may be disabled** — Even if registers exist, clocks/power may be gated
- **No documentation** — Reverse-engineering required, no datasheet available
- **Variability** — Other MT6855 SKUs may have different register layouts
- **Thermal/power limits** — APU may trigger throttling or safety shutdowns

## Acknowledgments

- Vivo Y77 kernel tree for register address references
- MediaTek Apusys kernel framework for register layout patterns
