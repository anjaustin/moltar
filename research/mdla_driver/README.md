# MDLA Driver Development Plan

## Overview

**Goal**: Unlock the MediaTek MDLA 2.x Deep Learning Accelerator on MT6855/Dimensity 930 for LLM inference acceleration.

---

## Phase 0: Hardware Verification (COMPLETED)

### Findings - February 10, 2026

**Hardware CONFIRMED via sysfs**:
- APU IOMMU devices: `apu-iommu0-bank{1-4}`, `apu-iommu1-bank{1-4}`
- APU region base: `apu-region-base`  
- APU voltage regulator: `mt6363_vsram_apu` (open_count=0)

**BUT - Cannot access without kernel module**:
```
adb shell $ ls /dev/mem
ls: /dev/mem: No such file or directory

adb shell $ cat /proc/iomem | grep 1903
(nothing - addresses not exposed)

adb shell $ ls /dev/apu* /dev/mdla*
ls: /dev/apu*: No such file or directory
```

**Root Cause**: Android blocks direct `/dev/mem` access (CONFIG_STRICT_DEVMEM) and Motorola removed APU device nodes.

### What Was Tested

| Test | Result | Notes |
|------|--------|-------|
| `/dev/mem` access | FAIL | ENODEV - Android security |
| Create `/dev/mem` node | SUCCESS | But can't open |
| `/sys/bus/platform/drivers/apu-*` | EXISTS | IOMMU drivers present |
| APU regulator | EXISTS | mt6363_vsram_apu |
| APUSYS kernel modules | ON DISK | Not loaded |

### Device Info

```
Kernel: 5.10.205-android12-9-00035-g264434cb4c98-ab12029476
Device: Motorola Moto G Power 5G 2023 (ZY22HWSKXX)
Chip:   MediaTek MT6855/Dimensity 930
SELinux: Enforcing
```

---

## Phase 1: Build Kernel Module

### 2.1 Understand MDLA Command Interface

The MDLA uses a command queue interface via CDMA registers:

```
CDMA0: Input activation address
CDMA1: Weight address  
CDMA2: Output address
CDMA3: Command parameters (MUL/AWK/ADD operations)
CDMA4-6: Extended parameters
```

### 2.2 Implement Command Queue

```c
// pseudocode for command submission
struct mdla_cmd {
    uint32_t op;          // Operation type (GEMM, CONV, etc)
    uint32_t src_addr;    // Source tensor physical address
    uint32_t dst_addr;     // Destination tensor physical address
    uint32_t src_size;    // Source size in bytes
    uint32_t dst_size;    // Destination size in bytes
    uint32_t params[8];   // Operation-specific parameters
};

int mdla_submit_cmd(mdla_t *dev, struct mdla_cmd *cmd) {
    // Write command to CDMA registers
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA0, cmd->src_addr);
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA1, cmd->dst_addr);
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA2, cmd->src_size);
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA3, cmd->op);
    
    // Trigger execution
    mdla_write_cmd(dev, REG_MREG_TOP_G_INTP0, 0);
    
    // Wait for completion
    while (!(mdla_read_intr(dev) & INTR_SWCMD_DONE))
        ;
    
    return 0;
}
```

### 2.3 Create Command Buffer Allocation

```c
// Allocate DMA-coherent memory for tensors
void *mdla_alloc(mdla_t *dev, size_t size) {
    // Use ion_alloc for DMA-capable memory
    // Or mmap with PROT_READ|PROT_WRITE|PROT_EXEC
}
```

---

## Phase 3: First GEMM Operation

### 3.1 Target Operation

**8-bit Integer Matrix Multiply** (matches Q4_0/Q8_0 format):

```
Input:  [M, K] int8 activation
Weight: [K, N] int8 weights  
Output: [M, N] int32 accumulation
```

### 3.2 Register Programming Sequence

```c
int mdla_gemm(mdla_t *dev,
              void *input, size_t M, size_t K,
              void *weights, size_t N,
              void *output) {
    
    // Configure source addresses
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA0, phys(input));
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA1, phys(weights));
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA2, phys(output));
    
    // Set dimensions
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA3, M);  // M
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA4, K);  // K  
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA5, N);  // N
    
    // Set operation: GEMM_INT8
    mdla_write_cmd(dev, REG_MREG_TOP_G_CDMA6, 0x80000001);
    
    // Trigger and wait
    mdla_write_cmd(dev, REG_MREG_TOP_G_DBG_CMDE01, 
                   LCE_WAIT_LAST_CFG | LCE_LAYER_END);
    
    while (!mdla_is_idle(dev))
        ;
    
    return 0;
}
```

### 3.3 Verification

```bash
# Run known matrix multiply
./test_gemm

# Compare MDLA output with NEON reference
# Expected: Exact match for int8 GEMM
```

---

## Phase 4: llama.cpp Integration

### 4.1 Backend Architecture

```
llama.cpp GGML ops
     ↓
ggml-backend API
     ↓
mdla-backend.c  ← New backend
     ↓
   MDLA Driver
     ↓
 MDLA Hardware
```

### 4.2 Implement Backend

```c
// ggml-mdla-backend.c

struct ggml_backend_mdla {
    ggml_backend_t base;
    mdla_t dev;
    int sync_mode;  // 0=async, 1=sync
};

ggml_backend_t ggml_backend_mdla_init(void) {
    // Open device, allocate command queue
}

ggml_status_t ggml_backend_mdla_alloc_buffer(ggml_backend_t backend,
                                            size_t size) {
    // Allocate DMA-capable buffer
}

void ggml_backend_mdla_free_buffer(ggml_backend_buffer_t buffer) {
    // Free allocated buffer
}

ggml_status_t ggml_backend_mdla_compute(ggml_backend_t backend,
                                        ggml_compute_params *params,
                                        ggml_tensor *tensor) {
    // Convert GGML operation to MDLA commands
    // For GEMV: invoke mdla_gemm()
}
```

### 4.3 Dispatch Integration

Modify `ggml/src/ggml-cpu/ggml-ops.h`:
```c
#if defined(GGML_USE_MDLA)
    if (ggml_backend_is_mdla(backend)) {
        return ggml_backend_mdla_mul_mat(ctx, a, b);
    }
#endif
```

---

## Phase 5: Optimization and Performance

### 5.1 Quantization Support

| Format | Support | Notes |
|--------|---------|-------|
| Q4_0 | Planned | 4-bit weights, int8 accum |
| Q8_0 | Planned | 8-bit weights, int8 accum |
| F16 | Easy | Native format |
| F32 | Easy | Reference |

### 5.2 Batching

- Process multiple inputs concurrently
- Pipeline command submission
- Overlap computation with data transfer

### 5.3 Performance Targets

| Metric | Target | Current (NEON) |
|--------|--------|----------------|
| GEMM throughput | 10-100 TOPS | ~0.1 TOPS |
| Power efficiency | >5 TOPS/W | ~0.5 TOPS/W |
| Latency (LFM2-350M) | <100ms/token | 13ms/token |

---

## File Roadmap

```
moltar/research/mdla_driver/
├── README.md              # This plan
├── mdla.h                # Driver API
├── mdla.c                # Implementation
├── test_mdla.c           # Hardware verification
├── test_gemm.c           # GEMM correctness test
├── bench_gemm.c          # Performance benchmark
├── ggml-mdla-backend.c   # llama.cpp backend
└── Makefile              # Build system
```

---

## Success Criteria

### Minimum Viable Product
- [ ] Hardware responds to register access
- [ ] SPM power bit = 1
- [ ] Successfully execute int8 GEMM
- [ ] Results match NEON reference
- [ ] Integrated with llama.cpp

### Production Ready
- [ ] Full Q4_0/Q8_0 support
- [ ] Command queue with <1ms latency
- [ ] No memory leaks or crashes
- [ ] Power consumption <5W
- [ ] Benchmarks show improvement over NEON

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| APU fused off in hardware | Medium | High | Explore RPC wake sequences |
| Missing documentation | High | Medium | Reverse-engineer from MTK driver |
| IOMMU blocking access | Low | High | Configure IOMMU passthrough |
| Thermal throttling | Medium | Medium | Limit frequency, add cooling |

---

## References

- MT6855 Kernel: `kernel-mtk/drivers/misc/mediatek/apusys/mdla/`
- MDLA 2.0 Register Map: `platform/v2_0/mdla_hw_*.h`
- GGML Backend API: `llama.cpp/ggml/docs/backends/`
