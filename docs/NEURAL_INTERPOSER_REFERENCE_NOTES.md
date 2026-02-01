# Neural Interposer – Reference Notes (Drafts + Ideas)

This file is intentionally a **parking lot** for architectural writeups and “future direction” sketches.

It’s useful as a **design north star**, but parts of it are **not yet validated** on our current hardware/runtime stack.

## Current validated reality (Feb 2026)

- **Device class**: moto g power 5G (2023)
- **SoC**: MediaTek Dimensity 930 (MT6855V)
- **GPU**: PowerVR BXM-8-256
- **Working baseline scheduler**: multi-submit (“wave by wave”), not an infinite persistent spin-wait kernel.

Related:
- `PRD_NEURAL_INTERPOSER_LFM2.md`
- `VULKAN_POWERVR_NOTES.md`
- runnable demo: `../research/brack/neural_interposer_demo/README.md`

---

## Reference draft: “One-way streaming” writeup (verbatim)

```text
# Neural Interposer

**A TriX-Based Soft-Chip Architecture for Heterogeneous Computing**

The Neural Interposer is a revolutionary architectural paradigm that treats heterogeneous hardware (CPU, GPU, cache) as **channels** rather than discrete computational units. Built on TriX's frozen computation primitives, it enables deterministic neural networks to orchestrate hardware resources as a unified dataflow substrate.

---

## Key Innovation

By treating computation as **frozen logic gates** and hardware as **signal channels**, the Neural Interposer transforms mobile devices into dedicated neural processors where the distinction between "running a program" and "configuring a circuit" disappears.

---

## Architecture: One-Way Streaming

┌─────────────────────────────────────────┐
│         NEURAL INTERPOSER               │
│    (One-Way Streaming Architecture)     │
│                                         │
│  GPU (Mali)          CPU (Cortex)      │
│     │                    ▲              │
│     │   state_t+1        │              │
│     └────────────────────┘              │
│        ONE-WAY STREAM                   │
│        (ION coherent)                   │
│        10-50 ns latency                 │
│                                         │
│  No polling. No waiting. Just flow.    │
└─────────────────────────────────────────┘

The Neural Interposer implements a **one-way streaming architecture** that eliminates polling and synchronization overhead. The GPU streams state updates directly to the CPU through ION coherent memory, achieving nanosecond-level latency.

---

## Core Components

### 1. Channel Abstraction Layer (CAL)

Provides hardware-agnostic interface for treating CPU, GPU, and cache as unified dataflow channels with zero-copy semantics.

**Features:**
- Coherent memory allocation (ION/DMA-BUF)
- Zero-copy read/write operations
- Lightweight signaling (futex-based)
- Circular buffer management
- Version tracking for coherency

### 2. TriX Execution Model

Frozen computation primitives that transform channel states.

**The 5 Primes:**
- `ADD` — Element-wise addition
- `MUL` — Element-wise multiplication
- `EXP` — Exponential function
- `MAX` — Maximum function
- `CONST` — Constant initialization

**Derived Operations:**
- ReLU, Sigmoid, Tanh, Softmax
- Linear layers, Matrix multiplication

### 3. LFM2 Chips

Frozen TriX chips for LFM2 350M model components:
- **ShortConv chip** — 1D convolution with persistent state
- **Attention chip** — Multi-head attention with KV-cache
- **FFN chip** — Feed-forward network (stateless)
- **LFM2 Layer chip** — Composition of above chips

### 4. Scheduler

Orchestrates chip execution across CPU and GPU channels.

**Features:**
- Execution graph (DAG) management
- CPU/GPU thread coordination
- State management in cache channel
- Persistent kernel support

---

## Porting to Motorola/Mali

### Required Changes

1. **ION Memory Allocator** (Android/MediaTek specific)
2. **Vulkan Integration** (import ION into Vulkan)
3. **Mali GPU Optimization**

---
```

## Notes / caveats about the draft (why we keep it as “reference”)

- **GPU (Mali)**: our validated GPU is **PowerVR**, so persistence/sync details differ.
- **ION coherent**: raw ION fd pathways are not a stable Android API; AHardwareBuffer-based import is more portable.
- **“No polling / no waiting”**: depends on the sync primitive; host-driven spin-waiting in a long-running dispatch is not reliable on our PowerVR driver.
- **Nanosecond latency claims**: not treated as factual until benchmarked on the target device with a published harness.

