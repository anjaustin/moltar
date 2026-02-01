# ExecuTorch + Soft‑Chip Integration (Neural Interposer Track)

This document describes how we integrate **ExecuTorch** (graph/runtime) with our **soft‑chip** execution path (Neural Interposer / Vulkan chips) so we can accelerate LFM2 hot-path kernels without abandoning the ExecuTorch ecosystem.

## What “ExecuTorch + soft‑chip” means

- **ExecuTorch stays the runtime**: `.pte` loading, scheduling, memory planning, operator dispatch.
- **Soft‑chip accelerates selected ops**: we replace specific subgraphs (starting with ShortConv) with a custom kernel implementation that calls our Vulkan chip runner.

## Why this is the correct integration surface

Our current device class (Dimensity 930 / PowerVR) makes “full delegate takeover” risky; the stable path is:

- **Keep most ops on the known-good path** (portable/optimized kernels, XNNPack where valid).
- **Offload only the validated bottleneck** (LFM2 ShortConv stack) into our chip code.

## Integration mechanism (v0): ExecuTorch custom op + out-variant kernel

We implement a **custom operator** with an **out variant** so ExecuTorch can plan memory and dispatch a native kernel at runtime.

Key constraints to design around:

- ExecuTorch custom ops are **out-variant oriented**.
- Custom ops currently support returning **a single `Tensor`** (or `()`), so explicit state must be handled via:
  - in-place mutation of a `Tensor(a!) state` argument, or
  - packing output+state into one tensor, or
  - splitting into multiple ops.

## What we’re accelerating first

- **LFM2 ShortConv**: the explicit-state depthwise conv step, validated end-to-end in the on-device harness under `research/brack/neural_interposer_demo/`.
- The chip runner uses **multi-submit scheduling** (PowerVR stability baseline).

## Where the pieces live (current repo)

- **Soft‑chip reference runner**: `research/brack/neural_interposer_demo/`
- **Export/golden generation**: `research/brack/lfm2_explicit_state/`
- **ExecuTorch tree (submodule)**: `research/spaceghost/executorch/`
  - Custom op examples: `examples/portable/custom_ops/`
  - LLM custom op patterns: `extension/llm/custom_ops/`
  - Kernel registration docs: `docs/source/kernel-library-custom-aten-kernel.md`

## How the Lincoln Manifold Method fits this work

This integration is exactly the kind of task where we should separate thinking from building.

- **Method**: `../LMM.md` (The Lincoln Manifold Method)
- **Recommended usage here**:
  - RAW: enumerate integration options (custom op vs backend) + constraints (single-output, explicit state, PowerVR stability)
  - NODES: extract the real decision points (ABI, state ownership, memory layout, dispatch scheduling, fallback)
  - REFLECT: resolve tensions (correctness vs perf, portability vs specialization)
  - SYNTHESIZE: write the operator ABI + build wiring + acceptance tests

## Acceptance criteria (v0)

- **Correctness**: chip-backed outputs and updated state match PyTorch golden outputs for one ShortConv layer over N waves.
- **Stability**: no device reboots / hangs over repeated runs (minutes).
- **Integration**: model executes under ExecuTorch with the custom op enabled; disabling the custom op falls back cleanly.

