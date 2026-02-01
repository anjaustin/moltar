# PRD: Neural Interposer + LFM2 (State Channels + ShortConv Chip)

## Goal

Ship an on-device prototype that runs **LFM2’s stateful components** through **explicit state channels**, starting with a correct and validated **ShortConv chip**, and scaling toward end-to-end LFM2 execution.

This PRD is written for the currently validated device class:
- **SoC**: MediaTek Dimensity 930 (MT6855V)
- **GPU**: PowerVR BXM-8-256

## Why (problem statement)

Traditional compilation/partitioning pipelines (e.g., Vulkan delegates) often reject or destabilize models with:
- **hidden mutable state** (conv state, KV-cache)
- large and unpredictable GPU allocations
- runtime instability that manifests as abrupt disconnects/reboots

The Neural Interposer approach re-frames state as a **channel value** and computation as a **frozen transfer function** between channel states.

## Non-goals (for the first milestone)

- Full “runs forever” persistent kernel on PowerVR (known to be unreliable with host-driven spin-wait signals)
- Full LFM2 end-to-end generation (token sampling, tokenizer integration, UI)
- GPU-resident full weights for LFM2 (memory constraints)

## Execution strategy (staged)

### Stage 0 (already done)
- Working Vulkan channel demo (`research/brack/neural_interposer_demo/`) proving:
  - coherent buffers
  - multi-wave state evolution (multi-submit)

### Stage 1 (this PRD’s first deliverable): ShortConv chip

Implement a **ShortConv chip** that matches the core of `ShortConvExplicit` in `research/brack/lfm2_explicit_state/lfm2_explicit_state_model.py` for **S=1** (“decode step”):
- consumes channel inputs (Bx, C, conv_state_in, weights)
- produces outputs (y, conv_state_out)
- validated vs CPU reference on-device

### Stage 2: ShortConvBlock chip

Expand ShortConv chip into the full ShortConvBlock behavior:
- normalization + B_proj/C_proj/x_proj/out_proj (can start on CPU, then incrementally move into GPU chips)
- establish memory layouts for intermediate channels

### Stage 3: LFM2 state channels ABI

Define and freeze a stable ABI for:
- conv state channel(s)
- KV-cache channel(s) (k/v)
- token/position input channel(s)

### Stage 4: LFM2 integration

Incrementally compose chips:
- ShortConv chips for conv layers
- attention chips (initially CPU or partial)
- validate correctness step-by-step

## Channel ABI (v0)

### Conventions
- **float32** baseline (FP16 optional later)
- **explicit state**: no in-place mutation of input state buffers; state updates are outputs
- **ping-pong** is allowed at the host layer (swap state_in/state_out buffers per step)

### ShortConv chip ABI (S=1, depthwise, bias=false)

Parameters (constants for a compiled chip instance):
- \(D\): hidden dimension (channels)
- \(L\): kernel size / cache length (LFM2 uses \(L=3\))

Buffers:
- **Bx**: `float[D]` (precomputed `B_proj(x) * x_proj(x)` for the current step)
- **C**: `float[D]` (precomputed `C_proj(x)` for the current step)
- **conv_state_in**: `float[D * (L-1)]` laid out as:
  - channel-major: `conv_state_in[i*(L-1) + j]`
- **weights**: `float[D * L]` laid out as:
  - channel-major: `weights[i*L + k]`
  - corresponds to depthwise conv weights for the per-channel kernel
- **y_out**: `float[D]`
- **conv_state_out**: `float[D * (L-1)]` where:
  - `conv_state_out[i*(L-1) + 0] = conv_state_in[i*(L-1) + 1]`
  - `conv_state_out[i*(L-1) + 1] = Bx[i]`

Computation per channel \(i\):
\[
conv\_out_i = \sum_{k=0}^{L-1} w_{i,k} \cdot x\_with\_state_{i,k}
\]
where \(x\_with\_state_{i,:} = [conv\_state\_in_{i,0}, \ldots, conv\_state\_in_{i,L-2}, Bx_i]\).

Then:
\[
y_i = C_i \cdot conv\_out_i
\]

## Acceptance criteria

### Correctness
- ShortConv GPU chip matches CPU reference within tolerance:
  - `max_abs_err < 1e-6` for float32 on device for \(D \le 4096\)

### Stability
- Multi-wave execution for 16+ steps does not crash or reboot the device.

### Reproducibility
- One command builds and one command runs:
  - build: `bash research/brack/neural_interposer_demo/scripts/build_android.sh`
  - run: `adb shell /data/local/tmp/interposer_demo --chip shortconv --waves 16 ...`

## Risks / mitigations

- **PowerVR persistent dispatch signal visibility**: avoid host spin-wait persistent kernels; use multi-submit scheduling.
- **Memory pressure**: keep buffers bounded; prefer streaming and ping-pong state.
- **Numerical drift**: keep float32 for v0; introduce fp16 later with explicit acceptance tolerances.

## References

- Neural Interposer architecture: `docs/NEURAL_INTERPOSER.md`
- PowerVR Vulkan notes: `docs/VULKAN_POWERVR_NOTES.md`
- Current runnable demo: `research/brack/neural_interposer_demo/README.md`
- LFM2 explicit-state reference model: `research/brack/lfm2_explicit_state/lfm2_explicit_state_model.py`

