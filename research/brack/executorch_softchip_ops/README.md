# ExecuTorch + Soft‑Chip (Code) — Minimal End‑to‑End

This directory contains the **first real code integration** of our soft‑chip path into the ExecuTorch runtime via a **custom op**.

## What this adds

- **Custom op**: `ni::shortconv3_step.out`
- **Runtime kernel**: Vulkan-backed implementation that runs the same shader as the Neural Interposer demo (`shortconv_chip`).
- **Build wiring**: `research/brack/executorch_android_runner/CMakeLists.txt` links the op into `executorch_runner`, so the op is registered at startup.

## Runtime requirement: SPIR‑V on device

The custom op loads SPIR‑V from disk at runtime.

- Default path: `/data/local/tmp/shortconv_chip.spv`
- Override: set env var `NI_SHORTCONV3_SPV` to a different `.spv` path.

To push the shader to device:

```bash
adb push research/brack/neural_interposer_demo/shaders/shortconv_chip.spv /data/local/tmp/shortconv_chip.spv
```

## Export a smoke model (.pte)

On host:

```bash
python research/brack/lfm2_explicit_state/ni_shortconv3_smoke_export.py --out smoke_shortconv3.pte --D 1024
```

Then push to device:

```bash
adb push smoke_shortconv3.pte /data/local/tmp/smoke_shortconv3.pte
```

## Run on device

Once you’ve built and pushed `executorch_runner` (from `research/brack/executorch_android_runner/`), run:

```bash
adb shell "cd /data/local/tmp && ./executorch_runner --model_path smoke_shortconv3.pte"
```

If Vulkan init succeeds, you should see a log line like:

```text
ni_shortconv3: Vulkan init OK (spv=/data/local/tmp/shortconv_chip.spv)
```

