# Neural Interposer Demo (Phase 0)

This is a **minimal, on-device proof** of the `docs/NEURAL_INTERPOSER.md` approach:

- **Channels** are modeled as **shared, coherent buffers** visible to both CPU and GPU.
- A tiny “frozen chip” runs on Vulkan compute:
  - \(y = x + state\)
  - `next_state = y`
- A **signal word** coordinates the wave.

This is intentionally small and stable before we attempt “true persistent kernels” and LFM2.

## Build (host)

```bash
export ANDROID_NDK="/path/to/android-ndk"
bash research/brack/neural_interposer_demo/scripts/build_android.sh
```

Outputs:
- `research/brack/neural_interposer_demo/build-android/interposer_demo`
- `research/brack/neural_interposer_demo/build-android/interposer_demo.spv`

## Run (device via adb)

```bash
adb push research/brack/neural_interposer_demo/build-android/interposer_demo /data/local/tmp/
adb push research/brack/neural_interposer_demo/build-android/interposer_demo.spv /data/local/tmp/
adb push research/brack/neural_interposer_demo/build-android/interposer_demo_persistent.spv /data/local/tmp/
adb push research/brack/neural_interposer_demo/build-android/shortconv_chip.spv /data/local/tmp/
adb shell chmod +x /data/local/tmp/interposer_demo
adb shell "/data/local/tmp/interposer_demo --spv /data/local/tmp/interposer_demo.spv --n 1024"
```

Expected logs:
- GPU name
- `signal[0]=2`
- `max_abs_err=0.000000`

## Run (bounded persistent-style multi-wave)

```bash
adb shell "/data/local/tmp/interposer_demo --mode persistent --spv /data/local/tmp/interposer_demo_persistent.spv --n 1024 --waves 16"
```

Status:
- **Experimental on PowerVR**: on our current device class, the GPU may not reliably observe CPU-driven signal transitions during a long-running dispatch.
- Use `--mode multi_submit` as the baseline for multi-wave scheduling.

## Run (recommended multi-wave baseline)

```bash
adb shell "/data/local/tmp/interposer_demo --mode multi_submit --spv /data/local/tmp/interposer_demo.spv --n 1024 --waves 16"
```

Expected logs:
- `Wave 0 done. expected=1.000000 ... max_abs_err(first64)=0.000000`
- `Wave 15 done. expected=16.000000 ... max_abs_err(first64)=0.000000`

## Run (ShortConv chip prototype)

This runs a stateful ShortConv chip (S=1) with explicit state-in/state-out, matching the
core of LFM2’s `ShortConvExplicit` depthwise conv path.

```bash
adb shell "/data/local/tmp/interposer_demo --chip shortconv --spv /data/local/tmp/shortconv_chip.spv --d 1024 --waves 16"
```

Expected logs:
- CPU vs GPU max error is small (float32)
- state ping-pongs cleanly across waves

## Run (LFM2-350M ShortConv “real weights” test)

This path validates the interposer **against PyTorch-derived expected tensors** for one LFM2 conv layer (currently layer 0).

### 1) Export bins on host

```bash
python3 research/brack/lfm2_explicit_state/export_shortconv_layer_bins.py \
  --config_json research/spaceghost/executorch/examples/models/lfm2/config/lfm2_350m_config.json \
  --checkpoint_pt research/brack/models/LFM2-350M/model.pt \
  --output_dir research/brack/neural_interposer_demo/build-weights/lfm2_sc \
  --waves 8
```

### 2) Push bins to device

```bash
adb shell mkdir -p /data/local/tmp/lfm2_sc
adb push research/brack/neural_interposer_demo/build-weights/lfm2_sc/. /data/local/tmp/lfm2_sc/
```

### 3) Run on device

```bash
adb shell "/data/local/tmp/interposer_demo --chip lfm2_shortconv \
  --spv /data/local/tmp/shortconv_pre.spv \
  --spv2 /data/local/tmp/matvec_out.spv \
  --d 1024 --waves 8 --weights_dir /data/local/tmp/lfm2_sc"
```

Expected logs:
- `LFM2 ShortConv wave N done. max_err(...)` small (~1e-5 or better)

## Run (LFM2-350M conv suite: all conv layers)

This exports + validates **all 10 conv layers** from `lfm2_350m_config.json`:
`conv_layer_ids=[0,1,3,4,6,7,9,11,13,15]`.

```bash
export ANDROID_NDK="/path/to/android-ndk"
bash research/brack/neural_interposer_demo/scripts/run_lfm2_conv_suite.sh
```

