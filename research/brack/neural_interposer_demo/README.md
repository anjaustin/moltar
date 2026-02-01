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

Expected logs:
- `Wave 0 done. expected=1.000000 ... max_abs_err(first64)=0.000000`
- `Wave 15 done. expected=16.000000 ... max_abs_err(first64)=0.000000`

