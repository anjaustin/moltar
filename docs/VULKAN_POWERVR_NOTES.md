# Vulkan on PowerVR (Moto g power 5G 2023)

This guide captures **known-good** build/run steps and the **real failure modes** we’ve observed on the current validated device class:

- **SoC**: MediaTek Dimensity 930 (MT6855V)
- **GPU**: PowerVR BXM-8-256

## Host prerequisites (macOS)

### Required tools

```bash
brew install shaderc
which glslc
glslc --version
```

### Android NDK

```bash
export ANDROID_NDK="$HOME/Library/Android/sdk/ndk/$(ls "$HOME/Library/Android/sdk/ndk" | head -1)"
```

## Known-good runnable demo (Neural Interposer)

See `../research/brack/neural_interposer_demo/README.md`.

### Why multi-submit is the baseline

On PowerVR, a long-running compute dispatch that spin-waits on **host-updated** memory can fail to observe signal transitions reliably (the host times out waiting for the GPU to flip a phase word).

The stable pattern is:
- **Submit a dispatch per “wave”** (multi-submit)
- Keep **state buffers** persistent across waves (state-as-channel-voltage still holds)

## Known failure modes and what to do

### `glslc from the Vulkan SDK must be installed`

- Install Shaderc (`brew install shaderc`)
- Ensure `glslc` is on `PATH`

### Sudden `adb: error: closed` / device reboot during Vulkan execution

Most often: **GPU OOM / driver reset**.

Actions:
```bash
adb logcat -c

# Re-run the failing command, then immediately:
adb logcat -d | tail -n 200
```

### Truncated files after reboot

If the device reboots mid-push, the on-device model can be silently smaller than expected.

```bash
adb shell ls -lh /data/local/tmp/*.pte
# If suspicious, re-push the file completely.
```

### `vmaAllocateMemory(...) returned -2`

This indicates out-of-device/host memory from the Vulkan allocator.

Mitigations:
- Reduce what’s delegated to Vulkan (blocklist heavy ops during export)
- Reduce model footprint and/or context
- Prefer multi-submit scheduling over persistent spin-wait loops

