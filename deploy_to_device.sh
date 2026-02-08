#!/bin/bash
# deploy_to_device.sh - Deploy llama.cpp binaries and models to Motorola device
#
# Pushes llama-bench, llama-cli, shared libraries, and GGUF models
# to /data/local/tmp/ on the attached Android device.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/research/llama.cpp/build-android"
DATA_DIR="${SCRIPT_DIR}/data"
DEVICE_DIR="/data/local/tmp"

# Verify device
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected"
    echo ""
    echo "Checklist:"
    echo "  1. USB cable connected"
    echo "  2. USB debugging enabled (Settings > Developer options)"
    echo "  3. Authorize this computer on the device"
    exit 1
fi

DEVICE_MODEL=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r')
echo "=== Deploying to: ${DEVICE_MODEL} ==="
echo "Target: ${DEVICE_DIR}"
echo ""

# Push binaries
echo "--- Binaries ---"
for bin in llama-bench llama-cli; do
    src="${BUILD_DIR}/bin/${bin}"
    if [ -f "$src" ]; then
        echo "  Pushing ${bin}..."
        adb push "$src" "${DEVICE_DIR}/${bin}" 2>&1 | tail -1
        adb shell "chmod 755 ${DEVICE_DIR}/${bin}"
    else
        echo "  Warning: ${bin} not found at ${src}"
    fi
done

# Push shared libraries
echo ""
echo "--- Libraries ---"
for lib in libggml-base.so libggml-cpu.so libggml.so libllama.so; do
    src="${BUILD_DIR}/bin/${lib}"
    if [ -f "$src" ]; then
        echo "  Pushing ${lib}..."
        adb push "$src" "${DEVICE_DIR}/${lib}" 2>&1 | tail -1
    fi
done

# Push models
echo ""
echo "--- Models ---"
for gguf in "${DATA_DIR}"/LFM2-350M-*.gguf; do
    if [ -f "$gguf" ]; then
        name=$(basename "$gguf")
        size=$(du -h "$gguf" | cut -f1)
        echo "  Pushing ${name} (${size})..."
        adb push "$gguf" "${DEVICE_DIR}/${name}" 2>&1 | tail -1
    fi
done

# Verify on device
echo ""
echo "--- Verification ---"
adb shell "ls -lh ${DEVICE_DIR}/llama-bench ${DEVICE_DIR}/llama-cli ${DEVICE_DIR}/*.gguf ${DEVICE_DIR}/lib*.so 2>/dev/null"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Run benchmark:"
echo "  adb shell 'cd ${DEVICE_DIR} && LD_LIBRARY_PATH=. taskset c0 ./llama-bench -m LFM2-350M-Q4_0.gguf -t 2 -fa 1'"
echo ""
echo "Or use the wrapper:"
echo "  ./run_lfm_mt6855v.sh bench"
