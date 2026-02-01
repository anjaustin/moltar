#!/usr/bin/env bash
set -euo pipefail

# Runs the LFM2-350M ShortConv validation suite (all conv layers).
#
# Prereqs:
# - ANDROID_NDK set (for build)
# - adb device connected
#
# Flow:
# 1) Build demo + shaders
# 2) Export per-layer bins (PyTorch reference)
# 3) For each conv layer dir:
#    - push bins to /data/local/tmp/lfm2_sc
#    - run interposer_demo --chip lfm2_shortconv
#    - dump InterposerDemo logcat

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../../.." && pwd)"

SUITE_DIR="${ROOT_DIR}/build-weights/lfm2_conv_suite"
DEVICE_DIR="/data/local/tmp/lfm2_sc"

WAVES="${WAVES:-8}"
D="${D:-1024}"

echo "[1/3] Build Android demo + shaders"
bash "${ROOT_DIR}/scripts/build_android.sh"

echo "[2/3] Export LFM2 conv suite bins (host)"
python3 "${REPO_ROOT}/research/brack/lfm2_explicit_state/export_shortconv_layer_bins.py" \
  --config_json "${REPO_ROOT}/research/spaceghost/executorch/examples/models/lfm2/config/lfm2_350m_config.json" \
  --checkpoint_pt "${REPO_ROOT}/research/brack/models/LFM2-350M/model.pt" \
  --output_dir "${SUITE_DIR}" \
  --all_conv_layers \
  --waves "${WAVES}" \
  --seed 0

echo "[3/3] Push demo + run suite on device"
adb wait-for-device
adb push "${ROOT_DIR}/build-android/interposer_demo" /data/local/tmp/
adb push "${ROOT_DIR}/build-android/shortconv_pre.spv" /data/local/tmp/
adb push "${ROOT_DIR}/build-android/matvec_out.spv" /data/local/tmp/
adb shell chmod +x /data/local/tmp/interposer_demo

for layer_dir in "${SUITE_DIR}"/layer_*; do
  layer_name="$(basename "${layer_dir}")"
  echo ""
  echo "=== ${layer_name} ==="
  adb shell "rm -rf \"${DEVICE_DIR}\" && mkdir -p \"${DEVICE_DIR}\""
  adb push "${layer_dir}/." "${DEVICE_DIR}/"

  adb logcat -c
  adb shell "/data/local/tmp/interposer_demo --chip lfm2_shortconv --spv /data/local/tmp/shortconv_pre.spv --spv2 /data/local/tmp/matvec_out.spv --d ${D} --waves ${WAVES} --weights_dir ${DEVICE_DIR}"
  adb logcat -d -s InterposerDemo
done

