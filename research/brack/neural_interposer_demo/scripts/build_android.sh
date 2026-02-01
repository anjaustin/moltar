#!/usr/bin/env bash
set -euo pipefail

# Builds the Neural Interposer demo as a standalone Android binary + SPIR-V shader.
#
# Requirements on host (macOS):
# - Android NDK installed (set ANDROID_NDK to its path)
# - shaderc installed (provides glslc): `brew install shaderc`
#
# Output:
# - build-android/interposer_demo
# - build-android/interposer_demo.spv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${ANDROID_NDK:-}" ]]; then
  echo "ERROR: ANDROID_NDK is not set (path to Android NDK root)." >&2
  exit 1
fi

ABI="${ABI:-arm64-v8a}"
ANDROID_API="${ANDROID_API:-29}"

BUILD_DIR="${ROOT_DIR}/build-android"
mkdir -p "${BUILD_DIR}"

echo "Compiling shader to SPIR-V..."
glslc -O "${ROOT_DIR}/shaders/interposer_demo.comp" -o "${BUILD_DIR}/interposer_demo.spv"
glslc -O "${ROOT_DIR}/shaders/interposer_demo_persistent.comp" -o "${BUILD_DIR}/interposer_demo_persistent.spv"
glslc -O "${ROOT_DIR}/shaders/shortconv_chip.comp" -o "${BUILD_DIR}/shortconv_chip.spv"
glslc -O "${ROOT_DIR}/shaders/shortconv_pre.comp" -o "${BUILD_DIR}/shortconv_pre.spv"
glslc -O "${ROOT_DIR}/shaders/matvec_out.comp" -o "${BUILD_DIR}/matvec_out.spv"

echo "Configuring CMake (ABI=${ABI} API=${ANDROID_API})..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DANDROID_ABI="${ABI}" \
  -DANDROID_PLATFORM="android-${ANDROID_API}" \
  -DANDROID_NDK="${ANDROID_NDK}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"

echo "Building..."
cmake --build "${BUILD_DIR}" -j

echo "Built:"
echo "  ${BUILD_DIR}/interposer_demo"
echo "  ${BUILD_DIR}/interposer_demo.spv"
echo "  ${BUILD_DIR}/interposer_demo_persistent.spv"
echo "  ${BUILD_DIR}/shortconv_chip.spv"
echo "  ${BUILD_DIR}/shortconv_pre.spv"
echo "  ${BUILD_DIR}/matvec_out.spv"

