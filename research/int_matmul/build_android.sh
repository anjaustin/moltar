#!/bin/bash
set -e

NDK="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/28.2.13676358}"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/darwin-x86_64"
CC="$TOOLCHAIN/bin/aarch64-linux-android24-clang"

echo "Building integer matmul benchmark for Android ARM64..."

$CC -O3 -march=armv8.2-a+dotprod -mtune=cortex-a78 \
    -ffast-math -fno-math-errno \
    -o int_matmul_arm int_matmul_arm.c \
    -static

echo "Built: int_matmul_arm"
ls -la int_matmul_arm

echo ""
echo "To run on device:"
echo "  adb push int_matmul_arm /data/local/tmp/"
echo "  adb shell 'chmod +x /data/local/tmp/int_matmul_arm && /data/local/tmp/int_matmul_arm'"
