#!/bin/bash
# Build Yinsen NEON kernels for Android ARM64
#
# Usage: ./build_android.sh

set -e

# Find NDK
if [ -z "$ANDROID_NDK" ]; then
    if [ -d "$HOME/Library/Android/sdk/ndk/28.2.13676358" ]; then
        ANDROID_NDK="$HOME/Library/Android/sdk/ndk/28.2.13676358"
    elif [ -d "$HOME/Library/Android/sdk/ndk/28.0.12916984" ]; then
        ANDROID_NDK="$HOME/Library/Android/sdk/ndk/28.0.12916984"
    elif [ -d "$HOME/Library/Android/sdk/ndk/27.0.12077973" ]; then
        ANDROID_NDK="$HOME/Library/Android/sdk/ndk/27.0.12077973"
    else
        echo "ERROR: Cannot find Android NDK. Set ANDROID_NDK environment variable."
        exit 1
    fi
fi

echo "Using NDK: $ANDROID_NDK"

# Compiler
CC="$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang"

# Compile flags for Cortex-A78/A55 (Dimensity 7020)
CFLAGS="-O3 -march=armv8.2-a+dotprod -mtune=cortex-a78 -ffast-math"

echo "Compiling neon_int8_matvec.c..."
$CC $CFLAGS -c neon_int8_matvec.c -o neon_int8_matvec.o

echo "Compiling bench_neon.c..."
$CC $CFLAGS -c bench_neon.c -o bench_neon.o

echo "Linking bench_neon..."
$CC $CFLAGS neon_int8_matvec.o bench_neon.o -o bench_neon -static

echo "Compiling membw_test.c..."
$CC $CFLAGS -c membw_test.c -o membw_test.o

echo "Linking membw_test..."
$CC $CFLAGS membw_test.o -o membw_test -static

echo "Compiling membw_mt_test.c..."
$CC $CFLAGS -c membw_mt_test.c -o membw_mt_test.o

echo "Linking membw_mt_test..."
$CC $CFLAGS membw_mt_test.o -o membw_mt_test -static

echo "Compiling q4_shiftadd.c..."
$CC $CFLAGS -c q4_shiftadd.c -o q4_shiftadd.o

echo "Compiling bench_q4_shiftadd.c..."
$CC $CFLAGS -c bench_q4_shiftadd.c -o bench_q4_shiftadd.o

echo "Linking bench_q4_shiftadd..."
$CC $CFLAGS q4_shiftadd.o bench_q4_shiftadd.o -o bench_q4_shiftadd -static

echo ""
echo "Build complete: bench_neon, membw_test, membw_mt_test, bench_q4_shiftadd"
echo ""
echo "To run on device:"
echo "  adb push bench_neon /data/local/tmp/"
echo "  adb shell chmod +x /data/local/tmp/bench_neon"
echo "  adb shell /data/local/tmp/bench_neon -n 4096 -k 4096 -i 1000"
