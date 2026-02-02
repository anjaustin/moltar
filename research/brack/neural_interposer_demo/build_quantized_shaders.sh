#!/bin/bash

# Build Quantized Shaders for Neural Interposer
# Compiles Vulkan compute shaders for quantized operations

set -e

echo "=== Building Quantized Neural Interposer Shaders ==="
echo

# Check for Vulkan SDK or glslc
if command -v glslc >/dev/null 2>&1; then
    GLSLC="glslc"
elif [ -n "$VULKAN_SDK" ] && [ -x "$VULKAN_SDK/bin/glslc" ]; then
    GLSLC="$VULKAN_SDK/bin/glslc"
else
    echo "Error: glslc not found. Please install Vulkan SDK or ensure glslc is in PATH"
    exit 1
fi

echo "Using GLSLC: $GLSLC"

# Create output directory
OUTPUT_DIR="build-quantized"
mkdir -p "$OUTPUT_DIR"

# Build quantized matrix multiplication shader
echo "Building quantized_matmul.comp..."
$GLSLC -fshader-stage=compute \
       --target-env=vulkan1.1 \
       -O \
       -o "$OUTPUT_DIR/quantized_matmul.spv" \
       shaders/quantized_matmul.comp

if [ $? -eq 0 ]; then
    echo "✅ quantized_matmul.spv built successfully"
    ls -lh "$OUTPUT_DIR/quantized_matmul.spv"
else
    echo "❌ Failed to build quantized_matmul.spv"
    exit 1
fi

# Build quantized attention shader
echo "Building quantized_attention.comp..."
$GLSLC -fshader-stage=compute \
       --target-env=vulkan1.1 \
       -O \
       -o "$OUTPUT_DIR/quantized_attention.spv" \
       shaders/quantized_attention.comp

if [ $? -eq 0 ]; then
    echo "✅ quantized_attention.spv built successfully"
    ls -lh "$OUTPUT_DIR/quantized_attention.spv"
else
    echo "❌ Failed to build quantized_attention.spv"
    exit 1
fi

echo
echo "=== Shader Build Complete ==="
echo "Generated files:"
ls -lh "$OUTPUT_DIR/"*.spv
echo
echo "Next steps:"
echo "1. Push shaders to device: adb push $OUTPUT_DIR/*.spv /data/local/tmp/"
echo "2. Set environment variables:"
echo "   export NI_QUANTIZED_MATMUL_SPV=/data/local/tmp/quantized_matmul.spv"
echo "   export NI_QUANTIZED_ATTENTION_SPV=/data/local/tmp/quantized_attention.spv"
echo "3. Run quantized inference test"