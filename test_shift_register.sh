#!/bin/bash
# test_shift_register.sh - Test Shift-Register MatMul Optimization

set -e

echo "🧪 Testing Shift-Register MatMul Optimization"
echo "=============================================="

# Clean and build
echo "Building shift-register MatMul library..."
rm -f mt6855v_matmul_shift.o mt6855v_matmul_shift.so

# Compile C wrapper
gcc -c mt6855v_matmul_shift.c -o mt6855v_matmul_shift.o -O3 -Wall

# Assemble shift-register kernels
gcc -c mt6855v_matmul_shift.S -o mt6855v_matmul_shift_asm.o -march=armv8.2-a+dotprod+fp16

# Link shared library
gcc -shared -o mt6855v_matmul_shift.so \
    mt6855v_matmul_shift.o \
    mt6855v_matmul_shift_asm.o \
    -lm

echo "✅ Build complete!"
ls -lh mt6855v_matmul_shift.so

echo ""
echo "🧪 Testing shift-register MatMul..."
echo ""

# Create test program
cat > test_shift.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern int matmul_shift_optimized(int16_t* out, const int8_t* weights, const int8_t* act, int n_out, int n_in);
extern int matmul_shift_available(void);

int main() {
    printf("Shift-Register MatMul Test\n");
    printf("============================\n");
    
    int16_t out[32];
    int8_t weights[256];
    int8_t act[32];
    
    // Initialize test data
    for (int i = 0; i < 256; i++) weights[i] = (i % 128) - 64;
    for (int i = 0; i < 32; i++) act[i] = (i % 16) - 8;
    
    if (!matmul_shift_available()) {
        printf("❌ Shift-register library not available\n");
        return 1;
    }
    
    printf("Testing Int8×Int16 MatMul (32×32)...\n");
    matmul_shift_optimized(out, weights, act, 32, 32);
    
    // Verify some results
    int errors = 0;
    for (int i = 0; i < 10; i++) {
        int16_t expected = (weights[0] * act[i]) + (weights[1] * act[i+1]);
        if (out[i] != expected) {
            printf("Error at %d: got %d, expected %d\n", i, out[i], expected);
            errors++;
        }
    }
    
    printf("%s\n", errors == 0 ? "✅ All results correct!" : "❌ Some errors detected");
    return errors == 0 ? 0 : 1;
}
EOF

# Compile test program
gcc -o test_shift test_shift.c -O3

echo ""
echo "🔍 Running test program..."
./test_shift

echo ""
echo "✅ Test complete!"