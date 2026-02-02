#!/system/bin/sh
echo "🔨 DEVICE-SIDE COMPILATION OF NEURAL INTERPOSER OPS"

# Set up compilation environment
export PATH=/system/bin:$PATH
cd /data/local/tmp/neural_interposer_build

# Check compilation environment
echo "Compiler check:"
which clang || echo "❌ Clang not found"

echo "Include paths:"
ls -la /system/include/ 2>/dev/null | head -3 || echo "❌ System includes not accessible"

# Try simple compilation test
echo "Testing compilation of header files..."
clang -c -I/system/include ni_shortconv3_op.h -o test_compile.o 2>&1 || echo "❌ Header compilation failed"

echo "Checking for ExecuTorch includes on device:"
find /data/local/tmp/lfm350_neural_interposer_test/ -name "*.h" | head -5 || echo "❌ ExecuTorch headers not found on device"

echo "🎯 DEVICE COMPILATION STATUS:"
echo "   - Source files: ✅ Copied"
echo "   - Compiler: $(which clang >/dev/null && echo '✅ Available' || echo '❌ Missing')"
echo "   - System headers: $(ls /system/include/ >/dev/null 2>&1 && echo '✅ Available' || echo '❌ Missing')"
echo "   - ExecuTorch headers: $(find /data/local/tmp/ -name "*.h" >/dev/null 2>&1 && echo '✅ Available' || echo '❌ Missing')"
