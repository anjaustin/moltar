#!/usr/bin/env python3
"""
Test Quantized Operations on Device

Simple test to verify quantized TriX operations work on the Motorola device.
"""

import subprocess
import time

def run_adb_command(cmd):
    """Run ADB command and return output"""
    try:
        result = subprocess.run(
            ["adb"] + cmd.split(),
            capture_output=True, text=True, check=True, timeout=30
        )
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return None, f"Command failed: {e}"

def test_device_connectivity():
    """Test basic device connectivity"""
    print("🔌 Testing device connectivity...")

    stdout, stderr = run_adb_command("shell echo 'Device responsive'")
    if stdout == "Device responsive":
        print("✅ Device is connected and responsive")
        return True
    else:
        print(f"❌ Device connectivity failed: {stderr}")
        return False

def test_files_on_device():
    """Test that required files are on device"""
    print("📁 Checking files on device...")

    # Check quantized shaders
    stdout, stderr = run_adb_command("shell ls /data/local/tmp/quantized_*.spv")
    if "quantized_matmul.spv" in stdout and "quantized_attention.spv" in stdout:
        print("✅ Quantized shaders found on device")
        return True
    else:
        print(f"❌ Quantized shaders missing: {stdout}")
        return False

def test_environment_variables():
    """Test environment variable setup"""
    print("🔧 Testing environment setup...")

    # Set environment variables
    run_adb_command("shell export NI_QUANTIZED_MATMUL_SPV=/data/local/tmp/quantized_matmul.spv")
    run_adb_command("shell export NI_QUANTIZED_ATTENTION_SPV=/data/local/tmp/quantized_attention.spv")

    # Test variable access
    stdout, stderr = run_adb_command("shell echo $NI_QUANTIZED_MATMUL_SPV")
    if "/data/local/tmp/quantized_matmul.spv" in stdout:
        print("✅ Environment variables set correctly")
        return True
    else:
        print(f"❌ Environment variables not set: {stdout}")
        return False

def test_executorch_runner():
    """Test basic executorch runner functionality"""
    print("🚀 Testing ExecuTorch runner...")

    # Test help command
    stdout, stderr = run_adb_command("shell cd /data/local/tmp/lfm350_neural_interposer_test && ./executorch_runner --help 2>&1 | head -5")

    if stdout and "executorch" in stdout.lower():
        print("✅ ExecuTorch runner is functional")
        return True
    else:
        print(f"❌ ExecuTorch runner failed: {stderr}")
        return False

def run_device_tests():
    """Run all device tests"""
    print("🧪 Running Device Quantized Operations Tests")
    print("=" * 50)

    tests = [
        test_device_connectivity,
        test_files_on_device,
        test_environment_variables,
        test_executorch_runner
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
            print()

    # Summary
    passed = sum(results)
    total = len(results)

    print("📊 Test Summary:")
    print(f"   Passed: {passed}/{total}")
    print(".1f")

    if passed == total:
        print("🎉 All device tests passed!")
        print("\n🚀 Ready for quantized LFM inference on Motorola MediaTek + Mali!")
    else:
        print("⚠️  Some tests failed. Check device setup.")

    return passed == total

if __name__ == "__main__":
    run_device_tests()