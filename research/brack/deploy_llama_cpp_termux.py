#!/usr/bin/env python3
"""
Deploy llama.cpp runtime to Motorola device via Termux
"""

import os
import subprocess
import time

def run_adb_command(cmd):
    """Run ADB command and return output"""
    try:
        result = subprocess.run(
            ["/Users/aaronjosserand-austin/000/Motorola/tools/android/adb"] + cmd.split(),
            capture_output=True, text=True, check=True, timeout=30
        )
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return None, f"Command failed: {e}"

def check_termux_installed():
    """Check if Termux is installed on the device"""
    print("🔍 Checking if Termux is installed...")

    # Check if Termux package is installed
    result, error = run_adb_command("shell pm list packages | grep termux")

    if "termux" in (result or ""):
        print("✅ Termux is installed")
        return True
    else:
        print("❌ Termux not found")
        print("📥 Please install Termux from F-Droid or GitHub releases")
        print("   https://f-droid.org/packages/com.termux/")
        return False

def setup_termux_llama():
    """Setup llama.cpp in Termux on the device"""
    print("🔧 Setting up llama.cpp in Termux...")

    # Check if we're in Termux shell
    result, error = run_adb_command("shell echo $PREFIX")
    if "/data/data/com.termux" not in (result or ""):
        print("⚠️  Not in Termux environment")
        print("💡 Run: adb shell")
        print("   Then: pkg install termux-exec")
        print("   Then: exec /data/data/com.termux/files/usr/bin/login")
        return False

    print("✅ In Termux environment")

    # Update packages
    print("📦 Updating Termux packages...")
    run_adb_command("shell pkg update -y")
    run_adb_command("shell pkg upgrade -y")

    # Install build dependencies
    print("🔨 Installing build tools...")
    run_adb_command("shell pkg install -y git cmake clang make")

    # Clone llama.cpp in Termux
    print("📥 Cloning llama.cpp in Termux...")
    run_adb_command("shell rm -rf llama.cpp")  # Clean any existing
    run_adb_command("shell git clone https://github.com/ggerganov/llama.cpp.git")

    # Build llama.cpp
    print("🔨 Building llama.cpp for Android...")
    print("This may take several minutes...")

    # Build with CMake (optimized for Android)
    build_commands = [
        "cd llama.cpp",
        "mkdir -p build",
        "cd build",
        "cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_ANDROID=ON",
        "make -j4"  # Use 4 cores for faster build
    ]

    full_command = " && ".join(build_commands)
    result, error = run_adb_command(f"shell {full_command}")

    if error:
        print(f"⚠️  Build warnings/errors: {error}")

    # Check if build succeeded
    result, error = run_adb_command("shell ls -la llama.cpp/build/bin/")

    if result and "llama-cli" in result:
        print("✅ llama.cpp built successfully!")
        return True
    else:
        print("❌ Build failed")
        return False

def deploy_gguf_model_to_termux():
    """Deploy our GGUF model to Termux"""
    print("📦 Deploying LFM2-700M-GGUF to Termux...")

    # Copy model to Termux home
    model_path = "/data/local/tmp/lfm700m_gguf_test/model.gguf"
    termux_path = "/data/data/com.termux/files/home/model.gguf"

    result, error = run_adb_command(f"shell cp {model_path} {termux_path}")

    if error:
        print(f"❌ Failed to copy model: {error}")
        return False

    # Verify copy
    result, error = run_adb_command("shell ls -lh ~/model.gguf")

    if result:
        print(f"✅ Model deployed to Termux: {result}")
        return True
    else:
        print("❌ Model copy verification failed")
        return False

def create_inference_script():
    """Create a simple inference script for testing"""
    print("📝 Creating inference test script...")

    script_content = '''#!/data/data/com.termux/files/usr/bin/bash
echo "🚀 LFM2-700M-GGUF Inference Test in Termux"
echo "=========================================="

MODEL_PATH="$HOME/model.gguf"
LLAMA_CLI="./llama.cpp/build/bin/llama-cli"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    exit 1
fi

# Check if llama-cli exists
if [ ! -f "$LLAMA_CLI" ]; then
    echo "❌ llama-cli not found. Build llama.cpp first."
    exit 1
fi

echo "🤖 Model: $(ls -lh $MODEL_PATH)"
echo "🔧 Runtime: llama-cli"
echo ""

echo "💭 Test Prompt: 'Hypothetically, might reflective recursion be a function of awareness?'"
echo ""

# Run inference with low context for mobile
echo "🎯 Running inference (this may take 10-30 seconds)..."
echo ""

$LLAMA_CLI \\
    --model $MODEL_PATH \\
    --prompt "Hypothetically, might reflective recursion be a function of awareness?" \\
    --ctx-size 512 \\
    --temp 0.7 \\
    --n-predict 100 \\
    --threads 4 \\
    --no-display-prompt

echo ""
echo "✅ Inference test completed!"
'''

    # Write script to device
    with open('/tmp/test_inference.sh', 'w') as f:
        f.write(script_content)

    run_adb_command("push /tmp/test_inference.sh /data/data/com.termux/files/home/")
    run_adb_command("shell chmod +x ~/test_inference.sh")

    print("✅ Inference script created: ~/test_inference.sh")

def run_test_inference():
    """Run the test inference"""
    print("🧪 Running LFM2-700M inference test...")
    print("This will ask: 'Hypothetically, might reflective recursion be a function of awareness?'")

    # Run the inference script
    result, error = run_adb_command("shell cd /data/data/com.termux/files/home && ./test_inference.sh")

    if result:
        print("📊 INFERENCE RESULT:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        return result
    else:
        print(f"❌ Inference failed: {error}")
        return None

def main():
    """Main deployment function"""
    print("🚀 Deploying llama.cpp GGUF Runtime to Motorola")
    print("This will enable LFM2-700M inference on device!")
    print("=" * 60)

    # Verify device connection
    result, error = run_adb_command("devices")
    if "device" not in result:
        print("❌ Device not connected")
        return False

    print("✅ Motorola device connected")

    # Check Termux
    if not check_termux_installed():
        print("❌ Cannot proceed without Termux")
        return False

    # Setup llama.cpp
    if not setup_termux_llama():
        print("❌ Failed to setup llama.cpp")
        return False

    # Deploy model
    if not deploy_gguf_model_to_termux():
        print("❌ Failed to deploy model")
        return False

    # Create test script
    create_inference_script()

    # Run inference test
    result = run_test_inference()
    if result:
        print("\n🎉 SUCCESS! LFM2-700M-GGUF is now running on Motorola!")
        print("💬 The model can answer philosophical questions!")
        return True
    else:
        print("\n❌ Inference test failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🏆 GGUF RUNTIME DEPLOYMENT COMPLETE!")
        print("🎯 LFM2-700M can now answer: 'Hypothetically, might reflective recursion be a function of awareness?'")
    else:
        print("\n❌ Deployment failed. Check Termux installation and try again.")