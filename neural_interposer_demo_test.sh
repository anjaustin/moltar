#!/system/bin/sh
echo "🧪 NEURAL INTERPOSER MINIMAL DEMO"

cd /data/local/tmp/lfm350_neural_interposer_test

echo "Testing what we know works..."
echo "1. Toy Vulkan model:"
./executorch_runner --model_path /data/local/tmp/toy_vulkan.pte 2>&1 | head -5

echo ""
echo "2. Checking if we can access Neural Interposer components:"
if [ -f "shortconv_chip.spv" ]; then
    echo "✅ Vulkan shader available: $(stat -c%s shortconv_chip.spv) bytes"
else
    echo "❌ Vulkan shader missing"
fi

echo ""
echo "3. Testing if we can create a hybrid approach..."
# Let's try to run multiple models sequentially to simulate Neural Interposer pipeline

echo "Running toy model + attempting shortconv (simulated pipeline):"
./executorch_runner --model_path /data/local/tmp/toy_vulkan.pte --num_executions=1 >/dev/null 2>&1 && echo "✅ Toy model executed successfully"

echo "Attempting shortconv integration..."
# Try to load shortconv after toy model to see if it helps
./executorch_runner --model_path /data/local/tmp/smoke_shortconv3.pte 2>&1 | head -3

echo ""
echo "🎯 DEMO RESULT:"
echo "Runner works for basic models, Neural Interposer ops needed for LFM"
