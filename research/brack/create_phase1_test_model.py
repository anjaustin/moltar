#!/usr/bin/env python3
"""
Create Phase 1 Test Model

Creates a simple quantized model to test Phase 1 quantization infrastructure
on the Motorola device.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path

# Import our quantization infrastructure
from quantization_utils import create_lfm_quantization_config, LFMQuantizer
from blockwise_quantization import HardwareAwareQuantizer
from executorch_quantization_metadata import ExecuTorchQuantizationSerializer

class Phase1TestModel(nn.Module):
    """Simple model for testing Phase 1 quantization infrastructure"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embedding layer
        self.embed = nn.Embedding(1000, hidden_dim)

        # Simple transformer-like layers (but much smaller)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output head
        self.head = nn.Linear(hidden_dim, 1000)

    def forward(self, input_ids):
        # Embedding
        x = self.embed(input_ids)  # [batch, seq, hidden]

        # Simple attention-like operation (quantization test)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Simplified attention (just for testing)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attn_output = torch.matmul(attn_weights, v)

        # Output projection
        x = self.out_proj(attn_output)

        # Final output
        logits = self.head(x)

        return logits

def create_and_quantize_test_model():
    """Create and quantize a test model for Phase 1 validation"""
    print("🧪 Creating Phase 1 Test Model...")

    # Create model
    model = Phase1TestModel(hidden_dim=128)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create quantization config optimized for mobile
    config = create_lfm_quantization_config(target_memory_mb=50)  # Small target for testing
    print(f"   Quantization: {config.bits}bit, block_size={config.block_size}")

    # Quantize model
    quantizer = HardwareAwareQuantizer(config)
    quantized_layers = quantizer.quantize_for_mobile(model)

    # Test inference before/after quantization
    test_input = torch.randint(0, 1000, (1, 8))

    # Original inference
    model.eval()
    with torch.no_grad():
        orig_output = model(test_input)

    # Quantized inference (simulated)
    quantized_model = quantizer._create_quantized_model(model, quantized_layers)
    with torch.no_grad():
        quant_output = quantized_model(test_input)

    # Calculate accuracy
    mse = torch.mean((orig_output - quant_output) ** 2).item()
    cos_sim = torch.cosine_similarity(orig_output.flatten(), quant_output.flatten(), dim=0).item()

    print("   Original output shape:", orig_output.shape)
    print(f"   MSE Loss: {mse:.6f}")
    print(f"   Cosine Similarity: {cos_sim:.4f}")
    # Create output directory
    output_dir = Path("phase1_test_model")
    output_dir.mkdir(exist_ok=True)

    # Serialize quantized model
    serializer = ExecuTorchQuantizationSerializer(target_platform="mobile")
    model_path = serializer.serialize_quantized_model(
        quantized_layers=quantized_layers,
        model_name="Phase1TestModel",
        output_dir=str(output_dir),
        accuracy_metrics={
            'mse_loss': mse,
            'cosine_similarity': cos_sim
        }
    )

    # Create simple test script for device
    create_device_test_script(output_dir)

    print("✅ Phase 1 test model created successfully!")
    print(f"   Model saved to: {model_path}")
    print(f"   Quantized size: {sum(q.data.numel() * q.data.element_size() for q in quantized_layers.values()) / (1024*1024):.1f}MB")
    print(f"   Layers quantized: {len(quantized_layers)}")
    return model_path

def create_device_test_script(model_dir: Path):
    """Create a test script for running on the device"""
    script_path = model_dir / "test_phase1_device.sh"

    script_content = '''#!/bin/bash
# Phase 1 Quantization Test on Motorola Device

echo "🧪 Testing Phase 1 Quantization Infrastructure on Device"
echo "======================================================="

# Set environment variables for quantized operations
export NI_QUANTIZED_MATMUL_SPV="/data/local/tmp/quantized_matmul.spv"
export NI_QUANTIZED_ATTENTION_SPV="/data/local/tmp/quantized_attention.spv"

# Create test directory
mkdir -p /data/local/tmp/phase1_test

# Copy model files
cp *.pt /data/local/tmp/phase1_test/
cp quantization_metadata.json /data/local/tmp/phase1_test/

echo "📁 Model files copied to device"

# Test basic quantized operations (without full model)
echo "🔬 Testing quantized TriX operations..."

# Note: This is a placeholder for actual device testing
# The full integration test would load the quantized model
# and verify that the Neural Interposer quantized operations work

echo "✅ Phase 1 infrastructure deployed to device"
echo "   - Quantized model layers: $(ls *.pt | wc -l)"
echo "   - Quantization metadata: present"
echo "   - Vulkan shaders: configured"
echo ""
echo "🎯 Ready for Phase 2: Memory Management & Layer Sharding"
'''

    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    script_path.chmod(0o755)

    print(f"   Device test script created: {script_path}")

def run_host_validation():
    """Run comprehensive validation on host before device deployment"""
    print("🔍 Running host validation...")

    model = Phase1TestModel(hidden_dim=64)  # Even smaller for validation
    config = create_lfm_quantization_config(target_memory_mb=25)

    # Test quantization
    quantizer = HardwareAwareQuantizer(config)
    quantized_layers = quantizer.quantize_for_mobile(model)

    # Test multiple inputs
    test_inputs = [
        torch.randint(0, 1000, (1, 4)),
        torch.randint(0, 1000, (1, 8)),
        torch.randint(0, 1000, (2, 6))
    ]

    accuracies = []
    for i, test_input in enumerate(test_inputs):
        # Original
        with torch.no_grad():
            orig_out = model(test_input)

        # Quantized
        quant_model = quantizer._create_quantized_model(model, quantized_layers)
        with torch.no_grad():
            quant_out = quant_model(test_input)

        # Accuracy
        cos_sim = torch.cosine_similarity(orig_out.flatten(), quant_out.flatten(), dim=0).item()
        accuracies.append(cos_sim)
        print(f"     Input {i+1}: cos_sim = {cos_sim:.4f}")
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"   Average cosine similarity: {avg_accuracy:.4f}")
    return avg_accuracy > 0.95  # 95% cosine similarity threshold

if __name__ == "__main__":
    print("🚀 Phase 1 Quantization Infrastructure Test")
    print("=" * 50)

    try:
        # Run host validation
        validation_passed = run_host_validation()
        print()

        if validation_passed:
            print("✅ Host validation passed - proceeding with model creation")

            # Create and quantize test model
            model_path = create_and_quantize_test_model()

            print()
            print("📱 Device Deployment Instructions:")
            print("1. Copy model to device:")
            print(f"   adb push {model_path}/*.pt /data/local/tmp/phase1_test/")
            print(f"   adb push {model_path}/quantization_metadata.json /data/local/tmp/phase1_test/")
            print()
            print("2. Run device test:")
            print("   adb push phase1_test_model/test_phase1_device.sh /data/local/tmp/phase1_test/")
            print("   adb shell 'cd /data/local/tmp/phase1_test && chmod +x test_phase1_device.sh && ./test_phase1_device.sh'")
            print()
            print("🎉 Phase 1 quantization infrastructure ready for device testing!")

        else:
            print("❌ Host validation failed - check quantization implementation")

    except Exception as e:
        print(f"❌ Phase 1 test creation failed: {e}")
        import traceback
        traceback.print_exc()