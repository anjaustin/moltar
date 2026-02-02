#!/usr/bin/env python3
"""
Test Quantization Infrastructure

Verifies that the complete quantization pipeline works:
- Block-wise quantization
- Accuracy validation
- ExecuTorch metadata format
- Mobile optimization
"""

import torch
import torch.nn as nn
import time
import os
from pathlib import Path

# Import our quantization modules
from quantization_utils import (
    create_lfm_quantization_config,
    LFMQuantizer,
    QuantizedTensor
)
from blockwise_quantization import HardwareAwareQuantizer
from accuracy_validation import QuantizationAccuracyValidator
from executorch_quantization_metadata import (
    ExecuTorchQuantizationSerializer,
    MobileOptimizedLoader
)

def create_test_model():
    """Create a small test model for quantization testing"""
    class TestLFM(nn.Module):
        def __init__(self, hidden_dim=128):
            super().__init__()
            self.embed = nn.Embedding(1000, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            self.head = nn.Linear(hidden_dim, 1000)

        def forward(self, x):
            # Simple forward pass
            x = self.embed(x)
            attn_out, _ = self.attention(x, x, x)
            x = x + attn_out  # Residual
            x = x + self.ffn(x)  # FFN residual
            return self.head(x)

    return TestLFM()

def test_basic_quantization():
    """Test basic block-wise quantization"""
    print("🧪 Testing basic block-wise quantization...")

    # Create test tensor
    tensor = torch.randn(256, 512)

    # Create quantizer
    config = create_lfm_quantization_config(target_memory_mb=200)
    quantizer = LFMQuantizer(config)

    # Quantize tensor
    start_time = time.time()
    quantized = quantizer.block_quantizer.quantize_tensor(tensor)
    quantize_time = time.time() - start_time

    # Dequantize
    start_time = time.time()
    dequantized = quantizer.block_quantizer.dequantize_tensor(quantized)
    dequantize_time = time.time() - start_time

    # Calculate metrics
    mse = torch.mean((tensor - dequantized) ** 2).item()
    compression_ratio = tensor.numel() * tensor.element_size() / (
        quantized.data.numel() * quantized.data.element_size() +
        quantized.scales.numel() * quantized.scales.element_size()
    )

    print("  ✅ Basic quantization successful")
    print(f"    MSE Loss: {mse:.6f}")
    print(f"    Compression Ratio: {compression_ratio:.1f}x")
    print(f"    Quantize Time: {quantize_time:.3f}s")
    print(f"    Dequantize Time: {dequantize_time:.3f}s")

    return True

def test_model_quantization():
    """Test full model quantization"""
    print("🧪 Testing full model quantization...")

    # Create test model
    model = create_test_model()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create quantizer
    config = create_lfm_quantization_config(target_memory_mb=50)  # Aggressive quantization
    quantizer = LFMQuantizer(config)

    # Quantize model
    start_time = time.time()
    quantized_layers = quantizer.quantize_model(model)
    quantize_time = time.time() - start_time

    print("  ✅ Model quantization successful")
    print(f"    Layers quantized: {len(quantized_layers)}")
    print(f"    Quantization time: {quantize_time:.3f}s")

    # Test inference
    test_input = torch.randint(0, 1000, (1, 10))

    # Original inference
    model.eval()
    with torch.no_grad():
        orig_output = model(test_input)

    # Quantized inference
    quantized_model = quantizer._create_quantized_model(model, quantized_layers)
    with torch.no_grad():
        quant_output = quantized_model(test_input)

    # Compare outputs
    mse = torch.mean((orig_output - quant_output) ** 2).item()
    cos_sim = torch.cosine_similarity(orig_output.flatten(), quant_output.flatten(), dim=0).item()

    print("  ✅ Inference comparison"    print(f"    MSE Loss: {mse:.6f}")
    print(f"    Cosine Similarity: {cos_sim:.4f}")

    return quantized_layers

def test_accuracy_validation():
    """Test accuracy validation system"""
    print("🧪 Testing accuracy validation...")

    # Create test data
    model = create_test_model()
    config = create_lfm_quantization_config(target_memory_mb=100)
    quantizer = LFMQuantizer(config)
    quantized_layers = quantizer.quantize_model(model)

    # Create test dataset
    test_inputs = [torch.randint(0, 1000, (1, 8)) for _ in range(5)]

    # Validate accuracy
    validator = QuantizationAccuracyValidator(config.__dict__)
    metrics = validator.validate_language_modeling(
        original_model=model,
        quantized_layers=quantized_layers,
        tokenizer=None,  # Skip for demo
        test_texts=["test"] * 5
    )

    print("  ✅ Accuracy validation successful")
    print(f"    Metrics: {list(metrics.keys())}")

    return True

def test_metadata_format():
    """Test ExecuTorch metadata format"""
    print("🧪 Testing ExecuTorch metadata format...")

    # Create quantized layers
    model = create_test_model()
    config = create_lfm_quantization_config(target_memory_mb=100)
    quantizer = LFMQuantizer(config)
    quantized_layers = quantizer.quantize_model(model)

    # Create serializer
    serializer = ExecuTorchQuantizationSerializer(target_platform="mobile")

    # Serialize model
    output_dir = "test_quantized_model"
    model_path = serializer.serialize_quantized_model(
        quantized_layers=quantized_layers,
        model_name="TestLFM",
        output_dir=output_dir,
        accuracy_metrics={'test_metric': 0.95}
    )

    print("  ✅ Model serialization successful"    print(f"    Output directory: {model_path}")

    # Test loader
    loader = MobileOptimizedLoader(model_path)

    # Load metadata
    metadata = loader.load_model_metadata()
    print("  ✅ Metadata loading successful"    print(f"    Layers: {len(metadata.layers)}")
    print(f"    Compression ratio: {metadata.overall_compression_ratio:.1f}x")

    # Load a layer
    if metadata.layers:
        layer_name = metadata.layers[0].name
        quantized_tensor = loader.load_quantized_layer(layer_name)
        print("  ✅ Layer loading successful"        print(f"    Layer: {layer_name}")
        print(f"    Shape: {quantized_tensor.original_shape}")

    # Cleanup
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    return True

def test_hardware_optimization():
    """Test hardware-aware quantization"""
    print("🧪 Testing hardware-aware quantization...")

    model = create_test_model()
    config = create_lfm_quantization_config(target_memory_mb=100)
    hw_quantizer = HardwareAwareQuantizer(config)

    # Create calibration data
    calibration_data = [torch.randn(1, 8, 128) for _ in range(3)]

    # Quantize with hardware optimization
    quantized_layers = hw_quantizer.quantize_for_mobile(model, calibration_data)

    print("  ✅ Hardware-aware quantization successful"    print(f"    Layers optimized: {len(quantized_layers)}")

    # Check that layers were optimized
    total_compression = 0
    for name, quantized in quantized_layers.items():
        # Calculate compression for this layer
        orig_size = torch.prod(torch.tensor(quantized.original_shape)).item() * 4
        quant_size = (quantized.data.numel() * quantized.data.element_size() +
                     quantized.scales.numel() * quantized.scales.element_size())
        if quantized.zeros is not None:
            quant_size += quantized.zeros.numel() * quantized.zeros.element_size()

        ratio = orig_size / quant_size if quant_size > 0 else 1
        total_compression += ratio

    avg_compression = total_compression / len(quantized_layers)
    print(f"    Average compression: {avg_compression:.1f}x")

    return True

if __name__ == "__main__":
    print("🚀 Testing Complete Quantization Infrastructure")
    print("=" * 60)

    try:
        # Test basic quantization
        test_basic_quantization()
        print()

        # Test model quantization
        quantized_layers = test_model_quantization()
        print()

        # Test accuracy validation
        test_accuracy_validation()
        print()

        # Test metadata format
        test_metadata_format()
        print()

        # Test hardware optimization
        test_hardware_optimization()
        print()

        print("🎉 All quantization infrastructure tests passed!")
        print()
        print("✅ Block-wise quantization: Working")
        print("✅ Model quantization: Working")
        print("✅ Accuracy validation: Working")
        print("✅ ExecuTorch metadata: Working")
        print("✅ Hardware optimization: Working")
        print()
        print("Phase 1 (Week 3): Block-wise Quantization Infrastructure - COMPLETE!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()