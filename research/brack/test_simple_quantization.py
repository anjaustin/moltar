#!/usr/bin/env python3
"""
Simple Quantization Test

Basic test of the quantization infrastructure
"""

import torch
from quantization_utils import create_lfm_quantization_config, LFMQuantizer

def test_basic_quantization():
    print("🧪 Testing basic quantization...")

    # Create test tensor
    tensor = torch.randn(100, 200)
    print(f"  Input tensor shape: {tensor.shape}")

    # Create quantizer
    config = create_lfm_quantization_config(target_memory_mb=200)
    quantizer = LFMQuantizer(config)

    # Quantize
    quantized = quantizer.block_quantizer.quantize_tensor(tensor)
    print(f"  Quantized bits: {quantized.bits}")
    print(f"  Block size: {quantized.block_size}")

    # Dequantize
    dequantized = quantizer.block_quantizer.dequantize_tensor(quantized)
    print(f"  Dequantized shape: {dequantized.shape}")

    # Calculate error
    mse = torch.mean((tensor - dequantized) ** 2).item()
    print(f"  MSE Loss: {mse:.6f}")

    # Calculate compression
    orig_size = tensor.numel() * tensor.element_size()
    quant_size = quantized.data.numel() * quantized.data.element_size()
    ratio = orig_size / quant_size
    print(f"  Compression ratio: {ratio:.1f}x")

    print("✅ Basic quantization test passed!")
    return True

if __name__ == "__main__":
    test_basic_quantization()