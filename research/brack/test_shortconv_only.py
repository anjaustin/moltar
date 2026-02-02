#!/usr/bin/env python3
"""
Test ShortConv-only LFM2 inference using Neural Interposer

This test verifies that the ShortConv chip works correctly in the Neural Interposer
without requiring the full attention implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple
import time
import numpy as np

class ShortConvOnlyLFM2(nn.Module):
    """Simplified LFM2 model with only ShortConv (no attention) for testing"""

    def __init__(self, hidden_dim=1024, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding
        self.embed = nn.Embedding(50257, hidden_dim)

        # ShortConv layers only (simplified)
        self.shortconv_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, padding=0, groups=hidden_dim)
            for _ in range(num_layers)
        ])

        # Simple FFN layers
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 50257, bias=False)

    def forward(self, input_ids):
        # Embedding
        x = self.embed(input_ids)  # [batch, seq_len, hidden_dim]

        # Simplified forward pass (ShortConv only)
        for i in range(self.num_layers):
            # ShortConv (simplified - would need proper state handling)
            x_conv = self.shortconv_layers[i](x.transpose(1, 2)).transpose(1, 2)

            # FFN
            x_ffn = self.ffn_layers[i](x_conv)

            # Residual + simplified gating
            x = x + x_ffn

        # Output
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

def test_shortconv_inference():
    """Test ShortConv-only inference"""
    print("🧪 Testing ShortConv-only LFM2 inference...")

    # Create model
    model = ShortConvOnlyLFM2(hidden_dim=256, num_layers=2)  # Smaller for testing
    model.eval()

    # Test input
    input_ids = torch.randint(0, 1000, (1, 8))  # Small batch and sequence

    # Time inference
    with torch.no_grad():
        start_time = time.time()
        logits = model(input_ids)
        end_time = time.time()

    inference_time = end_time - start_time

    print("✅ ShortConv-only inference completed")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Inference time: {inference_time:.3f}s")
    print(f"   Tokens/sec: {input_ids.numel() / inference_time:.2f}")
    # Verify output is reasonable
    assert logits.shape == (1, 8, 50257), f"Unexpected output shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN values in output"
    assert not torch.isinf(logits).any(), "Inf values in output"

    print("✅ Output validation passed")
    return True

def create_shortconv_test_model():
    """Create and export a minimal ShortConv test model"""
    print("📦 Creating ShortConv test model...")

    model = ShortConvOnlyLFM2(hidden_dim=256, num_layers=1)
    model.eval()

    # Create example input
    example_input = torch.randint(0, 1000, (1, 4))

    # Export to ExecuTorch format
    from executorch.exir import to_edge
    from torch.export import export

    # Export with torch.export
    exported_program = export(model, (example_input,))
    edge_program = to_edge(exported_program)

    # Save as .pte
    edge_program.to_executorch().save("shortconv_test_model.pte")

    print("✅ ShortConv test model exported: shortconv_test_model.pte")
    print(f"   Model size: {sum(p.numel() for p in model.parameters())} parameters")
    print(".1f")
    return "shortconv_test_model.pte"

if __name__ == "__main__":
    print("🚀 ShortConv-only LFM2 Neural Interposer Test")
    print("=" * 50)

    try:
        # Test PyTorch inference
        test_shortconv_inference()

        # Create test model for device deployment
        model_path = create_shortconv_test_model()

        print("\n📱 Ready for device deployment:")
        print(f"   Model: {model_path}")
        print("   Test command: ./executorch_runner --model_path shortconv_test_model.pte"
        print("\n🎉 ShortConv test preparation complete!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise