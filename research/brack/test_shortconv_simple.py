#!/usr/bin/env python3
"""
Simple ShortConv test for Neural Interposer
"""

import torch
import torch.nn as nn
import time

class SimpleShortConv(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(1000, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, groups=hidden_dim)
        self.head = nn.Linear(hidden_dim, 1000)

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)  # [B, H, S]
        x = self.conv(x).transpose(1, 2)   # [B, S, H]
        return self.head(x)

def test_simple():
    print("🧪 Testing simple ShortConv model...")

    model = SimpleShortConv()
    model.eval()

    input_ids = torch.randint(0, 1000, (1, 8))

    with torch.no_grad():
        start = time.time()
        output = model(input_ids)
        elapsed = time.time() - start

    print("✅ Inference completed")
    print(f"   Output shape: {output.shape}")
    print(".3f")
    print(f"   Tokens/sec: {input_ids.numel() / elapsed:.1f}")

    return True

if __name__ == "__main__":
    test_simple()
    print("🎉 Simple ShortConv test passed!")