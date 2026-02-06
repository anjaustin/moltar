#!/usr/bin/env python3
"""
Analyze Q4_0 weight distribution and simulate ternary conversion.

Q4_0 format: 4-bit signed integers from -8 to +7, stored as unsigned 0-15
with scale per 32-weight block.

Question: If we convert Q4_0 to ternary {-1, 0, +1}, what's the sparsity
and what's the quality loss?
"""

import numpy as np
import struct
import sys

# Q4_0 block: 32 weights, 1 FP16 scale
# Layout: 2 bytes scale + 16 bytes (32 * 4 bits)
BLOCK_SIZE = 32
BLOCK_BYTES = 18  # 2 + 16


def read_gguf_q4_0(path):
    """Read Q4_0 tensor from GGUF file (simplified - just stats)"""
    # We don't need to fully parse GGUF, just analyze the raw bytes
    # For now, let's create synthetic Q4_0 data that matches real distributions
    pass


def analyze_q4_distribution():
    """
    Analyze what happens when Q4_0 values are ternarized.

    Q4_0 values: -8 to +7 (stored as 0-15, subtract 8)
    Actual value = q4_value * scale

    If we ternarize at threshold T:
      -8 to -T-1 -> -1
      -T to +T   -> 0
      +T+1 to +7 -> +1
    """

    # Simulate typical Q4_0 distribution (roughly Gaussian, centered)
    np.random.seed(42)

    # LFM2-350M has ~350M parameters. Let's sample a representative chunk.
    n_weights = 10_000_000  # 10M weights for analysis

    # Typical trained weight distribution is roughly Gaussian
    # After Q4_0 quantization, we get integers -8 to +7
    raw_weights = np.random.randn(n_weights) * 2  # typical weight std ~2
    raw_weights = np.clip(raw_weights, -8, 7).astype(np.int8)

    print("=" * 60)
    print("Q4_0 Weight Distribution Analysis")
    print("=" * 60)

    print(f"\nSample size: {n_weights:,} weights")
    print(f"\nQ4_0 value distribution:")
    for v in range(-8, 8):
        count = np.sum(raw_weights == v)
        pct = 100 * count / n_weights
        bar = "#" * int(pct / 2)
        print(f"  {v:+2d}: {pct:5.1f}% {bar}")

    print("\n" + "=" * 60)
    print("Ternary Conversion Analysis")
    print("=" * 60)

    # Try different thresholds
    for threshold in [0, 1, 2, 3, 4]:
        pos = np.sum(raw_weights > threshold)
        neg = np.sum(raw_weights < -threshold)
        zero = n_weights - pos - neg

        sparsity = 100 * zero / n_weights

        # Compute error: ternary vs original
        ternary = np.zeros_like(raw_weights)
        ternary[raw_weights > threshold] = 1
        ternary[raw_weights < -threshold] = -1

        # Scale ternary to match original magnitude
        # E[|x|] for non-zero original weights
        nonzero_mask = raw_weights != 0
        if np.sum(nonzero_mask) > 0:
            avg_magnitude = np.mean(np.abs(raw_weights[nonzero_mask]))
        else:
            avg_magnitude = 1

        scaled_ternary = ternary * avg_magnitude
        mse = np.mean((raw_weights.astype(np.float32) - scaled_ternary) ** 2)
        rmse = np.sqrt(mse)

        # Relative error
        orig_rms = np.sqrt(np.mean(raw_weights.astype(np.float32) ** 2))
        rel_error = 100 * rmse / orig_rms if orig_rms > 0 else 0

        print(f"\nThreshold = {threshold}:")
        print(
            f"  +1: {100 * pos / n_weights:5.1f}%  0: {sparsity:5.1f}%  -1: {100 * neg / n_weights:5.1f}%"
        )
        print(f"  Sparsity: {sparsity:.1f}%")
        print(f"  RMSE: {rmse:.3f} ({rel_error:.1f}% relative)")
        print(f"  Ops reduction: {sparsity:.0f}% (from skip zeros)")

    print("\n" + "=" * 60)
    print("Key Insight for LFM2")
    print("=" * 60)
    print("""
Q4_0 has 16 discrete values (-8 to +7).
Ternary has 3 values (-1, 0, +1).

The problem: LFM2 was trained with 4-bit precision.
Converting to ternary loses significant information.

For CfC cells (Yinsen), ternary works because:
1. Small networks (8-32 neurons)
2. Can retrain/fine-tune for ternary
3. Task is anomaly detection, not language generation

For LFM2, ternary would require:
1. Ternary-aware fine-tuning (expensive)
2. Accept significant quality loss
3. Or: Decompose Q4_0 as sum of ternary matrices (complex)

The "frozen 8x8 matvec" idea might be:
- Precompute all possible 8-element dot products
- Q4_0 activation (4-bit) x Q4_0 weight (4-bit) = 8-bit result
- For 8 elements: 2^32 possible inputs -> too large

Actually, the key insight is:
- Q4_0 has discrete values
- For small blocks, we can precompute ALL possible results
- But 32 elements (Q4_0 block) x 4-bit each = 128 bits = 2^128 combinations
  -> Way too large

Unless... we decompose differently:
- Each Q4_0 value is 4 bits = 16 possible values
- Each weight element contributes weight * activation
- Could precompute 16x16 = 256 products per weight position
- Then just add them up (table lookup + add)

This IS the SDOT approach but with more optimization potential.
""")


if __name__ == "__main__":
    analyze_q4_distribution()
