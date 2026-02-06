#!/usr/bin/env python3
"""
Compare quantization error: per-block scale vs global scale

Q4_0 format:
- Per-block: Each 32-element block has its own FP16 scale
- Global: Entire tensor shares one scale

We'll load actual LFM2 weights and measure the error.
"""

import numpy as np
import struct
import sys


def read_gguf_tensor(filepath, tensor_name):
    """Very simplified GGUF reader - just for Q4_0 tensors"""
    # This is a stub - for real testing we'd need proper GGUF parsing
    pass


def quantize_q4_per_block(x, block_size=32):
    """Quantize to Q4_0 format with per-block scale"""
    x = x.flatten()
    n_blocks = len(x) // block_size
    x = x[: n_blocks * block_size].reshape(n_blocks, block_size)

    # Per-block scale: max absolute value in block
    scales = np.abs(x).max(axis=1, keepdims=True)
    scales = np.where(scales == 0, 1, scales)

    # Quantize to 4-bit (-8 to 7)
    q = np.round(x / scales * 7).astype(np.int8)
    q = np.clip(q, -8, 7)

    # Dequantize
    x_hat = q.astype(np.float32) * scales / 7

    return x_hat.flatten(), scales.flatten()


def quantize_q4_global(x, block_size=32):
    """Quantize to Q4_0 format with global scale"""
    x_flat = x.flatten()
    n = len(x_flat)
    n_blocks = n // block_size
    x_flat = x_flat[: n_blocks * block_size]

    # Global scale: max absolute value in entire tensor
    global_scale = np.abs(x_flat).max()
    if global_scale == 0:
        global_scale = 1

    # Quantize to 4-bit (-8 to 7)
    q = np.round(x_flat / global_scale * 7).astype(np.int8)
    q = np.clip(q, -8, 7)

    # Dequantize
    x_hat = q.astype(np.float32) * global_scale / 7

    return x_hat, global_scale


def quantize_q4_per_row(x, block_size=32):
    """Quantize to Q4 format with per-row scale (for 2D weight matrix)"""
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_rows, n_cols = x.shape
    # Pad columns to multiple of block_size
    n_cols_padded = (n_cols // block_size) * block_size
    x = x[:, :n_cols_padded]

    # Per-row scale: max absolute value in each row
    scales = np.abs(x).max(axis=1, keepdims=True)
    scales = np.where(scales == 0, 1, scales)

    # Quantize to 4-bit (-8 to 7)
    q = np.round(x / scales * 7).astype(np.int8)
    q = np.clip(q, -8, 7)

    # Dequantize
    x_hat = q.astype(np.float32) * scales / 7

    return x_hat.flatten(), scales.flatten()


def main():
    # Generate synthetic weight tensor mimicking neural network weights
    # Neural network weights typically follow a Gaussian distribution
    np.random.seed(42)

    print("=== Quantization Error Analysis ===\n")

    # Test different tensor sizes and distributions
    test_cases = [
        ("Uniform [-1, 1]", lambda n: np.random.uniform(-1, 1, n)),
        ("Gaussian N(0, 0.1)", lambda n: np.random.randn(n) * 0.1),
        ("Gaussian N(0, 0.01)", lambda n: np.random.randn(n) * 0.01),
        (
            "Mixed scales",
            lambda n: np.concatenate(
                [
                    np.random.randn(n // 3) * 0.001,
                    np.random.randn(n // 3) * 0.01,
                    np.random.randn(n // 3) * 0.1,
                ]
            ),
        ),
        (
            "Outliers",
            lambda n: np.concatenate(
                [
                    np.random.randn(n - 10) * 0.01,
                    np.random.randn(10) * 1.0,  # 10 outliers
                ]
            ),
        ),
    ]

    sizes = [1024, 32768, 1024 * 1024]

    for name, gen_fn in test_cases:
        print(f"Distribution: {name}")
        print("-" * 60)

        for size in sizes:
            x = gen_fn(size).astype(np.float32)

            # Per-block quantization
            x_pb, _ = quantize_q4_per_block(x)
            err_pb = np.abs(x[: len(x_pb)] - x_pb)

            # Global quantization
            x_gl, _ = quantize_q4_global(x)
            err_gl = np.abs(x[: len(x_gl)] - x_gl)

            # Metrics
            rmse_pb = np.sqrt(np.mean(err_pb**2))
            rmse_gl = np.sqrt(np.mean(err_gl**2))
            max_pb = err_pb.max()
            max_gl = err_gl.max()

            print(f"  Size {size:>8}: Per-block RMSE={rmse_pb:.6f}, Max={max_pb:.6f}")
            print(
                f"             Global    RMSE={rmse_gl:.6f}, Max={max_gl:.6f} ({rmse_gl / rmse_pb:.2f}x worse)"
            )

        print()

    # Now simulate the actual FFN weight distribution
    # LFM2 has SwiGLU FFN with gate, up, down projections
    print("=== Simulating LFM2-like FFN weights ===")

    # FFN dimensions
    hidden_dim = 1024
    ffn_dim = 4608

    # Generate weights like a typical transformer
    # (these would be ~Glorot initialized)
    gate_weights = np.random.randn(hidden_dim, ffn_dim).astype(np.float32) / np.sqrt(
        hidden_dim
    )
    up_weights = np.random.randn(hidden_dim, ffn_dim).astype(np.float32) / np.sqrt(
        hidden_dim
    )
    down_weights = np.random.randn(ffn_dim, hidden_dim).astype(np.float32) / np.sqrt(
        ffn_dim
    )

    print("\nFormat comparison (relative error, lower is better):")
    print(f"{'Tensor':<8} {'Per-block':<12} {'Per-row':<12} {'Global':<12}")
    print("-" * 48)

    for name, w in [("gate", gate_weights), ("up", up_weights), ("down", down_weights)]:
        x_pb, _ = quantize_q4_per_block(w)
        x_pr, _ = quantize_q4_per_row(w)
        x_gl, _ = quantize_q4_global(w)

        orig = w.flatten()[: len(x_pb)]

        rmse_pb = np.sqrt(np.mean((orig - x_pb) ** 2))
        rmse_pr = np.sqrt(np.mean((orig[: len(x_pr)] - x_pr) ** 2))
        rmse_gl = np.sqrt(np.mean((orig - x_gl) ** 2))

        # Relative error
        std = np.std(orig)
        rel_pb = rmse_pb / std
        rel_pr = rmse_pr / std
        rel_gl = rmse_gl / std

        print(f"{name:<8} {rel_pb:.4f}       {rel_pr:.4f}        {rel_gl:.4f}")

    print()
    print("Memory comparison (for gate [4608, 1024]):")
    K, N = 1024, 4608
    nb = K // 32
    perblock_bytes = N * nb * 18  # 18 bytes per block
    perrow_bytes = N * (K // 2) + N * 4  # K/2 bytes quants + 4 bytes scale per row
    global_bytes = N * (K // 2) + 4  # K/2 bytes quants + 1 scale total

    print(f"Per-block: {perblock_bytes / 1024:.0f} KB")
    print(
        f"Per-row:   {perrow_bytes / 1024:.0f} KB ({100 * (1 - perrow_bytes / perblock_bytes):.1f}% smaller)"
    )
    print(
        f"Global:    {global_bytes / 1024:.0f} KB ({100 * (1 - global_bytes / perblock_bytes):.1f}% smaller)"
    )


if __name__ == "__main__":
    main()
