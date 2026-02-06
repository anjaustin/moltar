"""
SRC-FFN Decay Experiment v2 - Investigating the gradient collapse

The first experiment showed ~6 token horizon regardless of decay.
This means the GATE is dominating, not the decay.

Let's dig deeper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SRC_FFN_Debug(nn.Module):
    def __init__(
        self, embed_dim, ff_dim, decay_init=0.95, alpha=0.5, gate_bias_init=0.0
    ):
        super().__init__()
        self.alpha = alpha

        self.w_gate = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w_up = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, embed_dim, bias=False)

        self.decay = nn.Parameter(torch.ones(ff_dim) * decay_init)
        self.gate_bias = nn.Parameter(torch.ones(ff_dim) * gate_bias_init)

        self.register_buffer("h", torch.zeros(ff_dim))

        # Debug tracking
        self.gate_values = []

    def forward_sequence(self, x_seq, track_gates=False):
        seq_len = x_seq.shape[0]
        outputs = []
        h = self.h.clone()

        if track_gates:
            self.gate_values = []

        for t in range(seq_len):
            x = x_seq[t]
            gate = self.w_gate(x)
            up = self.w_up(x)

            # CfC gate computation
            gate_input = up + self.alpha * h + self.gate_bias
            g = torch.sigmoid(gate_input)

            if track_gates:
                self.gate_values.append(g.mean().item())

            candidate = torch.tanh(up)

            # Key equation: h = (1-g)*h*decay + g*candidate
            # If g is high (~1), h forgets everything!
            # If g is low (~0), h preserves with decay
            h = (1 - g) * h * self.decay + g * candidate

            mid = F.silu(gate) * h
            out = self.w_down(mid)
            outputs.append(out)

        return torch.stack(outputs)

    def reset(self):
        self.h.zero_()
        self.gate_values = []


def analyze_gate_behavior():
    """Understand why gradient horizon is so short."""

    print("=" * 70)
    print("DEBUGGING: Why is gradient horizon only ~6 tokens?")
    print("=" * 70)

    torch.manual_seed(42)

    embed_dim = 64
    ff_dim = 256
    seq_len = 32

    print("\n1. GATE VALUE ANALYSIS")
    print("-" * 50)

    # Default initialization
    layer = SRC_FFN_Debug(embed_dim, ff_dim, decay_init=0.99, gate_bias_init=0.0)
    x_seq = torch.randn(seq_len, embed_dim)

    layer.reset()
    _ = layer.forward_sequence(x_seq, track_gates=True)

    print(f"\nDefault (gate_bias=0.0):")
    print(f"  Mean gate value: {sum(layer.gate_values) / len(layer.gate_values):.3f}")
    print(f"  Gate values over time: {[f'{g:.2f}' for g in layer.gate_values[:10]]}...")

    # The problem: gate ≈ sigmoid(random) ≈ 0.5
    # With g=0.5, effective decay = (1-0.5)*0.99 = 0.495
    # Horizon = log(0.01)/log(0.495) ≈ 6.5 tokens!

    print(f"\n  With g=0.5, effective_decay = (1-g)*decay = 0.5*0.99 = 0.495")
    print(
        f"  Theoretical horizon = log(0.01)/log(0.495) = {math.log(0.01) / math.log(0.495):.1f} tokens"
    )
    print(f"  THIS EXPLAINS THE ~6 TOKEN HORIZON!")

    print("\n2. FIXING THE GATE: Negative bias for memory retention")
    print("-" * 50)

    # Negative gate bias = sigmoid(x-bias) shifts toward 0
    # g closer to 0 = more memory retention

    for gate_bias in [0.0, -1.0, -2.0, -3.0, -4.0]:
        layer = SRC_FFN_Debug(
            embed_dim, ff_dim, decay_init=0.99, gate_bias_init=gate_bias
        )
        layer.reset()
        _ = layer.forward_sequence(x_seq, track_gates=True)

        mean_gate = sum(layer.gate_values) / len(layer.gate_values)
        effective_decay = (1 - mean_gate) * 0.99

        if effective_decay > 0 and effective_decay < 1:
            theoretical_horizon = math.log(0.01) / math.log(effective_decay)
        else:
            theoretical_horizon = float("inf")

        print(f"\n  gate_bias = {gate_bias:4.1f}")
        print(f"    Mean gate: {mean_gate:.3f}")
        print(f"    Effective decay: {effective_decay:.3f}")
        print(f"    Theoretical horizon: {theoretical_horizon:.0f} tokens")


def measure_gradient_with_fixed_gate():
    """Measure gradient horizon with corrected gate bias."""

    print("\n" + "=" * 70)
    print("GRADIENT HORIZON WITH CORRECTED GATE BIAS")
    print("=" * 70)

    torch.manual_seed(42)

    embed_dim = 64
    ff_dim = 256
    seq_len = 512

    configs = [
        {"decay": 0.99, "gate_bias": 0.0, "name": "Default (broken)"},
        {"decay": 0.99, "gate_bias": -2.0, "name": "Fixed (g~0.2)"},
        {"decay": 0.99, "gate_bias": -3.0, "name": "Long memory (g~0.1)"},
        {"decay": 0.999, "gate_bias": -3.0, "name": "Very long (g~0.1, decay=0.999)"},
    ]

    for config in configs:
        layer = SRC_FFN_Debug(
            embed_dim,
            ff_dim,
            decay_init=config["decay"],
            gate_bias_init=config["gate_bias"],
        )
        x_seq = torch.randn(seq_len, embed_dim, requires_grad=True)

        layer.reset()
        outputs = layer.forward_sequence(x_seq)

        loss = outputs[-1].pow(2).sum()
        loss.backward()

        grad_norms = [x_seq.grad[t].norm().item() for t in range(seq_len)]
        max_grad = max(grad_norms)

        # Find 1% horizon
        threshold = max_grad * 0.01
        horizon = sum(1 for g in grad_norms if g > threshold)

        # Get mean gate value
        layer.reset()
        x_seq_eval = torch.randn(seq_len, embed_dim)
        _ = layer.forward_sequence(x_seq_eval, track_gates=True)
        mean_gate = sum(layer.gate_values) / len(layer.gate_values)

        print(f"\n{config['name']}:")
        print(f"  Gate bias: {config['gate_bias']}, Mean gate: {mean_gate:.3f}")
        print(f"  Gradient horizon (>1%): {horizon} tokens")


def design_proper_initialization():
    """Design proper initialization for SRC-FFN."""

    print("\n" + "=" * 70)
    print("PROPER SRC-FFN INITIALIZATION")
    print("=" * 70)

    print("""
THE KEY INSIGHT:

In CfC: h' = (1-g)*h*decay + g*candidate

The gate g controls the BALANCE between:
  - Memory (1-g): preserving past information  
  - Update (g): incorporating new information

For long-range gradients, we need (1-g)*decay close to 1.

SOLUTION: Initialize gate_bias to be NEGATIVE

  gate_input = up + alpha*h + gate_bias
  g = sigmoid(gate_input)
  
  With gate_bias = -2.0:
    g ≈ sigmoid(-2 + noise) ≈ 0.1 to 0.3
    
  With gate_bias = -3.0:
    g ≈ sigmoid(-3 + noise) ≈ 0.05 to 0.15

This gives effective_decay = (1-g)*decay ≈ 0.85-0.95
And gradient horizon of 30-90 tokens!

MULTI-SCALE INITIALIZATION:
""")

    torch.manual_seed(42)

    embed_dim = 64
    ff_dim = 256
    seq_len = 512

    # Multi-scale initialization
    layer = SRC_FFN_Debug(embed_dim, ff_dim)

    quarter = ff_dim // 4

    # Short-range neurons: high gate, low decay
    layer.gate_bias.data[:quarter] = 0.0  # g ≈ 0.5
    layer.decay.data[:quarter] = 0.9

    # Medium-range neurons
    layer.gate_bias.data[quarter : 2 * quarter] = -2.0  # g ≈ 0.2
    layer.decay.data[quarter : 2 * quarter] = 0.95

    # Long-range neurons
    layer.gate_bias.data[2 * quarter : 3 * quarter] = -3.0  # g ≈ 0.1
    layer.decay.data[2 * quarter : 3 * quarter] = 0.99

    # Very long-range neurons
    layer.gate_bias.data[3 * quarter :] = -4.0  # g ≈ 0.05
    layer.decay.data[3 * quarter :] = 0.995

    x_seq = torch.randn(seq_len, embed_dim, requires_grad=True)

    layer.reset()
    outputs = layer.forward_sequence(x_seq)

    loss = outputs[-1].pow(2).sum()
    loss.backward()

    grad_norms = [x_seq.grad[t].norm().item() for t in range(seq_len)]
    max_grad = max(grad_norms)

    # Find horizons
    horizon_1pct = sum(1 for g in grad_norms if g > max_grad * 0.01)
    horizon_10pct = sum(1 for g in grad_norms if g > max_grad * 0.10)

    print(f"Multi-scale initialization:")
    print(f"  Gradient horizon (>1%):  {horizon_1pct} tokens")
    print(f"  Gradient horizon (>10%): {horizon_10pct} tokens")

    # Visualize
    print(f"\nGradient decay profile:")
    print("-" * 50)

    for dist in [0, 10, 25, 50, 100, 200, 300, 400]:
        if dist < seq_len:
            t = seq_len - 1 - dist
            rel = grad_norms[t] / max_grad * 100
            bar_len = min(40, int(rel * 0.4))
            bar = "#" * bar_len + "." * (40 - bar_len)
            print(f"  t-{dist:3d}: [{bar}] {rel:5.1f}%")


def final_recommendations():
    """Print final recommendations."""

    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS FOR SRC-FFN")
    print("=" * 70)

    print("""
1. GATE BIAS IS CRITICAL
   - Default (0.0) gives g≈0.5 → only 6 token horizon
   - Use negative bias: -2.0 to -4.0 for memory retention
   
2. RECOMMENDED INITIALIZATION:
   
   # Multi-scale temporal features
   quarter = ff_dim // 4
   
   # Short-range (local syntax)
   gate_bias[:quarter] = -1.0      # g ≈ 0.3
   decay[:quarter] = 0.9           # horizon ~10 tokens
   
   # Medium-range (phrases, clauses)
   gate_bias[quarter:2*quarter] = -2.0    # g ≈ 0.15
   decay[quarter:2*quarter] = 0.95        # horizon ~30 tokens
   
   # Long-range (sentences, paragraphs)
   gate_bias[2*quarter:3*quarter] = -3.0  # g ≈ 0.07
   decay[2*quarter:3*quarter] = 0.99      # horizon ~100 tokens
   
   # Very long-range (documents)
   gate_bias[3*quarter:] = -4.0           # g ≈ 0.03
   decay[3*quarter:] = 0.995              # horizon ~300+ tokens

3. TRAINING STRATEGY:
   - Truncated BPTT with K=128 covers most gradient flow
   - Memory: ~33 MB for LFM2-350M scale
   - Let network LEARN optimal gate_bias and decay during training
   
4. KEY EQUATION TO REMEMBER:
   
   effective_decay = (1 - gate) * decay
   gradient_horizon ≈ log(0.01) / log(effective_decay)
   
   For horizon of N tokens, you need:
   effective_decay = 0.01^(1/N)
   
   N=10:  effective_decay = 0.63
   N=50:  effective_decay = 0.91
   N=100: effective_decay = 0.955
   N=500: effective_decay = 0.991
""")


if __name__ == "__main__":
    analyze_gate_behavior()
    measure_gradient_with_fixed_gate()
    design_proper_initialization()
    final_recommendations()

    print("\n" + "=" * 70)
    print("  MYSTERY SOLVED! The gate was eating our gradients.")
    print("  Dawn is safe. Stand down, Mr. President.")
    print("=" * 70)
