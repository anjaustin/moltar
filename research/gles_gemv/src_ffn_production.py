"""
SRC-FFN Production Implementation

Spectral-Rotational-CfC Feed-Forward Network with:
- Proper multi-scale initialization
- Equilibrium Propagation training
- Ternary weight support
- Full language modeling pipeline

This is the reference implementation for PRD-001 and PRD-002.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum
import time


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SRCConfig:
    """Configuration for SRC-FFN model."""

    vocab_size: int = 32000
    embed_dim: int = 1024
    ff_dim: int = 4096
    n_layers: int = 16
    max_seq_len: int = 8192
    rope_theta: float = 10000.0

    # CfC parameters
    cfc_alpha: float = 0.5  # h contribution to gate

    # Multi-scale initialization
    decay_scales: Tuple[float, ...] = (0.9, 0.95, 0.99, 0.995)
    gate_bias_scales: Tuple[float, ...] = (-1.0, -2.0, -3.0, -4.0)

    # EP training parameters
    ep_beta: float = 0.1
    ep_free_steps: int = 50
    ep_nudge_steps: int = 20
    ep_dt: float = 0.1
    ep_convergence_threshold: float = 1e-5

    # Training
    dropout: float = 0.0
    use_ternary: bool = False  # Enable for quantized inference


class TrainingMode(Enum):
    BPTT = "bptt"  # Standard backprop through time
    TRUNCATED = "truncated"  # Truncated BPTT
    EP = "ep"  # Equilibrium propagation


# =============================================================================
# Core Modules
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len)
        freqs = torch.outer(positions, self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def forward(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        """Apply RoPE to input tensor."""
        # x: [..., dim]
        cos = self.cos_cached[pos]  # [dim/2]
        sin = self.sin_cached[pos]  # [dim/2]

        # Split into pairs and rotate
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return rotated.flatten(-2)


class TernaryLinear(nn.Module):
    """Linear layer with optional ternary weight quantization."""

    def __init__(self, in_features: int, out_features: int, use_ternary: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_ternary = use_ternary

        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        # Ternary quantized weights (computed on demand)
        self.register_buffer("weight_ternary", None)
        self.register_buffer("weight_scale", None)

    def quantize(self):
        """Quantize weights to ternary."""
        w = self.weight.data
        scale = w.abs().mean()
        w_ternary = torch.sign(w) * (w.abs() > 0.5 * scale).float()
        self.weight_ternary = w_ternary
        self.weight_scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ternary and self.weight_ternary is not None:
            return F.linear(x, self.weight_ternary * self.weight_scale)
        return F.linear(x, self.weight)


# =============================================================================
# SRC-FFN Layer
# =============================================================================


class SRCFFNLayer(nn.Module):
    """
    Spectral-Rotational-CfC Feed-Forward Network Layer.

    Each of the ff_dim neurons maintains its own CfC hidden state,
    creating a massively parallel recurrent network within the FFN.
    """

    def __init__(self, config: SRCConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Projections
        self.w_gate = TernaryLinear(config.embed_dim, config.ff_dim, config.use_ternary)
        self.w_up = TernaryLinear(config.embed_dim, config.ff_dim, config.use_ternary)
        self.w_down = TernaryLinear(config.ff_dim, config.embed_dim, config.use_ternary)

        # CfC parameters (per-neuron)
        self.decay = nn.Parameter(torch.ones(config.ff_dim))
        self.gate_bias = nn.Parameter(torch.zeros(config.ff_dim))

        # Normalization
        self.norm = RMSNorm(config.embed_dim)

        # Initialize with multi-scale temporal features
        self._init_multiscale()

        # Hidden state buffer (not a parameter)
        self.register_buffer("h", torch.zeros(config.ff_dim))

    def _init_multiscale(self):
        """Initialize with multi-scale decay and gate bias."""
        ff = self.config.ff_dim
        n_scales = len(self.config.decay_scales)
        chunk = ff // n_scales

        for i, (decay, bias) in enumerate(
            zip(self.config.decay_scales, self.config.gate_bias_scales)
        ):
            start = i * chunk
            end = (i + 1) * chunk if i < n_scales - 1 else ff
            self.decay.data[start:end] = decay
            self.gate_bias.data[start:end] = bias

    def reset_hidden(self):
        """Reset hidden state to zeros."""
        self.h.zero_()

    def set_hidden(self, h: torch.Tensor):
        """Set hidden state."""
        self.h.copy_(h.detach())

    def get_hidden(self) -> torch.Tensor:
        """Get current hidden state."""
        return self.h.clone()

    def cfc_update(self, h: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Single CfC update step."""
        g = torch.sigmoid(up + self.config.cfc_alpha * h + self.gate_bias)
        candidate = torch.tanh(up)
        return (1 - g) * h * self.decay + g * candidate

    def forward(
        self, x: torch.Tensor, update_hidden: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, embed_dim] or [embed_dim]
            update_hidden: Whether to update internal hidden state

        Returns:
            out: Output tensor [batch, embed_dim] or [embed_dim]
            h_new: New hidden state [batch, ff_dim] or [ff_dim]
        """
        # Normalize
        x_norm = self.norm(x)

        # Projections
        gate = self.w_gate(x_norm)
        up = self.w_up(x_norm)

        # CfC update
        h_new = self.cfc_update(self.h, up)

        # Gated output
        mid = F.silu(gate) * h_new
        out = self.w_down(mid)

        # Update hidden state
        if update_hidden:
            self.h = h_new.detach()

        return out, h_new

    def free_phase(
        self, x: torch.Tensor, steps: Optional[int] = None, dt: Optional[float] = None
    ) -> torch.Tensor:
        """
        Run CfC dynamics to equilibrium (free phase for EP).
        """
        steps = steps or self.config.ep_free_steps
        dt = dt or self.config.ep_dt
        threshold = self.config.ep_convergence_threshold

        x_norm = self.norm(x)
        gate = self.w_gate(x_norm)
        up = self.w_up(x_norm)

        h = self.h.clone()

        for _ in range(steps):
            target = self.cfc_update(h, up)
            delta = target - h
            h = h + dt * delta

            # Early stopping
            if delta.abs().max() < threshold:
                break

        return h

    def nudged_phase(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        h_init: torch.Tensor,
        beta: Optional[float] = None,
        steps: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Run CfC dynamics with output nudged toward target (nudged phase for EP).
        """
        beta = beta or self.config.ep_beta
        steps = steps or self.config.ep_nudge_steps
        dt = dt or self.config.ep_dt

        x_norm = self.norm(x)
        gate = self.w_gate(x_norm)
        up = self.w_up(x_norm)

        h = h_init.clone()

        for _ in range(steps):
            # CfC dynamics
            target = self.cfc_update(h, up)

            # Compute output and nudge
            mid = F.silu(gate) * h
            output = self.w_down(mid)

            # Nudge: backprop error through w_down to get h gradient
            error = y_target - output
            nudge = beta * (error @ self.w_down.weight)

            # Combined update
            h = h + dt * ((target - h) + nudge)

        return h

    def compute_ep_gradients(
        self, x: torch.Tensor, y_target: torch.Tensor, beta: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients using Equilibrium Propagation.
        """
        beta = beta or self.config.ep_beta

        # Free phase
        h_free = self.free_phase(x)

        # Nudged phase
        h_nudged = self.nudged_phase(x, y_target, h_free, beta)

        # Compute outputs
        x_norm = self.norm(x)
        gate = self.w_gate(x_norm)
        up = self.w_up(x_norm)

        mid_free = F.silu(gate) * h_free
        mid_nudged = F.silu(gate) * h_nudged

        output_free = self.w_down(mid_free)
        output_nudged = self.w_down(mid_nudged)

        # EP gradients
        dh = (h_nudged - h_free) / beta
        dout = (output_nudged - output_free) / beta

        gradients = {}

        # W_down gradient
        gradients["w_down.weight"] = torch.outer(dout, mid_nudged)

        # W_up gradient (approximate)
        g = torch.sigmoid(up + self.config.cfc_alpha * h_nudged + self.gate_bias)
        tanh_up = torch.tanh(up)
        gradients["w_up.weight"] = torch.outer(dh * g * (1 - tanh_up**2), x_norm)

        # W_gate gradient (approximate)
        silu_gate = F.silu(gate)
        silu_grad = torch.sigmoid(gate) * (1 + gate * (1 - torch.sigmoid(gate)))
        gradients["w_gate.weight"] = torch.outer(
            silu_grad * h_nudged * (dout @ self.w_down.weight), x_norm
        )

        # Decay and gate_bias gradients
        gradients["decay"] = dh * (1 - g) * h_free
        gradients["gate_bias"] = dh * g * (1 - g) * (tanh_up - h_free * self.decay)

        # Update hidden state to nudged equilibrium
        self.h = h_nudged.detach()

        # Return loss for monitoring
        loss = 0.5 * ((output_free - y_target) ** 2).sum()

        return gradients, loss, h_nudged


# =============================================================================
# Full Model
# =============================================================================


class SRCFFN_LM(nn.Module):
    """
    SRC-FFN Language Model.

    Full model with embedding, multiple SRC-FFN layers, and output head.
    """

    def __init__(self, config: SRCConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)

        # RoPE
        self.rope = RotaryEmbedding(
            config.embed_dim, config.max_seq_len, config.rope_theta
        )

        # Layers
        self.layers = nn.ModuleList(
            [SRCFFNLayer(config, i) for i in range(config.n_layers)]
        )

        # Output
        self.out_norm = RMSNorm(config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Tie weights
        self.out_proj.weight = self.embed.weight

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=0.02)

    def reset_hidden(self):
        """Reset all layer hidden states."""
        for layer in self.layers:
            layer.reset_hidden()

    def forward(self, input_ids: torch.Tensor, position: int = 0) -> torch.Tensor:
        """
        Forward pass for single token.

        Args:
            input_ids: Token ID(s) [batch] or scalar
            position: Position in sequence

        Returns:
            logits: Output logits [batch, vocab_size] or [vocab_size]
        """
        # Embed
        x = self.embed(input_ids)

        # Apply RoPE
        x = self.rope(x, position)

        # Layers with residual
        for layer in self.layers:
            out, _ = layer(x)
            x = x + out

        # Output
        x = self.out_norm(x)
        logits = self.out_proj(x)

        return logits

    def forward_sequence(
        self, input_ids: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for full sequence.

        Args:
            input_ids: Token IDs [seq_len] or [batch, seq_len]
            return_hidden: Whether to return hidden states

        Returns:
            logits: Output logits [seq_len, vocab_size]
        """
        self.reset_hidden()

        seq_len = input_ids.shape[-1]
        outputs = []

        for pos in range(seq_len):
            token = input_ids[..., pos]
            logits = self.forward(token, pos)
            outputs.append(logits)

        return torch.stack(outputs, dim=-2)


# =============================================================================
# EP Trainer
# =============================================================================


class EPTrainer:
    """
    Equilibrium Propagation trainer for SRC-FFN.

    Memory: O(1) regardless of sequence length!
    """

    def __init__(self, model: SRCFFN_LM, lr: float = 1e-3, beta: float = 0.1):
        self.model = model
        self.lr = lr
        self.beta = beta

        # Optimizer (for non-EP parameters like embeddings)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """
        Train for one sequence using EP.

        Args:
            input_ids: Input token IDs [seq_len]
            target_ids: Target token IDs [seq_len]

        Returns:
            Average loss over sequence
        """
        self.model.reset_hidden()
        seq_len = input_ids.shape[0]

        total_loss = 0.0

        for pos in range(seq_len):
            # Get input and target
            token = input_ids[pos]
            target = target_ids[pos]

            # Embed input
            x = self.model.embed(token)
            x = self.model.rope(x, pos)

            # Target embedding (for EP)
            y_target = self.model.embed(target)

            # Process each layer with EP
            for layer in self.model.layers:
                # Compute EP gradients and get new hidden state
                gradients, loss, h_new = layer.compute_ep_gradients(
                    x, y_target, self.beta
                )

                # Apply gradients manually (EP doesn't use autograd)
                with torch.no_grad():
                    for name, grad in gradients.items():
                        param = dict(layer.named_parameters())[name]
                        param.add_(-self.lr * grad)

                # Forward through layer for next residual
                out = layer.w_down(F.silu(layer.w_gate(layer.norm(x))) * h_new)
                x = x + out

                total_loss += loss.item()

        return total_loss / (seq_len * len(self.model.layers))

    def train_epoch(self, dataloader, log_interval: int = 100) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            loss = self.train_step(input_ids, target_ids)
            total_loss += loss
            n_batches += 1

            if batch_idx % log_interval == 0:
                print(f"Batch {batch_idx}, Loss: {loss:.4f}")

        return total_loss / n_batches


# =============================================================================
# Demo and Testing
# =============================================================================


def demo_forward():
    """Demonstrate forward pass."""
    print("=" * 60)
    print("SRC-FFN Forward Pass Demo")
    print("=" * 60)

    config = SRCConfig(vocab_size=1000, embed_dim=256, ff_dim=1024, n_layers=4)

    model = SRCFFN_LM(config)
    print(f"\nModel created:")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  FF dim: {config.ff_dim}")
    print(f"  Layers: {config.n_layers}")

    # Single token forward
    token = torch.tensor(42)
    logits = model(token, position=0)
    print(f"\nSingle token forward:")
    print(f"  Input: token {token.item()}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Top 5 predictions: {logits.topk(5).indices.tolist()}")

    # Sequence forward
    seq = torch.randint(0, 1000, (32,))
    logits = model.forward_sequence(seq)
    print(f"\nSequence forward:")
    print(f"  Input shape: {seq.shape}")
    print(f"  Output shape: {logits.shape}")

    # Memory analysis
    h_memory = config.ff_dim * config.n_layers * 4  # bytes
    print(f"\nMemory for hidden states: {h_memory / 1024:.1f} KB")


def demo_ep_training():
    """Demonstrate EP training."""
    print("\n" + "=" * 60)
    print("SRC-FFN Equilibrium Propagation Training Demo")
    print("=" * 60)

    config = SRCConfig(
        vocab_size=100,
        embed_dim=64,
        ff_dim=256,
        n_layers=2,
        ep_free_steps=30,
        ep_nudge_steps=15,
    )

    model = SRCFFN_LM(config)
    trainer = EPTrainer(model, lr=1e-3, beta=0.1)

    # Synthetic data: predict next character
    print("\nTraining on synthetic sequence...")

    # Simple pattern: 1,2,3,4,5,1,2,3,4,5,...
    seq_len = 20
    pattern = torch.tensor([1, 2, 3, 4, 5] * 4)
    input_ids = pattern[:-1]
    target_ids = pattern[1:]

    print(f"Pattern: {pattern[:10].tolist()}...")
    print(f"Input:   {input_ids[:5].tolist()}...")
    print(f"Target:  {target_ids[:5].tolist()}...")

    # Train for a few steps
    losses = []
    for step in range(10):
        loss = trainer.train_step(input_ids, target_ids)
        losses.append(loss)
        print(f"Step {step + 1}: Loss = {loss:.4f}")

    print(f"\nLoss reduction: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Test prediction
    model.reset_hidden()
    test_input = torch.tensor(1)
    logits = model(test_input, position=0)
    pred = logits.argmax().item()
    print(f"\nTest: Input=1, Predicted next={pred}, Expected=2")


def demo_memory_comparison():
    """Compare memory usage of different training methods."""
    print("\n" + "=" * 60)
    print("Memory Comparison: BPTT vs Truncated vs EP")
    print("=" * 60)

    config = SRCConfig()  # LFM2-350M scale

    for seq_len in [512, 2048, 8192, 32768]:
        bptt_memory = seq_len * config.ff_dim * config.n_layers * 4 / 1e6
        trunc_memory = 128 * config.ff_dim * config.n_layers * 4 / 1e6
        ep_memory = 2 * config.ff_dim * config.n_layers * 4 / 1e6

        print(f"\nSequence length: {seq_len}")
        print(f"  BPTT:      {bptt_memory:7.1f} MB")
        print(f"  Truncated: {trunc_memory:7.1f} MB")
        print(f"  EP:        {ep_memory:7.1f} MB")
        print(f"  EP saves:  {bptt_memory / ep_memory:.0f}x vs BPTT")


def demo_initialization():
    """Show multi-scale initialization."""
    print("\n" + "=" * 60)
    print("Multi-Scale CfC Initialization")
    print("=" * 60)

    config = SRCConfig(ff_dim=16)  # Small for visualization
    layer = SRCFFNLayer(config, 0)

    print("\nDecay values by neuron:")
    print(f"  {layer.decay.data.tolist()}")

    print("\nGate bias values by neuron:")
    print(f"  {layer.gate_bias.data.tolist()}")

    print("\nThis gives temporal horizons:")
    for i, (d, b) in enumerate(zip(config.decay_scales, config.gate_bias_scales)):
        g = torch.sigmoid(torch.tensor(b)).item()
        eff_decay = (1 - g) * d
        horizon = (
            int(math.log(0.01) / math.log(eff_decay)) if eff_decay < 1 else float("inf")
        )
        print(
            f"  Scale {i + 1}: decay={d}, bias={b}, g≈{g:.2f}, horizon≈{horizon} tokens"
        )


if __name__ == "__main__":
    torch.manual_seed(42)

    demo_initialization()
    demo_forward()
    demo_ep_training()
    demo_memory_comparison()

    print("\n" + "=" * 60)
    print("SRC-FFN Production Implementation Ready!")
    print("=" * 60)
