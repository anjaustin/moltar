"""
Hybrid Evolving Language Model

Weights trained with gradient descent, dynamics evolved.
This should produce coherent output while still allowing temporal hierarchy to emerge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional

from hybrid_layer import HybridLayer, HybridConfig


@dataclass
class HybridModelConfig:
    """Configuration for hybrid model."""

    vocab_size: int = 50257  # GPT-2
    embed_dim: int = 128
    n_layers: int = 8
    n_neurons_per_layer: int = 256

    # Evolution
    evolve_every: int = 100
    selection_pressure: float = 0.5
    elite_fraction: float = 0.1
    mutation_rate: float = 0.5
    mutation_strength: float = 0.2

    # CfC dynamics
    alpha: float = 0.5
    decay_min: float = 0.8
    decay_max: float = 0.999
    gate_bias_min: float = -4.0
    gate_bias_max: float = 2.0


class HybridModel(nn.Module):
    """
    Hybrid evolving language model.

    Architecture:
    - Token embedding (gradient trained)
    - N hybrid layers (weights: gradient, dynamics: evolution)
    - Output projection (tied to embedding)
    """

    def __init__(self, config: HybridModelConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

        # Hybrid layers
        layer_config = HybridConfig(
            n_neurons=config.n_neurons_per_layer,
            embed_dim=config.embed_dim,
            alpha=config.alpha,
            decay_min=config.decay_min,
            decay_max=config.decay_max,
            gate_bias_min=config.gate_bias_min,
            gate_bias_max=config.gate_bias_max,
        )

        self.layers = nn.ModuleList(
            [HybridLayer(layer_config, layer_idx=i) for i in range(config.n_layers)]
        )

        # Output
        self.out_norm = nn.LayerNorm(config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight  # Tie weights

        # State
        self.tokens_seen = 0
        self.total_evolutions = 0

        # Trajectory tracking
        self.trajectory = {
            "generations": [],
            "tokens": [],
            "decay_spread": [],
            "gate_spread": [],
            "per_layer_decays": [],
            "per_layer_gates": [],
            "mean_fitness": [],
        }
        self.track_trajectory = False

    def reset_state(self):
        """Reset all hidden states."""
        for layer in self.layers:
            layer.reset_state()

    def set_trajectory_tracking(self, enabled: bool = True):
        """Enable/disable trajectory tracking."""
        self.track_trajectory = enabled
        if enabled:
            self.trajectory = {
                "generations": [],
                "tokens": [],
                "decay_spread": [],
                "gate_spread": [],
                "per_layer_decays": [],
                "per_layer_gates": [],
                "mean_fitness": [],
            }

    def _record_trajectory(self, pre_evolve_fitness: List[float] = None):
        """Record current state to trajectory."""
        if not self.track_trajectory:
            return

        decays = []
        gates = []

        for layer in self.layers:
            stats = layer.get_stats()
            decays.append(stats["mean_decay"])
            gates.append(stats["mean_gate_bias"])

        self.trajectory["generations"].append(self.total_evolutions)
        self.trajectory["tokens"].append(self.tokens_seen)
        self.trajectory["decay_spread"].append(max(decays) - min(decays))
        self.trajectory["gate_spread"].append(max(gates) - min(gates))
        self.trajectory["per_layer_decays"].append(decays[:])
        self.trajectory["per_layer_gates"].append(gates[:])

        # Use pre-evolve fitness if provided
        if pre_evolve_fitness:
            self.trajectory["mean_fitness"].append(
                sum(pre_evolve_fitness) / len(pre_evolve_fitness)
            )
        else:
            self.trajectory["mean_fitness"].append(0.0)

    def forward(self, token_id: int) -> torch.Tensor:
        """
        Forward pass for a single token.

        Returns logits [vocab_size].
        """
        # Get embedding
        x = self.embed.weight[token_id]  # [embed_dim]

        # Pass through layers with residual
        for layer in self.layers:
            out = layer(x)
            x = x + out

        # Output
        x = self.out_norm(x)
        logits = self.out_proj(x)

        self.tokens_seen += 1
        return logits

    def maybe_evolve(self):
        """Evolve if enough tokens have been processed."""
        if self.tokens_seen > 0 and self.tokens_seen % self.config.evolve_every == 0:
            self.evolve_all_layers()

    def evolve_all_layers(self):
        """Run evolution on all layers."""
        # Record fitness BEFORE evolution (since evolve resets fitness)
        pre_evolve_fitness = []
        for layer in self.layers:
            layer.population.finalize_fitness()
            pre_evolve_fitness.append(layer.population.fitness.mean().item())

        for layer in self.layers:
            layer.evolve(
                selection_pressure=self.config.selection_pressure,
                elite_fraction=self.config.elite_fraction,
                mutation_rate=self.config.mutation_rate,
                mutation_strength=self.config.mutation_strength,
            )

        self.total_evolutions += 1
        self._record_trajectory(pre_evolve_fitness)

    def get_hierarchy_summary(self) -> Dict:
        """Get summary of hierarchy across layers."""
        decays = []
        gates = []

        for layer in self.layers:
            stats = layer.get_stats()
            decays.append(stats["mean_decay"])
            gates.append(stats["mean_gate_bias"])

        return {
            "decays_by_layer": decays,
            "gate_biases_by_layer": gates,
            "decay_spread": max(decays) - min(decays),
            "gate_spread": max(gates) - min(gates),
        }

    def get_all_stats(self) -> List[Dict]:
        """Get stats for all layers."""
        return [layer.get_stats() for layer in self.layers]

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "config": self.config,
                "model_state": self.state_dict(),
                "tokens_seen": self.tokens_seen,
                "total_evolutions": self.total_evolutions,
                "trajectory": self.trajectory,
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str) -> "HybridModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        model.tokens_seen = checkpoint["tokens_seen"]
        model.total_evolutions = checkpoint["total_evolutions"]
        model.trajectory = checkpoint.get("trajectory", model.trajectory)
        return model


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing HybridModel...")

    config = HybridModelConfig(
        vocab_size=1000,
        embed_dim=64,
        n_layers=4,
        n_neurons_per_layer=32,
        evolve_every=50,
    )

    model = HybridModel(config)

    print(f"Model created:")
    print(f"  Layers: {config.n_layers}")
    print(f"  Neurons/layer: {config.n_neurons_per_layer}")
    print(f"  Total neurons: {config.n_layers * config.n_neurons_per_layer}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable: {trainable:,}")

    # Test forward and backward
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nTraining for 200 tokens...")
    model.set_trajectory_tracking(True)

    total_loss = 0
    correct = 0

    for i in range(200):
        token_id = i % 1000
        target_id = (i + 1) % 1000

        logits = model.forward(token_id)
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if logits.argmax().item() == target_id:
            correct += 1

        model.maybe_evolve()

    print(f"  Avg loss: {total_loss / 200:.3f}")
    print(f"  Accuracy: {correct / 200 * 100:.1f}%")
    print(f"  Evolutions: {model.total_evolutions}")

    # Check hierarchy
    hierarchy = model.get_hierarchy_summary()
    print(f"\nHierarchy:")
    print(f"  Decays: {[f'{d:.3f}' for d in hierarchy['decays_by_layer']]}")
    print(f"  Gates: {[f'{g:.2f}' for g in hierarchy['gate_biases_by_layer']]}")
    print(f"  Decay spread: {hierarchy['decay_spread']:.3f}")

    # Check trajectory
    print(f"\nTrajectory points: {len(model.trajectory['generations'])}")
    if model.trajectory["decay_spread"]:
        print(
            f"  Decay spread trend: {model.trajectory['decay_spread'][0]:.3f} -> {model.trajectory['decay_spread'][-1]:.3f}"
        )

    print("\nHybridModel test passed!")
