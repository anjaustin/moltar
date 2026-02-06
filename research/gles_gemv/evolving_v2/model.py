"""
Evolving SRC-FFN Model - Full Assembly

8 layers x 512 neurons = 4,096 evolving neurons
Embeddings and output projection trained with gradient descent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from dataclasses import dataclass

from vectorized_population import VectorizedLayer, PopulationConfig


@dataclass
class ModelConfig:
    """Configuration for the full model."""

    vocab_size: int = 50257  # GPT-2 tokenizer
    embed_dim: int = 256
    n_layers: int = 8
    n_neurons_per_layer: int = 512

    # Evolution settings
    evolve_every: int = 100  # Tokens between evolution steps
    selection_pressure: float = 0.5
    elite_fraction: float = 0.1
    mutation_rate: float = 0.3
    mutation_strength: float = 0.1
    sexual_rate: float = 0.3

    # Diversity maintenance
    min_diversity: float = 0.05
    diversity_injection_rate: float = 0.1

    # CfC settings
    alpha: float = 0.5
    decay_min: float = 0.8
    decay_max: float = 0.999
    gate_bias_min: float = -4.0
    gate_bias_max: float = 0.0


class EvolvingModel(nn.Module):
    """
    Full evolving SRC-FFN language model.

    Architecture:
    - Token embedding (gradient trained)
    - N evolving SRC-FFN layers
    - Output projection (tied to embedding)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding (gradient trained)
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

        # Evolving layers
        pop_config = PopulationConfig(
            n_neurons=config.n_neurons_per_layer,
            embed_dim=config.embed_dim,
            alpha=config.alpha,
            decay_min=config.decay_min,
            decay_max=config.decay_max,
            gate_bias_min=config.gate_bias_min,
            gate_bias_max=config.gate_bias_max,
        )

        self.layers = nn.ModuleList(
            [VectorizedLayer(pop_config, layer_idx=i) for i in range(config.n_layers)]
        )

        # Output normalization and projection
        self.out_norm = nn.LayerNorm(config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Tie output projection to embedding
        self.out_proj.weight = self.embed.weight

        # Training state
        self.tokens_seen = 0
        self.total_evolutions = 0

        # Trajectory tracking (for Delta Observer-style analysis)
        self.trajectory = {
            "generations": [],  # Evolution count at each snapshot
            "tokens": [],  # Tokens seen at each snapshot
            "decay_spread": [],  # Hierarchy strength (max-min decay)
            "gate_spread": [],  # Gate diversity
            "per_layer_decays": [],  # Full decay per layer
            "per_layer_gates": [],  # Full gate bias per layer
            "diversity": [],  # Mean diversity across layers
        }
        self.track_trajectory = False  # Enable via set_trajectory_tracking()

    def reset_state(self):
        """Reset all hidden states (call between sequences)."""
        for layer in self.layers:
            layer.reset_state()

    def set_trajectory_tracking(self, enabled: bool = True):
        """Enable/disable trajectory tracking for Delta Observer-style analysis."""
        self.track_trajectory = enabled
        if enabled:
            # Clear previous trajectory
            self.trajectory = {
                "generations": [],
                "tokens": [],
                "decay_spread": [],
                "gate_spread": [],
                "per_layer_decays": [],
                "per_layer_gates": [],
                "diversity": [],
            }

    def _record_trajectory_point(self):
        """Record current hierarchy state to trajectory."""
        if not self.track_trajectory:
            return

        decays = []
        gates = []
        diversities = []

        for layer in self.layers:
            stats = layer.get_stats()
            decays.append(stats["mean_decay"])
            gates.append(stats["mean_gate_bias"])
            diversities.append(stats["std_decay"])

        self.trajectory["generations"].append(self.total_evolutions)
        self.trajectory["tokens"].append(self.tokens_seen)
        self.trajectory["decay_spread"].append(max(decays) - min(decays))
        self.trajectory["gate_spread"].append(max(gates) - min(gates))
        self.trajectory["per_layer_decays"].append(decays[:])
        self.trajectory["per_layer_gates"].append(gates[:])
        self.trajectory["diversity"].append(sum(diversities) / len(diversities))

    def forward(self, token_id: int, target_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass for a single token.

        Args:
            token_id: Input token ID
            target_id: Target token ID for fitness tracking (optional)

        Returns:
            logits: [vocab_size]
        """
        # Get embeddings
        x = self.embed.weight[token_id]  # [embed_dim]
        target_embed = self.embed.weight[target_id] if target_id is not None else None

        # Pass through evolving layers with residual connections
        for layer in self.layers:
            out = layer(x, target_embed=target_embed)
            x = x + out  # Residual

        # Output projection
        x = self.out_norm(x)
        logits = self.out_proj(x)  # [vocab_size]

        # Track tokens
        self.tokens_seen += 1

        return logits

    def forward_batch(
        self, token_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for a batch of tokens (sequential, not parallel).

        Args:
            token_ids: [seq_len] input token IDs
            target_ids: [seq_len] target token IDs (optional)

        Returns:
            logits: [seq_len, vocab_size]
        """
        seq_len = token_ids.shape[0]
        logits = []

        for i in range(seq_len):
            tid = token_ids[i].item()
            tgt = target_ids[i].item() if target_ids is not None else None
            logits.append(self.forward(tid, tgt))

        return torch.stack(logits)

    def maybe_evolve(self):
        """Evolve if enough tokens have been seen."""
        if self.tokens_seen % self.config.evolve_every == 0:
            self.evolve_all_layers()

    def evolve_all_layers(self):
        """Run evolution on all layers."""
        for layer in self.layers:
            # Check diversity before evolution
            pop = layer.population
            diversity = pop.decay.std().item()

            # Inject diversity if collapsed
            if diversity < self.config.min_diversity:
                self._inject_diversity(layer)

            # Evolve
            layer.evolve(
                selection_pressure=self.config.selection_pressure,
                elite_fraction=self.config.elite_fraction,
                mutation_rate=self.config.mutation_rate,
                mutation_strength=self.config.mutation_strength,
                sexual_rate=self.config.sexual_rate,
            )

        self.total_evolutions += 1

        # Record trajectory point after evolution
        self._record_trajectory_point()

    def _inject_diversity(self, layer: VectorizedLayer):
        """Inject random neurons to maintain diversity."""
        pop = layer.population
        n_inject = int(pop.n_neurons * self.config.diversity_injection_rate)

        # Find worst neurons
        worst_indices = torch.argsort(pop.fitness)[:n_inject]

        # Reinitialize them randomly
        cfg = pop.config
        for idx in worst_indices:
            idx = idx.item()
            pop.decay[idx] = cfg.decay_min + torch.rand(1).item() * (
                cfg.decay_max - cfg.decay_min
            )
            pop.gate_bias[idx] = cfg.gate_bias_min + torch.rand(1).item() * (
                cfg.gate_bias_max - cfg.gate_bias_min
            )
            pop.w_up[idx].normal_(0, cfg.weight_std)
            pop.w_gate[idx].normal_(0, cfg.weight_std)
            pop.w_down[idx].normal_(0, cfg.weight_std)
            pop.h[idx] = 0
            pop.age[idx] = 0
            pop.generation_born[idx] = layer.generation

    def get_all_stats(self) -> List[Dict]:
        """Get statistics for all layers."""
        return [layer.get_stats() for layer in self.layers]

    def get_hierarchy_summary(self) -> Dict:
        """Get summary of temporal hierarchy across layers."""
        decays = []
        gate_biases = []

        for layer in self.layers:
            stats = layer.get_stats()
            decays.append(stats["mean_decay"])
            gate_biases.append(stats["mean_gate_bias"])

        return {
            "decays_by_layer": decays,
            "gate_biases_by_layer": gate_biases,
            "decay_spread": max(decays) - min(decays),
            "gate_bias_spread": max(gate_biases) - min(gate_biases),
            "total_generations": sum(l.generation for l in self.layers),
            "total_births": sum(l.total_births for l in self.layers),
            "total_deaths": sum(l.total_deaths for l in self.layers),
        }

    def save_checkpoint(self, path: str):
        """Save model state."""
        torch.save(
            {
                "config": self.config,
                "model_state": self.state_dict(),
                "tokens_seen": self.tokens_seen,
                "total_evolutions": self.total_evolutions,
                "layer_generations": [l.generation for l in self.layers],
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str) -> "EvolvingModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        model.tokens_seen = checkpoint["tokens_seen"]
        model.total_evolutions = checkpoint["total_evolutions"]
        for i, gen in enumerate(checkpoint["layer_generations"]):
            model.layers[i].generation = gen
        return model


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing EvolvingModel...")

    # Small config for testing
    config = ModelConfig(
        vocab_size=1000,
        embed_dim=64,
        n_layers=4,
        n_neurons_per_layer=32,
        evolve_every=50,
    )

    model = EvolvingModel(config)

    print(f"Model created:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Neurons/layer: {config.n_neurons_per_layer}")
    print(f"  Total neurons: {config.n_layers * config.n_neurons_per_layer}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    logits = model.forward(42, target_id=100)
    print(f"\nForward pass:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Tokens seen: {model.tokens_seen}")

    # Run many tokens to trigger evolution
    print("\nRunning 200 tokens...")
    for i in range(200):
        logits = model.forward(i % 1000, target_id=(i + 1) % 1000)
        model.maybe_evolve()

    print(f"  Tokens seen: {model.tokens_seen}")
    print(f"  Total evolutions: {model.total_evolutions}")

    # Check hierarchy
    hierarchy = model.get_hierarchy_summary()
    print(f"\nHierarchy summary:")
    print(f"  Decays by layer: {[f'{d:.3f}' for d in hierarchy['decays_by_layer']]}")
    print(
        f"  Gate biases by layer: {[f'{g:.2f}' for g in hierarchy['gate_biases_by_layer']]}"
    )
    print(f"  Decay spread: {hierarchy['decay_spread']:.3f}")
    print(f"  Total births: {hierarchy['total_births']}")
    print(f"  Total deaths: {hierarchy['total_deaths']}")

    print("\nEvolvingModel test passed!")
