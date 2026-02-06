"""
Hybrid Trainable/Evolvable Layer

Key insight: Weights need gradients (precise learning), dynamics need evolution (exploration).

- Weights (w_up, w_gate, w_down): nn.Parameter, trained with gradient descent
- Dynamics (decay, gate_bias): buffer, evolved based on gradient-magnitude fitness

Fitness = how much gradient flows through each neuron = how much it matters for the loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import random


@dataclass
class HybridConfig:
    """Configuration for hybrid layer."""

    n_neurons: int = 256
    embed_dim: int = 128

    # CfC dynamics
    alpha: float = 0.5

    # Evolvable parameter ranges
    decay_min: float = 0.8
    decay_max: float = 0.999
    gate_bias_min: float = -4.0
    gate_bias_max: float = 2.0  # Allow positive gate bias for very open gates

    # Weight initialization
    weight_std: float = 0.02


class HybridPopulation(nn.Module):
    """
    Population of neurons with trainable weights and evolvable dynamics.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.n_neurons = config.n_neurons
        self.embed_dim = config.embed_dim

        # === TRAINABLE weights (gradient descent) ===
        self.w_up = nn.Parameter(
            torch.randn(config.n_neurons, config.embed_dim) * config.weight_std
        )
        self.w_gate = nn.Parameter(
            torch.randn(config.n_neurons, config.embed_dim) * config.weight_std
        )
        self.w_down = nn.Parameter(
            torch.randn(config.n_neurons, config.embed_dim) * config.weight_std
        )

        # === EVOLVABLE dynamics (evolution) ===
        self.register_buffer("decay", torch.zeros(config.n_neurons))
        self.register_buffer("gate_bias", torch.zeros(config.n_neurons))

        # === State ===
        self.register_buffer("h", torch.zeros(config.n_neurons))

        # === Fitness tracking (gradient magnitude) ===
        self.register_buffer("fitness", torch.zeros(config.n_neurons))
        self.register_buffer("grad_accum", torch.zeros(config.n_neurons))
        self.register_buffer("grad_count", torch.zeros(1))

        # === Lineage ===
        self.register_buffer("age", torch.zeros(config.n_neurons, dtype=torch.long))
        self.register_buffer(
            "generation_born", torch.zeros(config.n_neurons, dtype=torch.long)
        )

        self._initialize_dynamics()

    def _initialize_dynamics(self):
        """Initialize evolvable parameters with diversity."""
        n = self.n_neurons
        cfg = self.config

        # Spread decay across the full range
        self.decay.copy_(torch.linspace(cfg.decay_min, cfg.decay_max, n))
        self.gate_bias.copy_(torch.linspace(cfg.gate_bias_min, cfg.gate_bias_max, n))

        # Shuffle for diversity
        perm = torch.randperm(n)
        self.decay.copy_(self.decay[perm])
        self.gate_bias.copy_(self.gate_bias[perm])

    def reset_state(self):
        """Reset hidden states."""
        self.h.zero_()

    def reset_fitness(self):
        """Reset fitness accumulators."""
        self.fitness.zero_()
        self.grad_accum.zero_()
        self.grad_count.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns output and neuron_outputs (for gradient-based fitness).
        """
        # Project input
        up = torch.mv(self.w_up, x)  # [n_neurons]
        gate_proj = torch.mv(self.w_gate, x)  # [n_neurons]

        # CfC dynamics (detach hidden state for BPTT truncation)
        h_detached = self.h.detach()

        # Gate: how much to update vs remember
        g = torch.sigmoid(up + self.config.alpha * h_detached + self.gate_bias)

        # Candidate: what to potentially write
        candidate = torch.tanh(up)

        # State update
        new_h = (1 - g) * h_detached * self.decay + g * candidate
        self.h = new_h.detach()  # Store for next step

        # Output: SiLU-gated hidden state
        silu_gate = F.silu(gate_proj)
        neuron_outputs = silu_gate * new_h  # Keep grad for fitness

        # Combine outputs
        output = torch.mv(self.w_down.t(), neuron_outputs)

        # Age all neurons
        self.age += 1

        return output, neuron_outputs

    def accumulate_fitness(self, neuron_grad: torch.Tensor):
        """
        Accumulate gradient magnitude as fitness.

        Args:
            neuron_grad: gradient of loss w.r.t. neuron_outputs [n_neurons]
        """
        with torch.no_grad():
            self.grad_accum += neuron_grad.abs()
            self.grad_count += 1

    def finalize_fitness(self):
        """Convert accumulated gradients to fitness scores."""
        if self.grad_count > 0:
            self.fitness.copy_(self.grad_accum / self.grad_count)

    def get_stats(self) -> Dict:
        """Get population statistics."""
        return {
            "n_neurons": self.n_neurons,
            "mean_decay": self.decay.mean().item(),
            "std_decay": self.decay.std().item(),
            "min_decay": self.decay.min().item(),
            "max_decay": self.decay.max().item(),
            "mean_gate_bias": self.gate_bias.mean().item(),
            "std_gate_bias": self.gate_bias.std().item(),
            "min_gate_bias": self.gate_bias.min().item(),
            "max_gate_bias": self.gate_bias.max().item(),
            "mean_fitness": self.fitness.mean().item(),
            "std_fitness": self.fitness.std().item(),
            "max_fitness": self.fitness.max().item(),
            "min_fitness": self.fitness.min().item(),
        }


class HybridLayer(nn.Module):
    """
    Single hybrid layer with evolving dynamics.
    """

    def __init__(self, config: HybridConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.population = HybridPopulation(config)
        self.norm = nn.LayerNorm(config.embed_dim)

        # Evolution tracking
        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0

        # For capturing neuron gradients
        self._neuron_outputs = None

    def reset_state(self):
        self.population.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, storing neuron_outputs for gradient capture."""
        x_norm = self.norm(x)
        output, neuron_outputs = self.population.forward(x_norm)

        # Store for gradient hook
        self._neuron_outputs = neuron_outputs

        # Register hook to capture gradient
        if neuron_outputs.requires_grad:
            neuron_outputs.register_hook(self._grad_hook)

        return output

    def _grad_hook(self, grad):
        """Hook called during backward to capture neuron gradients."""
        self.population.accumulate_fitness(grad)

    def evolve(
        self,
        selection_pressure: float = 0.5,
        elite_fraction: float = 0.1,
        mutation_rate: float = 0.5,
        mutation_strength: float = 0.2,
    ):
        """
        Evolve the population based on gradient-magnitude fitness.

        Only evolves decay and gate_bias. Weights are left to gradients.
        """
        pop = self.population
        n = pop.n_neurons

        # Finalize fitness
        pop.finalize_fitness()

        # Sort by fitness
        sorted_indices = torch.argsort(pop.fitness, descending=True)

        n_survivors = int(n * selection_pressure)
        n_elite = int(n * elite_fraction)

        survivor_indices = sorted_indices[:n_survivors].tolist()
        casualty_indices = sorted_indices[n_survivors:].tolist()

        self.total_deaths += len(casualty_indices)

        # Replace casualties
        for dst_idx in casualty_indices:
            parent_idx = random.choice(survivor_indices)
            is_elite = parent_idx in sorted_indices[:n_elite].tolist()

            # Clone dynamics (not weights!)
            pop.decay[dst_idx] = pop.decay[parent_idx].clone()
            pop.gate_bias[dst_idx] = pop.gate_bias[parent_idx].clone()

            # Mutate if not elite
            if not is_elite:
                if random.random() < mutation_rate:
                    delta = torch.randn(1).item() * mutation_strength * 0.1
                    pop.decay[dst_idx] = torch.clamp(
                        pop.decay[dst_idx] + delta,
                        pop.config.decay_min,
                        pop.config.decay_max,
                    )

                if random.random() < mutation_rate:
                    delta = torch.randn(1).item() * mutation_strength * 1.0
                    pop.gate_bias[dst_idx] = torch.clamp(
                        pop.gate_bias[dst_idx] + delta,
                        pop.config.gate_bias_min,
                        pop.config.gate_bias_max,
                    )

            # Reset age
            pop.age[dst_idx] = 0
            pop.generation_born[dst_idx] = self.generation + 1
            pop.h[dst_idx] = 0

            self.total_births += 1

        # Reset fitness for next generation
        pop.reset_fitness()
        self.generation += 1

    def get_stats(self) -> Dict:
        stats = self.population.get_stats()
        stats["layer_idx"] = self.layer_idx
        stats["generation"] = self.generation
        stats["total_births"] = self.total_births
        stats["total_deaths"] = self.total_deaths
        return stats


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing HybridLayer...")

    config = HybridConfig(n_neurons=64, embed_dim=32)
    layer = HybridLayer(config)

    # Test forward and gradient flow
    x = torch.randn(32, requires_grad=True)
    target = torch.randn(32)

    output = layer(x)
    loss = F.mse_loss(output, target)
    loss.backward()

    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.6f}")
    print(f"x.grad exists: {x.grad is not None}")

    # Run many steps to accumulate fitness
    print("\nRunning 100 steps...")
    for i in range(100):
        x = torch.randn(32, requires_grad=True)
        target = torch.randn(32)
        output = layer(x)
        loss = F.mse_loss(output, target)
        loss.backward()

    # Finalize and check fitness
    layer.population.finalize_fitness()
    stats = layer.get_stats()
    print(f"\nFitness stats:")
    print(f"  mean: {stats['mean_fitness']:.6f}")
    print(f"  std: {stats['std_fitness']:.6f}")
    print(f"  range: [{stats['min_fitness']:.6f}, {stats['max_fitness']:.6f}]")

    # Evolve
    print("\nEvolving...")
    layer.evolve()
    print(f"Deaths: {layer.total_deaths}, Births: {layer.total_births}")

    # Check dynamics changed
    print(f"\nDynamics after evolution:")
    print(
        f"  Decay: mean={layer.population.decay.mean().item():.4f}, std={layer.population.decay.std().item():.4f}"
    )
    print(
        f"  Gate bias: mean={layer.population.gate_bias.mean().item():.2f}, std={layer.population.gate_bias.std().item():.2f}"
    )

    print("\nHybridLayer test passed!")
