"""
Vectorized Neural Population for Evolving SRC-FFN

All neurons in a layer computed as a single batched operation.
50-100x faster than per-neuron Python loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import random


@dataclass
class PopulationConfig:
    """Configuration for a neural population."""

    n_neurons: int = 512
    embed_dim: int = 256

    # CfC dynamics
    alpha: float = 0.5  # State feedback strength

    # Initialization ranges
    decay_min: float = 0.8
    decay_max: float = 0.999
    gate_bias_min: float = -4.0
    gate_bias_max: float = 0.0
    weight_std: float = 0.02


class VectorizedPopulation(nn.Module):
    """
    A population of neurons with vectorized computation.

    All neurons share the same structure but have individual:
    - decay: memory persistence (0.8 to 0.999)
    - gate_bias: openness to new input (-4 to 0)
    - w_up: input projection [embed_dim]
    - w_gate: gate projection [embed_dim]
    - w_down: output projection [embed_dim]
    """

    def __init__(self, config: PopulationConfig):
        super().__init__()
        self.config = config
        self.n_neurons = config.n_neurons
        self.embed_dim = config.embed_dim

        # === Evolvable Parameters (not nn.Parameter - we manage them manually) ===

        # Temporal dynamics [n_neurons]
        self.register_buffer("decay", torch.zeros(config.n_neurons))
        self.register_buffer("gate_bias", torch.zeros(config.n_neurons))

        # Weight matrices [n_neurons, embed_dim]
        self.register_buffer("w_up", torch.zeros(config.n_neurons, config.embed_dim))
        self.register_buffer("w_gate", torch.zeros(config.n_neurons, config.embed_dim))
        self.register_buffer("w_down", torch.zeros(config.n_neurons, config.embed_dim))

        # === State ===
        self.register_buffer("h", torch.zeros(config.n_neurons))  # Hidden state

        # === Fitness tracking ===
        self.register_buffer("fitness", torch.zeros(config.n_neurons))
        self.register_buffer("fitness_accum", torch.zeros(config.n_neurons))
        self.register_buffer("fitness_count", torch.zeros(config.n_neurons))

        # === Lineage tracking ===
        self.register_buffer("age", torch.zeros(config.n_neurons, dtype=torch.long))
        self.register_buffer(
            "generation_born", torch.zeros(config.n_neurons, dtype=torch.long)
        )
        self.register_buffer(
            "offspring_count", torch.zeros(config.n_neurons, dtype=torch.long)
        )
        self.neuron_ids = list(range(config.n_neurons))
        self.next_id = config.n_neurons

        # Initialize
        self._initialize_population()

    def _initialize_population(self):
        """Initialize population with diverse parameters."""
        n = self.n_neurons
        cfg = self.config

        # Diverse temporal scales - spread across range
        # Use linear spacing for initial diversity
        self.decay.copy_(torch.linspace(cfg.decay_min, cfg.decay_max, n))
        self.gate_bias.copy_(torch.linspace(cfg.gate_bias_min, cfg.gate_bias_max, n))

        # Shuffle to mix scales
        perm = torch.randperm(n)
        self.decay.copy_(self.decay[perm])
        self.gate_bias.copy_(self.gate_bias[perm])

        # Random weights
        self.w_up.normal_(0, cfg.weight_std)
        self.w_gate.normal_(0, cfg.weight_std)
        self.w_down.normal_(0, cfg.weight_std)

    def reset_state(self):
        """Reset hidden states (call between sequences)."""
        self.h.zero_()

    def reset_fitness(self):
        """Reset fitness accumulators (call before evolution)."""
        self.fitness.zero_()
        self.fitness_accum.zero_()
        self.fitness_count.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all neurons.

        Args:
            x: Input tensor [embed_dim]

        Returns:
            output: Combined output [embed_dim]
            neuron_outputs: Per-neuron scalar outputs [n_neurons]
        """
        # Project input to all neurons at once
        # x: [embed_dim], w_up: [n_neurons, embed_dim]
        up = torch.mv(self.w_up, x)  # [n_neurons]
        gate_proj = torch.mv(self.w_gate, x)  # [n_neurons]

        # CfC dynamics (vectorized) - detach state to prevent gradient accumulation
        h_detached = self.h.detach()

        # g = sigmoid(up + alpha * h + gate_bias)
        g = torch.sigmoid(
            up + self.config.alpha * h_detached + self.gate_bias
        )  # [n_neurons]

        # candidate = tanh(up)
        candidate = torch.tanh(up)  # [n_neurons]

        # State update: h = (1-g) * h * decay + g * candidate
        # Detach the new state so it doesn't accumulate gradients across tokens
        new_h = (1 - g) * h_detached * self.decay + g * candidate
        self.h = new_h.detach()  # Store detached for next step

        # Output: SiLU gate * hidden state (use new_h for gradient flow this step)
        silu_gate = F.silu(gate_proj)  # [n_neurons]
        neuron_outputs = silu_gate * new_h  # [n_neurons]

        # Combine: sum of (neuron_output * w_down)
        # neuron_outputs: [n_neurons], w_down: [n_neurons, embed_dim]
        output = torch.mv(self.w_down.t(), neuron_outputs)  # [embed_dim]

        # Track age
        self.age += 1

        return output, neuron_outputs

    def update_fitness(self, neuron_outputs: torch.Tensor, target_embed: torch.Tensor):
        """
        Update fitness based on contribution to target prediction.

        Args:
            neuron_outputs: Per-neuron outputs [n_neurons]
            target_embed: Target token embedding [embed_dim]
        """
        # Detach from computation graph - fitness is NOT for gradient descent
        with torch.no_grad():
            neuron_outputs = neuron_outputs.detach()
            target_embed = target_embed.detach()

            # Contribution = how much each neuron pushed toward target
            # contribution_i = dot(neuron_output_i * w_down_i, target_embed)
            contributions = (
                torch.mv(self.w_down, target_embed) * neuron_outputs
            )  # [n_neurons]

        # Accumulate for averaging
        self.fitness_accum += contributions
        self.fitness_count += 1

    def finalize_fitness(self):
        """Convert accumulated fitness to final fitness score."""
        # Average over all tokens seen
        mask = self.fitness_count > 0
        self.fitness[mask] = self.fitness_accum[mask] / self.fitness_count[mask]

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
            "max_fitness": self.fitness.max().item(),
            "min_fitness": self.fitness.min().item(),
            "mean_age": self.age.float().mean().item(),
            "max_age": self.age.max().item(),
        }

    def get_neuron_data(self, idx: int) -> Dict:
        """Get data for a specific neuron."""
        return {
            "id": self.neuron_ids[idx],
            "decay": self.decay[idx].item(),
            "gate_bias": self.gate_bias[idx].item(),
            "fitness": self.fitness[idx].item(),
            "age": self.age[idx].item(),
            "generation_born": self.generation_born[idx].item(),
            "offspring_count": self.offspring_count[idx].item(),
        }

    def clone_neuron(
        self,
        src_idx: int,
        dst_idx: int,
        mutate: bool = True,
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.1,
    ):
        """
        Clone neuron from src to dst with optional mutation.

        Args:
            src_idx: Source neuron index
            dst_idx: Destination neuron index
            mutate: Whether to apply mutation
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Scale of mutations
        """
        # Copy parameters
        self.decay[dst_idx] = self.decay[src_idx].clone()
        self.gate_bias[dst_idx] = self.gate_bias[src_idx].clone()
        self.w_up[dst_idx] = self.w_up[src_idx].clone()
        self.w_gate[dst_idx] = self.w_gate[src_idx].clone()
        self.w_down[dst_idx] = self.w_down[src_idx].clone()

        if mutate:
            # Mutate decay
            if random.random() < mutation_rate:
                delta = torch.randn(1).item() * mutation_strength * 0.05
                self.decay[dst_idx] = torch.clamp(
                    self.decay[dst_idx] + delta,
                    self.config.decay_min,
                    self.config.decay_max,
                )

            # Mutate gate_bias
            if random.random() < mutation_rate:
                delta = torch.randn(1).item() * mutation_strength * 0.5
                self.gate_bias[dst_idx] = torch.clamp(
                    self.gate_bias[dst_idx] + delta,
                    self.config.gate_bias_min - 2,  # Allow some expansion
                    self.config.gate_bias_max + 2,
                )

            # Mutate weights
            if random.random() < mutation_rate:
                self.w_up[dst_idx] += (
                    torch.randn_like(self.w_up[dst_idx]) * mutation_strength * 0.01
                )
            if random.random() < mutation_rate:
                self.w_gate[dst_idx] += (
                    torch.randn_like(self.w_gate[dst_idx]) * mutation_strength * 0.01
                )
            if random.random() < mutation_rate:
                self.w_down[dst_idx] += (
                    torch.randn_like(self.w_down[dst_idx]) * mutation_strength * 0.01
                )

        # Update lineage
        self.offspring_count[src_idx] += 1
        self.neuron_ids[dst_idx] = self.next_id
        self.next_id += 1
        self.age[dst_idx] = 0
        self.h[dst_idx] = 0  # Reset state

    def crossover(self, parent1_idx: int, parent2_idx: int, child_idx: int):
        """
        Sexual reproduction: combine two parents into child.

        Args:
            parent1_idx: First parent index
            parent2_idx: Second parent index
            child_idx: Child neuron index
        """
        # Random choice for scalar parameters
        if random.random() < 0.5:
            self.decay[child_idx] = self.decay[parent1_idx].clone()
        else:
            self.decay[child_idx] = self.decay[parent2_idx].clone()

        if random.random() < 0.5:
            self.gate_bias[child_idx] = self.gate_bias[parent1_idx].clone()
        else:
            self.gate_bias[child_idx] = self.gate_bias[parent2_idx].clone()

        # Element-wise random choice for weights
        mask = torch.rand(self.embed_dim) < 0.5
        self.w_up[child_idx] = torch.where(
            mask, self.w_up[parent1_idx], self.w_up[parent2_idx]
        )

        mask = torch.rand(self.embed_dim) < 0.5
        self.w_gate[child_idx] = torch.where(
            mask, self.w_gate[parent1_idx], self.w_gate[parent2_idx]
        )

        mask = torch.rand(self.embed_dim) < 0.5
        self.w_down[child_idx] = torch.where(
            mask, self.w_down[parent1_idx], self.w_down[parent2_idx]
        )

        # Update lineage
        self.offspring_count[parent1_idx] += 1
        self.offspring_count[parent2_idx] += 1
        self.neuron_ids[child_idx] = self.next_id
        self.next_id += 1
        self.age[child_idx] = 0
        self.h[child_idx] = 0


class VectorizedLayer(nn.Module):
    """
    A single SRC-FFN layer with evolving population.
    """

    def __init__(self, config: PopulationConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # The population
        self.population = VectorizedPopulation(config)

        # Layer norm (gradient-trained)
        self.norm = nn.LayerNorm(config.embed_dim)

        # Evolution tracking
        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0

    def reset_state(self):
        """Reset hidden states."""
        self.population.reset_state()

    def forward(
        self, x: torch.Tensor, target_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional fitness tracking.

        Args:
            x: Input [embed_dim]
            target_embed: Target embedding for fitness [embed_dim], optional

        Returns:
            output: Layer output [embed_dim]
        """
        # Normalize
        x_norm = self.norm(x)

        # Forward through population
        output, neuron_outputs = self.population.forward(x_norm)

        # Track fitness if target provided
        if target_embed is not None:
            self.population.update_fitness(neuron_outputs, target_embed)

        return output

    def evolve(
        self,
        selection_pressure: float = 0.5,
        elite_fraction: float = 0.1,
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.1,
        sexual_rate: float = 0.3,
    ):
        """
        Run one generation of evolution.

        Args:
            selection_pressure: Fraction that survives (0.5 = top 50%)
            elite_fraction: Fraction protected from mutation (0.1 = top 10%)
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Scale of mutations
            sexual_rate: Fraction of offspring from crossover
        """
        pop = self.population
        n = pop.n_neurons

        # Finalize fitness scores
        pop.finalize_fitness()

        # Sort by fitness (descending)
        sorted_indices = torch.argsort(pop.fitness, descending=True)

        # Determine survivors and casualties
        n_survivors = int(n * selection_pressure)
        n_elite = int(n * elite_fraction)
        survivor_indices = sorted_indices[:n_survivors].tolist()
        casualty_indices = sorted_indices[n_survivors:].tolist()

        self.total_deaths += len(casualty_indices)

        # Replace casualties with offspring from survivors
        for i, dst_idx in enumerate(casualty_indices):
            if random.random() < sexual_rate and len(survivor_indices) >= 2:
                # Sexual reproduction
                parents = random.sample(survivor_indices, 2)
                pop.crossover(parents[0], parents[1], dst_idx)
                # Also mutate the child
                pop.clone_neuron(
                    dst_idx,
                    dst_idx,
                    mutate=True,
                    mutation_rate=mutation_rate,
                    mutation_strength=mutation_strength,
                )
            else:
                # Asexual reproduction
                parent_idx = random.choice(survivor_indices)
                # Elite neurons: copy without mutation
                is_elite = parent_idx in sorted_indices[:n_elite].tolist()
                pop.clone_neuron(
                    parent_idx,
                    dst_idx,
                    mutate=not is_elite,
                    mutation_rate=mutation_rate,
                    mutation_strength=mutation_strength,
                )

            self.total_births += 1

        # Reset fitness for next generation
        pop.reset_fitness()

        # Update generation counter
        self.generation += 1
        for idx in casualty_indices:
            pop.generation_born[idx] = self.generation

    def get_stats(self) -> Dict:
        """Get layer statistics."""
        stats = self.population.get_stats()
        stats["layer_idx"] = self.layer_idx
        stats["generation"] = self.generation
        stats["total_births"] = self.total_births
        stats["total_deaths"] = self.total_deaths
        return stats


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing VectorizedPopulation...")

    config = PopulationConfig(n_neurons=64, embed_dim=32)
    layer = VectorizedLayer(config, layer_idx=0)

    # Test forward pass
    x = torch.randn(32)
    target = torch.randn(32)

    print(f"Input shape: {x.shape}")

    # Run some forward passes
    for i in range(100):
        output = layer(x, target_embed=target)

    print(f"Output shape: {output.shape}")
    print(f"Stats before evolution: {layer.get_stats()}")

    # Evolve
    layer.evolve()
    print(f"Stats after evolution: {layer.get_stats()}")

    # Run more and evolve again
    for i in range(100):
        output = layer(x, target_embed=target)
    layer.evolve()
    print(f"Stats after 2nd evolution: {layer.get_stats()}")

    print("\nVectorizedPopulation test passed!")
