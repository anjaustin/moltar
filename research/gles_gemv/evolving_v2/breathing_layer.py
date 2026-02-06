"""
Breathing CfC Layer

The FFN and Equilibrium work as one system with a breathing rhythm:
- INHALE: New input arrives, FFN computes target state
- BREATHE: State relaxes toward target over multiple steps
- EXHALE: Output from equilibrated state

Key insight: With balanced gates (g~0.5) and lower decay (0.5-0.9),
equilibrium is reached in ~5 steps, not 30.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import random


@dataclass
class BreathingConfig:
    """Configuration for breathing layer."""

    n_neurons: int = 256
    embed_dim: int = 128

    # Breathing rhythm
    n_breathe: int = 5  # Steps per token (the rhythm)

    # Dynamics - tuned for fast equilibrium
    decay_min: float = 0.5  # Lower = faster settling
    decay_max: float = 0.9  # Still allows some memory
    gate_bias_min: float = -1.0  # More balanced gates
    gate_bias_max: float = 1.0

    # CfC
    alpha: float = 0.5

    # Niche-based evolution
    n_niches: int = 4  # Number of temporal niches
    use_niche_evolution: bool = True  # Whether to use niche-based evolution


class BreathingPopulation(nn.Module):
    """
    Population of CfC neurons with breathing dynamics.

    Weights are trainable (gradient descent).
    Dynamics (decay, gate_bias) are evolvable.
    """

    def __init__(self, config: BreathingConfig):
        super().__init__()
        self.config = config
        self.n_neurons = config.n_neurons
        self.embed_dim = config.embed_dim

        # Trainable weights
        self.w_up = nn.Parameter(torch.randn(config.n_neurons, config.embed_dim) * 0.02)
        self.w_gate = nn.Parameter(
            torch.randn(config.n_neurons, config.embed_dim) * 0.02
        )
        self.w_down = nn.Parameter(
            torch.randn(config.n_neurons, config.embed_dim) * 0.02
        )

        # Evolvable dynamics
        self.register_buffer("decay", torch.zeros(config.n_neurons))
        self.register_buffer("gate_bias", torch.zeros(config.n_neurons))

        # State
        self.register_buffer("h", torch.zeros(config.n_neurons))

        # Fitness tracking
        self.register_buffer("fitness", torch.zeros(config.n_neurons))
        self.register_buffer("grad_accum", torch.zeros(config.n_neurons))
        self.register_buffer("grad_count", torch.zeros(1))

        # Lineage
        self.register_buffer("age", torch.zeros(config.n_neurons, dtype=torch.long))

        # Niche assignment (which temporal band each neuron belongs to)
        self.register_buffer("niche", torch.zeros(config.n_neurons, dtype=torch.long))

        self._initialize_dynamics()

    def _initialize_dynamics(self):
        """Initialize with diversity across temporal scales, assigning niches."""
        n = self.n_neurons
        cfg = self.config
        n_niches = cfg.n_niches

        # Compute niche boundaries
        decay_range = cfg.decay_max - cfg.decay_min
        niche_width = decay_range / n_niches

        # Assign neurons to niches evenly
        neurons_per_niche = n // n_niches
        for i in range(n_niches):
            start_idx = i * neurons_per_niche
            end_idx = start_idx + neurons_per_niche if i < n_niches - 1 else n

            # Niche decay bounds
            niche_min = cfg.decay_min + i * niche_width
            niche_max = niche_min + niche_width

            # Initialize decay within niche
            niche_size = end_idx - start_idx
            self.decay[start_idx:end_idx] = torch.linspace(
                niche_min, niche_max, niche_size
            )

            # Assign niche ID
            self.niche[start_idx:end_idx] = i

            # Gate bias spread within niche
            self.gate_bias[start_idx:end_idx] = torch.linspace(
                cfg.gate_bias_min, cfg.gate_bias_max, niche_size
            )

        # Shuffle within each niche to avoid ordered structure
        for i in range(n_niches):
            mask = self.niche == i
            indices = torch.where(mask)[0]
            perm = indices[torch.randperm(len(indices))]
            self.decay[indices] = self.decay[perm].clone()
            self.gate_bias[indices] = self.gate_bias[perm].clone()

    def reset_state(self):
        self.h.zero_()

    def reset_fitness(self):
        self.fitness.zero_()
        self.grad_accum.zero_()
        self.grad_count.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Breathing forward pass.

        1. INHALE: Compute target from input
        2. BREATHE: Relax state toward target (no grad through iterations)
        3. EXHALE: Return equilibrated output
        """
        # INHALE: FFN computes target
        up = torch.mv(self.w_up, x)
        target = torch.tanh(up)

        # Gate projection (constant during breathing)
        gate_input = torch.mv(self.w_gate, x)

        # BREATHE: Multiple relaxation steps (detached for efficiency)
        # Only the final state affects gradients
        h = self.h.detach()
        for _ in range(self.config.n_breathe):
            g = torch.sigmoid(
                gate_input.detach() + self.config.alpha * h + self.gate_bias
            )
            h = (1 - g) * h * self.decay + g * target.detach()

        # Store equilibrated state
        self.h = h.detach()

        # EXHALE: Output from equilibrated state (this part has gradients)
        # Recompute with grad for the final step only
        g_final = torch.sigmoid(gate_input + self.config.alpha * h + self.gate_bias)
        h_final = (1 - g_final) * h * self.decay + g_final * target

        silu_gate = F.silu(gate_input)
        neuron_outputs = silu_gate * h_final

        output = torch.mv(self.w_down.t(), neuron_outputs)

        self.age += 1
        return output, neuron_outputs

    def accumulate_fitness(self, grad: torch.Tensor):
        """Accumulate gradient magnitude as fitness."""
        with torch.no_grad():
            self.grad_accum += grad.abs()
            self.grad_count += 1

    def finalize_fitness(self):
        """Convert accumulated gradients to fitness."""
        if self.grad_count > 0:
            self.fitness.copy_(self.grad_accum / self.grad_count)

    def kuramoto_order_parameter(self, t: float = 50.0) -> float:
        """
        Compute Kuramoto order parameter measuring temporal diversity.

        r × e^(iψ) = (1/N) × Σ e^(iθ_j)
        where ω_j = 1 - decay_j (natural frequency)
              θ_j = ω_j × t (phase at time t)

        Returns:
            r: Order parameter in [0, 1]
               r → 0: Diverse (good, neurons operate at different timescales)
               r → 1: Synchronized (bad, redundant neurons)
        """
        omega = 1.0 - self.decay  # Natural frequencies
        theta = omega * t  # Phases at time t

        # Complex order parameter
        z = torch.exp(1j * theta.to(torch.complex64))
        r = torch.abs(z.mean()).item()
        return r

    def get_niche_stats(self) -> Dict:
        """Get statistics for each niche."""
        n_niches = self.config.n_niches
        stats = {}
        for i in range(n_niches):
            mask = self.niche == i
            if mask.sum() > 0:
                stats[f"niche_{i}_count"] = mask.sum().item()
                stats[f"niche_{i}_decay_mean"] = self.decay[mask].mean().item()
                stats[f"niche_{i}_decay_std"] = (
                    self.decay[mask].std().item() if mask.sum() > 1 else 0.0
                )
                stats[f"niche_{i}_fitness_mean"] = self.fitness[mask].mean().item()
        return stats

    def get_stats(self) -> Dict:
        stats = {
            "n_neurons": self.n_neurons,
            "mean_decay": self.decay.mean().item(),
            "std_decay": self.decay.std().item(),
            "min_decay": self.decay.min().item(),
            "max_decay": self.decay.max().item(),
            "mean_gate_bias": self.gate_bias.mean().item(),
            "std_gate_bias": self.gate_bias.std().item(),
            "mean_fitness": self.fitness.mean().item(),
            "std_fitness": self.fitness.std().item(),
            "kuramoto_r": self.kuramoto_order_parameter(t=50.0),
        }
        # Add niche stats
        stats.update(self.get_niche_stats())
        return stats


class BreathingLayer(nn.Module):
    """Single breathing CfC layer with evolution."""

    def __init__(self, config: BreathingConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.population = BreathingPopulation(config)
        self.norm = nn.LayerNorm(config.embed_dim)

        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0

        self._neuron_outputs = None

    def reset_state(self):
        self.population.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        output, neuron_outputs = self.population.forward(x_norm)

        self._neuron_outputs = neuron_outputs
        if neuron_outputs.requires_grad:
            neuron_outputs.register_hook(self._grad_hook)

        return output

    def _grad_hook(self, grad):
        self.population.accumulate_fitness(grad)

    def evolve(
        self,
        selection_pressure: float = 0.5,
        elite_fraction: float = 0.1,
        mutation_rate: float = 0.5,
        mutation_strength: float = 0.2,
    ):
        """Evolve dynamics based on gradient-magnitude fitness."""
        pop = self.population
        pop.finalize_fitness()

        if self.config.use_niche_evolution:
            self._evolve_niche_based(
                selection_pressure, elite_fraction, mutation_rate, mutation_strength
            )
        else:
            self._evolve_global(
                selection_pressure, elite_fraction, mutation_rate, mutation_strength
            )

        pop.reset_fitness()
        self.generation += 1

    def _evolve_niche_based(
        self,
        selection_pressure: float,
        elite_fraction: float,
        mutation_rate: float,
        mutation_strength: float,
    ):
        """
        Niche-based evolution: neurons compete within their temporal niche.

        This preserves diversity by ensuring each niche (fast, medium, slow neurons)
        maintains its population. Prevents synchronization collapse.
        """
        pop = self.population
        cfg = pop.config
        n_niches = cfg.n_niches

        # Compute niche boundaries for mutation clamping
        decay_range = cfg.decay_max - cfg.decay_min
        niche_width = decay_range / n_niches

        for niche_idx in range(n_niches):
            # Get neurons in this niche
            mask = pop.niche == niche_idx
            niche_indices = torch.where(mask)[0]

            if len(niche_indices) < 2:
                continue

            # Sort by fitness within niche
            niche_fitness = pop.fitness[niche_indices]
            sorted_order = torch.argsort(niche_fitness, descending=True)
            sorted_indices = niche_indices[sorted_order]

            n_niche = len(sorted_indices)
            n_survivors = max(1, int(n_niche * selection_pressure))
            n_elite = max(1, int(n_niche * elite_fraction))

            survivor_indices = sorted_indices[:n_survivors].tolist()
            casualty_indices = sorted_indices[n_survivors:].tolist()
            elite_set = set(sorted_indices[:n_elite].tolist())

            self.total_deaths += len(casualty_indices)

            # Niche bounds for mutation clamping
            niche_decay_min = cfg.decay_min + niche_idx * niche_width
            niche_decay_max = niche_decay_min + niche_width

            for dst_idx in casualty_indices:
                parent_idx = random.choice(survivor_indices)
                is_elite = parent_idx in elite_set

                # Clone dynamics
                pop.decay[dst_idx] = pop.decay[parent_idx].clone()
                pop.gate_bias[dst_idx] = pop.gate_bias[parent_idx].clone()

                # Mutate if not elite, but stay within niche bounds
                if not is_elite:
                    if random.random() < mutation_rate:
                        delta = torch.randn(1).item() * mutation_strength * 0.1
                        pop.decay[dst_idx] = torch.clamp(
                            pop.decay[dst_idx] + delta,
                            niche_decay_min,
                            niche_decay_max,
                        )

                    if random.random() < mutation_rate:
                        delta = torch.randn(1).item() * mutation_strength * 0.5
                        pop.gate_bias[dst_idx] = torch.clamp(
                            pop.gate_bias[dst_idx] + delta,
                            cfg.gate_bias_min,
                            cfg.gate_bias_max,
                        )

                pop.age[dst_idx] = 0
                pop.h[dst_idx] = 0
                self.total_births += 1

    def _evolve_global(
        self,
        selection_pressure: float,
        elite_fraction: float,
        mutation_rate: float,
        mutation_strength: float,
    ):
        """Original global evolution (can cause diversity collapse)."""
        pop = self.population
        n = pop.n_neurons

        sorted_indices = torch.argsort(pop.fitness, descending=True)
        n_survivors = int(n * selection_pressure)
        n_elite = int(n * elite_fraction)

        survivor_indices = sorted_indices[:n_survivors].tolist()
        casualty_indices = sorted_indices[n_survivors:].tolist()

        self.total_deaths += len(casualty_indices)

        for dst_idx in casualty_indices:
            parent_idx = random.choice(survivor_indices)
            is_elite = parent_idx in sorted_indices[:n_elite].tolist()

            # Clone dynamics
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
                    delta = torch.randn(1).item() * mutation_strength * 0.5
                    pop.gate_bias[dst_idx] = torch.clamp(
                        pop.gate_bias[dst_idx] + delta,
                        pop.config.gate_bias_min,
                        pop.config.gate_bias_max,
                    )

            pop.age[dst_idx] = 0
            pop.h[dst_idx] = 0
            self.total_births += 1

    def get_stats(self) -> Dict:
        stats = self.population.get_stats()
        stats["layer_idx"] = self.layer_idx
        stats["generation"] = self.generation
        stats["total_births"] = self.total_births
        stats["total_deaths"] = self.total_deaths
        stats["n_breathe"] = self.config.n_breathe
        stats["use_niche_evolution"] = self.config.use_niche_evolution
        stats["n_niches"] = self.config.n_niches
        return stats


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing BreathingLayer with Niche Evolution...")
    print("=" * 60)

    config = BreathingConfig(
        n_neurons=64, embed_dim=32, n_breathe=5, n_niches=4, use_niche_evolution=True
    )
    layer = BreathingLayer(config)

    print(f"Config: {config.n_neurons} neurons, {config.n_breathe} breaths/token")
    print(f"Decay range: [{config.decay_min}, {config.decay_max}]")
    print(f"Niches: {config.n_niches} (niche evolution: {config.use_niche_evolution})")

    # Test niche initialization
    print(f"\nNiche distribution:")
    for i in range(config.n_niches):
        mask = layer.population.niche == i
        count = mask.sum().item()
        decay_mean = layer.population.decay[mask].mean().item()
        decay_std = layer.population.decay[mask].std().item()
        print(
            f"  Niche {i}: {count} neurons, decay={decay_mean:.3f} +/- {decay_std:.3f}"
        )

    # Test Kuramoto
    r_initial = layer.population.kuramoto_order_parameter(t=50.0)
    print(f"\nKuramoto order parameter (t=50): r={r_initial:.4f}")
    print(f"  (r -> 0: diverse, r -> 1: synchronized)")

    # Test forward
    x = torch.randn(32, requires_grad=True)
    target = torch.randn(32)

    output = layer(x)
    loss = F.mse_loss(output, target)
    loss.backward()

    print(f"\nForward pass:")
    print(f"  Output norm: {output.norm().item():.4f}")
    print(f"  h norm: {layer.population.h.norm().item():.4f}")

    # Run sequence and track diversity
    print(f"\nRunning 500 tokens with 5 evolutions...")
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    kuramoto_history = [r_initial]
    decay_std_history = [layer.population.decay.std().item()]

    for i in range(500):
        x = torch.randn(32)
        output = layer(x)
        loss = F.mse_loss(output, torch.randn(32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            layer.evolve()
            r = layer.population.kuramoto_order_parameter(t=50.0)
            kuramoto_history.append(r)
            decay_std_history.append(layer.population.decay.std().item())
            print(
                f"  Gen {layer.generation}: Kuramoto r={r:.4f}, decay std={decay_std_history[-1]:.4f}"
            )

    # Final stats
    stats = layer.get_stats()
    print(f"\nFinal Statistics:")
    print(
        f"  Fitness: mean={stats['mean_fitness']:.6f}, std={stats['std_fitness']:.6f}"
    )
    print(f"  Decay: mean={stats['mean_decay']:.3f}, std={stats['std_decay']:.3f}")
    print(f"  Kuramoto r: {stats['kuramoto_r']:.4f}")
    print(
        f"  Evolutions: {layer.generation}, Deaths: {layer.total_deaths}, Births: {layer.total_births}"
    )

    # Diversity preservation check
    print(f"\nDiversity Preservation:")
    print(f"  Initial decay std: {decay_std_history[0]:.4f}")
    print(f"  Final decay std:   {decay_std_history[-1]:.4f}")
    diversity_change = (
        (decay_std_history[-1] - decay_std_history[0]) / decay_std_history[0] * 100
    )
    print(f"  Change: {diversity_change:+.1f}%")

    if abs(diversity_change) < 20:
        print("  PASS: Diversity preserved!")
    else:
        print("  WARNING: Significant diversity change")

    # Compare with global evolution
    print("\n" + "=" * 60)
    print("Comparison: Global Evolution (diversity collapse expected)")
    print("=" * 60)

    config_global = BreathingConfig(
        n_neurons=64, embed_dim=32, n_breathe=5, n_niches=4, use_niche_evolution=False
    )
    layer_global = BreathingLayer(config_global)

    r_initial_g = layer_global.population.kuramoto_order_parameter(t=50.0)
    decay_std_initial_g = layer_global.population.decay.std().item()

    optimizer_g = torch.optim.Adam(layer_global.parameters(), lr=1e-3)
    for i in range(500):
        x = torch.randn(32)
        output = layer_global(x)
        loss = F.mse_loss(output, torch.randn(32))
        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        if (i + 1) % 100 == 0:
            layer_global.evolve()

    r_final_g = layer_global.population.kuramoto_order_parameter(t=50.0)
    decay_std_final_g = layer_global.population.decay.std().item()

    print(
        f"  Initial: Kuramoto r={r_initial_g:.4f}, decay std={decay_std_initial_g:.4f}"
    )
    print(f"  Final:   Kuramoto r={r_final_g:.4f}, decay std={decay_std_final_g:.4f}")
    diversity_change_g = (
        (decay_std_final_g - decay_std_initial_g) / decay_std_initial_g * 100
    )
    print(f"  Decay std change: {diversity_change_g:+.1f}%")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Niche evolution diversity change:  {diversity_change:+.1f}%")
    print(f"  Global evolution diversity change: {diversity_change_g:+.1f}%")
    print("=" * 60)
    print("\nBreathingLayer test complete!")
