"""
Evolving SRC-FFN: Neural Darwinism Meets Language Modeling

Each FFN neuron is an individual with:
- Genome: decay, gate_bias, weights
- Fitness: contribution to prediction
- Reproduction: successful neurons spawn offspring
- Death: unsuccessful neurons are replaced

This is not a metaphor. This is the architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np
from collections import deque


# =============================================================================
# Neuron: The Individual
# =============================================================================


@dataclass
class NeuronGenome:
    """The genetic code of a single neuron."""

    decay: float  # Memory lifespan
    gate_bias: float  # Openness to new information
    w_up: torch.Tensor  # What I respond to
    w_gate: torch.Tensor  # What controls my gate
    w_down: torch.Tensor  # How I contribute to output

    # Lineage tracking
    generation_born: int = 0
    parent_id: Optional[int] = None
    mutations: int = 0

    def clone(self) -> "NeuronGenome":
        return NeuronGenome(
            decay=self.decay,
            gate_bias=self.gate_bias,
            w_up=self.w_up.clone(),
            w_gate=self.w_gate.clone(),
            w_down=self.w_down.clone(),
            generation_born=self.generation_born,
            parent_id=self.parent_id,
            mutations=self.mutations,
        )


class Neuron:
    """A single evolving neuron with CfC dynamics."""

    _id_counter = 0

    def __init__(self, embed_dim: int, genome: Optional[NeuronGenome] = None):
        self.id = Neuron._id_counter
        Neuron._id_counter += 1

        self.embed_dim = embed_dim

        if genome is None:
            # Random initialization with variation
            self.genome = NeuronGenome(
                decay=0.8 + 0.19 * random.random(),  # 0.8 to 0.99
                gate_bias=-4.0 + 4.0 * random.random(),  # -4 to 0
                w_up=torch.randn(embed_dim) * 0.02,
                w_gate=torch.randn(embed_dim) * 0.02,
                w_down=torch.randn(embed_dim) * 0.02,
            )
        else:
            self.genome = genome

        # State
        self.h = 0.0  # Hidden state (scalar for efficiency)

        # Fitness tracking
        self.fitness = 0.0
        self.fitness_history = deque(maxlen=100)
        self.activation_history = deque(maxlen=100)
        self.contribution_history = deque(maxlen=100)

        # Lifetime stats
        self.age = 0
        self.total_activations = 0
        self.times_reproduced = 0

    def reset(self):
        self.h = 0.0

    def forward(self, x: torch.Tensor, alpha: float = 0.5) -> Tuple[float, float]:
        """
        Forward pass through this neuron.

        Returns:
            output: This neuron's contribution to layer output
            activation: The magnitude of this neuron's response
        """
        # Projections
        up = (x * self.genome.w_up).sum().item()
        gate_val = (x * self.genome.w_gate).sum().item()

        # CfC dynamics
        g = 1.0 / (1.0 + math.exp(-(up + alpha * self.h + self.genome.gate_bias)))
        candidate = math.tanh(up)

        # State update
        self.h = (1 - g) * self.h * self.genome.decay + g * candidate

        # Output
        activation = abs(self.h)
        silu_gate = gate_val / (1.0 + math.exp(-gate_val)) if gate_val > -20 else 0
        output_scalar = silu_gate * self.h

        # Track
        self.activation_history.append(activation)
        self.total_activations += 1
        self.age += 1

        return output_scalar, activation

    def compute_contribution(
        self, output_scalar: float, target_direction: float
    ) -> float:
        """How much did this neuron help?"""
        # Contribution = alignment with what was needed
        contribution = output_scalar * target_direction
        self.contribution_history.append(contribution)
        return contribution

    def update_fitness(self, contribution: float, method: str = "exponential"):
        """Update fitness based on contribution."""
        if method == "exponential":
            # Exponential moving average
            alpha = 0.1
            self.fitness = alpha * contribution + (1 - alpha) * self.fitness
        elif method == "cumulative":
            self.fitness += contribution
        elif method == "recent":
            # Average of recent contributions
            self.fitness = (
                np.mean(list(self.contribution_history))
                if self.contribution_history
                else 0
            )

        self.fitness_history.append(self.fitness)

    def mutate(
        self, mutation_rate: float = 0.1, mutation_strength: float = 0.1
    ) -> "Neuron":
        """Create a mutated offspring."""
        child_genome = self.genome.clone()

        # Mutate decay
        if random.random() < mutation_rate:
            child_genome.decay = max(
                0.5,
                min(
                    0.999, child_genome.decay + random.gauss(0, mutation_strength * 0.1)
                ),
            )

        # Mutate gate_bias
        if random.random() < mutation_rate:
            child_genome.gate_bias = max(
                -6, min(2, child_genome.gate_bias + random.gauss(0, mutation_strength))
            )

        # Mutate weights
        if random.random() < mutation_rate:
            child_genome.w_up = (
                child_genome.w_up
                + torch.randn_like(child_genome.w_up) * mutation_strength * 0.02
            )
        if random.random() < mutation_rate:
            child_genome.w_gate = (
                child_genome.w_gate
                + torch.randn_like(child_genome.w_gate) * mutation_strength * 0.02
            )
        if random.random() < mutation_rate:
            child_genome.w_down = (
                child_genome.w_down
                + torch.randn_like(child_genome.w_down) * mutation_strength * 0.02
            )

        child_genome.parent_id = self.id
        child_genome.mutations = self.genome.mutations + 1

        child = Neuron(self.embed_dim, child_genome)
        self.times_reproduced += 1

        return child

    def crossover(self, other: "Neuron") -> "Neuron":
        """Sexual reproduction with another neuron."""
        child_genome = NeuronGenome(
            decay=self.genome.decay if random.random() < 0.5 else other.genome.decay,
            gate_bias=self.genome.gate_bias
            if random.random() < 0.5
            else other.genome.gate_bias,
            w_up=torch.where(
                torch.rand_like(self.genome.w_up) < 0.5,
                self.genome.w_up,
                other.genome.w_up,
            ),
            w_gate=torch.where(
                torch.rand_like(self.genome.w_gate) < 0.5,
                self.genome.w_gate,
                other.genome.w_gate,
            ),
            w_down=torch.where(
                torch.rand_like(self.genome.w_down) < 0.5,
                self.genome.w_down,
                other.genome.w_down,
            ),
            parent_id=self.id,  # Track one parent
        )

        return Neuron(self.embed_dim, child_genome)


# =============================================================================
# Population: The Ecosystem
# =============================================================================


class NeuronPopulation:
    """A population of neurons that evolves."""

    def __init__(
        self,
        size: int,
        embed_dim: int,
        selection_pressure: float = 0.5,  # Fraction that survives
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        sexual_reproduction_rate: float = 0.2,
    ):
        self.size = size
        self.embed_dim = embed_dim
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.sexual_reproduction_rate = sexual_reproduction_rate

        # Create initial population with diversity
        self.neurons = []
        for i in range(size):
            neuron = Neuron(embed_dim)
            # Ensure initial diversity in temporal scales
            scale = i / size
            neuron.genome.decay = 0.8 + 0.19 * scale
            neuron.genome.gate_bias = -4.0 + 4.0 * (1 - scale)
            self.neurons.append(neuron)

        # Evolution stats
        self.generation = 0
        self.births = 0
        self.deaths = 0
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.diversity_history = []

    def reset(self):
        for n in self.neurons:
            n.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all neurons, returning combined output."""
        output = torch.zeros(self.embed_dim)
        activations = []

        for neuron in self.neurons:
            scalar_out, activation = neuron.forward(x)
            output += scalar_out * neuron.genome.w_down
            activations.append((neuron, activation, scalar_out))

        return output, activations

    def assign_fitness(self, activations: List, target_direction: torch.Tensor):
        """Assign fitness to each neuron based on contribution."""
        target_dir = target_direction.mean().item()  # Simplified

        for neuron, activation, scalar_out in activations:
            contribution = neuron.compute_contribution(scalar_out, target_dir)
            neuron.update_fitness(contribution)

    def evolve(self):
        """One generation of evolution."""
        # Sort by fitness
        self.neurons.sort(key=lambda n: n.fitness, reverse=True)

        # Track stats
        fitnesses = [n.fitness for n in self.neurons]
        self.best_fitness_history.append(max(fitnesses))
        self.mean_fitness_history.append(np.mean(fitnesses))

        # Measure diversity (std of decay and gate_bias)
        decays = [n.genome.decay for n in self.neurons]
        biases = [n.genome.gate_bias for n in self.neurons]
        diversity = np.std(decays) + np.std(biases)
        self.diversity_history.append(diversity)

        # Selection: top fraction survives
        n_survivors = int(self.size * self.selection_pressure)
        survivors = self.neurons[:n_survivors]

        # Track deaths
        self.deaths += self.size - n_survivors

        # Reproduction
        offspring = []
        n_offspring_needed = self.size - n_survivors

        while len(offspring) < n_offspring_needed:
            if random.random() < self.sexual_reproduction_rate and len(survivors) >= 2:
                # Sexual reproduction
                parent1, parent2 = random.sample(survivors, 2)
                child = parent1.crossover(parent2)
                # Also mutate
                if random.random() < self.mutation_rate:
                    child = child.mutate(self.mutation_rate, self.mutation_strength)
            else:
                # Asexual reproduction (clone + mutate)
                parent = random.choice(survivors)
                child = parent.mutate(self.mutation_rate, self.mutation_strength)

            child.genome.generation_born = self.generation + 1
            offspring.append(child)
            self.births += 1

        # New population
        self.neurons = survivors + offspring
        self.generation += 1

        # Reset fitness for next generation
        for n in self.neurons:
            n.fitness = 0.0

    def get_stats(self) -> Dict:
        """Get population statistics."""
        decays = [n.genome.decay for n in self.neurons]
        biases = [n.genome.gate_bias for n in self.neurons]
        ages = [n.age for n in self.neurons]

        return {
            "generation": self.generation,
            "births": self.births,
            "deaths": self.deaths,
            "mean_decay": np.mean(decays),
            "std_decay": np.std(decays),
            "mean_gate_bias": np.mean(biases),
            "std_gate_bias": np.std(biases),
            "mean_age": np.mean(ages),
            "max_age": max(ages),
            "best_fitness": self.best_fitness_history[-1]
            if self.best_fitness_history
            else 0,
            "mean_fitness": self.mean_fitness_history[-1]
            if self.mean_fitness_history
            else 0,
        }


# =============================================================================
# Evolving SRC-FFN Layer
# =============================================================================


class EvolvingSRCFFNLayer(nn.Module):
    """An SRC-FFN layer where neurons evolve."""

    def __init__(
        self,
        embed_dim: int,
        n_neurons: int = 256,
        evolve_every: int = 10,  # Evolve every N tokens
        **evolution_kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_neurons = n_neurons
        self.evolve_every = evolve_every

        # The population
        self.population = NeuronPopulation(n_neurons, embed_dim, **evolution_kwargs)

        # Normalization (still learned traditionally)
        self.norm = nn.LayerNorm(embed_dim)

        # Token counter for evolution timing
        self.tokens_seen = 0

    def reset(self):
        self.population.reset()

    def forward(
        self, x: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional fitness assignment.

        Args:
            x: Input [embed_dim]
            target: Target for fitness assignment [embed_dim], optional

        Returns:
            output: Layer output [embed_dim]
        """
        x_norm = self.norm(x)

        # Forward through population
        output, activations = self.population.forward(x_norm)

        # Assign fitness if target provided
        if target is not None:
            error = target - output
            self.population.assign_fitness(activations, error)

        # Maybe evolve
        self.tokens_seen += 1
        if self.tokens_seen % self.evolve_every == 0:
            self.population.evolve()

        return output


# =============================================================================
# Full Evolving Model
# =============================================================================


class EvolvingSRCFFN(nn.Module):
    """Full evolving SRC-FFN language model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_layers: int,
        n_neurons_per_layer: int = 256,
        evolve_every: int = 10,
        **evolution_kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # Embedding (learned traditionally)
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Evolving layers
        self.layers = nn.ModuleList(
            [
                EvolvingSRCFFNLayer(
                    embed_dim, n_neurons_per_layer, evolve_every, **evolution_kwargs
                )
                for _ in range(n_layers)
            ]
        )

        # Output (learned traditionally)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight  # Tie weights

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, token_id: int, target_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward one token.

        Args:
            token_id: Input token
            target_id: Target token for fitness (optional)

        Returns:
            logits: [vocab_size]
        """
        x = self.embed.weight[token_id]

        target_embed = self.embed.weight[target_id] if target_id is not None else None

        for layer in self.layers:
            out = layer(x, target_embed)
            x = x + out

        x = self.out_norm(x)
        logits = self.out_proj(x)

        return logits

    def get_evolution_stats(self) -> List[Dict]:
        """Get evolution statistics for all layers."""
        return [layer.population.get_stats() for layer in self.layers]


# =============================================================================
# Training Loop: Evolution in Action
# =============================================================================


def train_evolving_model(
    model: EvolvingSRCFFN,
    text: str,
    char_to_id: Dict[str, int],
    id_to_char: Dict[int, str],
    n_epochs: int = 5,
    log_every: int = 100,
):
    """
    Train the evolving model on text.

    This is not backprop. This is evolution.
    """
    print("=" * 60)
    print("EVOLVING SRC-FFN TRAINING")
    print("=" * 60)
    print(f"Text length: {len(text)} characters")
    print(f"Vocab size: {len(char_to_id)}")
    print(f"Neurons per layer: {model.layers[0].n_neurons}")
    print(f"Evolve every: {model.layers[0].evolve_every} tokens")
    print()

    # Also train embeddings with simple gradient descent
    embed_optimizer = torch.optim.Adam([model.embed.weight], lr=0.01)

    for epoch in range(n_epochs):
        print(f"=== Epoch {epoch + 1}/{n_epochs} ===")
        model.reset()

        total_loss = 0
        correct = 0
        total = 0

        for i in range(len(text) - 1):
            char = text[i]
            next_char = text[i + 1]

            token_id = char_to_id.get(char, 0)
            target_id = char_to_id.get(next_char, 0)

            # Forward with fitness assignment
            logits = model(token_id, target_id)

            # Track accuracy
            pred = logits.argmax().item()
            if pred == target_id:
                correct += 1
            total += 1

            # Simple embedding update (cross-entropy gradient)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id]))
            total_loss += loss.item()

            # Gradient descent on embeddings only
            embed_optimizer.zero_grad()
            loss.backward()
            embed_optimizer.step()

            # Log progress
            if (i + 1) % log_every == 0:
                stats = model.get_evolution_stats()[0]  # First layer
                print(f"  Token {i + 1}/{len(text) - 1}")
                print(f"    Loss: {total_loss / (i + 1):.4f}")
                print(f"    Accuracy: {correct / total * 100:.1f}%")
                print(f"    Generation: {stats['generation']}")
                print(
                    f"    Mean decay: {stats['mean_decay']:.3f} (std: {stats['std_decay']:.3f})"
                )
                print(
                    f"    Mean gate_bias: {stats['mean_gate_bias']:.2f} (std: {stats['std_gate_bias']:.2f})"
                )
                print(f"    Births/Deaths: {stats['births']}/{stats['deaths']}")

        # End of epoch stats
        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Final accuracy: {correct / total * 100:.1f}%")
        print(f"  Final loss: {total_loss / total:.4f}")

        # Evolution summary per layer
        print(f"\n  Evolution summary:")
        for i, stats in enumerate(model.get_evolution_stats()):
            print(
                f"    Layer {i}: Gen {stats['generation']}, "
                + f"decay μ={stats['mean_decay']:.3f}, "
                + f"gate_bias μ={stats['mean_gate_bias']:.2f}"
            )
        print()

    return model


def generate(
    model: EvolvingSRCFFN,
    prompt: str,
    char_to_id: Dict,
    id_to_char: Dict,
    length: int = 100,
    temperature: float = 0.8,
) -> str:
    """Generate text from the evolved model."""
    model.reset()

    # Process prompt
    for char in prompt[:-1]:
        token_id = char_to_id.get(char, 0)
        _ = model(token_id)

    # Generate
    result = prompt
    current_char = prompt[-1]

    for _ in range(length):
        token_id = char_to_id.get(current_char, 0)
        logits = model(token_id)

        # Sample
        probs = F.softmax(logits / temperature, dim=0)
        next_id = torch.multinomial(probs, 1).item()
        next_char = id_to_char.get(next_id, "?")

        result += next_char
        current_char = next_char

    return result


# =============================================================================
# Demo: Watch Evolution Happen
# =============================================================================


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         EVOLVING SRC-FFN: NEURAL DARWINISM                   ║")
    print("║                                                              ║")
    print("║   Each neuron is an individual.                              ║")
    print("║   Fitness determines survival.                               ║")
    print("║   The fittest reproduce.                                     ║")
    print("║   The population evolves.                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Simple dataset: repeating pattern
    text = "hello world " * 100

    # Build vocabulary
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for c, i in char_to_id.items()}

    print(f"Vocabulary: {chars}")
    print(f"Vocab size: {len(chars)}")
    print()

    # Create model
    model = EvolvingSRCFFN(
        vocab_size=len(chars),
        embed_dim=32,
        n_layers=2,
        n_neurons_per_layer=64,
        evolve_every=20,
        selection_pressure=0.5,
        mutation_rate=0.2,
        mutation_strength=0.15,
        sexual_reproduction_rate=0.3,
    )

    print(f"Model created:")
    print(f"  Embed dim: {model.embed_dim}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Neurons per layer: 64")
    print(f"  Total neurons: {64 * 2}")
    print()

    # Train
    model = train_evolving_model(
        model, text, char_to_id, id_to_char, n_epochs=3, log_every=200
    )

    # Generate
    print("\n" + "=" * 60)
    print("GENERATION FROM EVOLVED MODEL")
    print("=" * 60)

    for temp in [0.5, 0.8, 1.2]:
        print(f"\nTemperature {temp}:")
        output = generate(
            model, "hello", char_to_id, id_to_char, length=50, temperature=temp
        )
        print(f"  '{output}'")

    # Analyze evolved population
    print("\n" + "=" * 60)
    print("EVOLVED NEURON ANALYSIS")
    print("=" * 60)

    for layer_idx, layer in enumerate(model.layers):
        pop = layer.population
        print(f"\nLayer {layer_idx}:")

        # Sort by fitness
        neurons = sorted(pop.neurons, key=lambda n: n.fitness, reverse=True)

        print(f"  Top 5 neurons:")
        for i, n in enumerate(neurons[:5]):
            print(
                f"    #{i + 1}: decay={n.genome.decay:.3f}, "
                + f"gate_bias={n.genome.gate_bias:.2f}, "
                + f"fitness={n.fitness:.4f}, "
                + f"age={n.age}, "
                + f"offspring={n.times_reproduced}"
            )

        print(f"  Bottom 5 neurons:")
        for i, n in enumerate(neurons[-5:]):
            print(
                f"    #{len(neurons) - 4 + i}: decay={n.genome.decay:.3f}, "
                + f"gate_bias={n.genome.gate_bias:.2f}, "
                + f"fitness={n.fitness:.4f}, "
                + f"age={n.age}"
            )

        # Decay distribution
        decays = [n.genome.decay for n in neurons]
        print(
            f"  Decay distribution: min={min(decays):.3f}, max={max(decays):.3f}, "
            + f"mean={np.mean(decays):.3f}, std={np.std(decays):.3f}"
        )

        # Gate bias distribution
        biases = [n.genome.gate_bias for n in neurons]
        print(
            f"  Gate bias distribution: min={min(biases):.2f}, max={max(biases):.2f}, "
            + f"mean={np.mean(biases):.2f}, std={np.std(biases):.2f}"
        )

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print("The neurons have evolved.")
    print("The fittest survived.")
    print("This is neural Darwinism.")
    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    main()
