"""
Breathing Language Model

FFN and Equilibrium work as one system with a breathing rhythm.
Each token: INHALE (FFN) -> BREATHE (equilibrate) -> EXHALE (output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List

from breathing_layer import BreathingLayer, BreathingConfig


@dataclass
class BreathingModelConfig:
    """Configuration for breathing model."""

    vocab_size: int = 50257
    embed_dim: int = 128
    n_layers: int = 8
    n_neurons_per_layer: int = 256

    # Breathing rhythm
    n_breathe: int = 5  # Steps per token

    # Evolution
    evolve_every: int = 100
    selection_pressure: float = 0.5
    elite_fraction: float = 0.1
    mutation_rate: float = 0.5
    mutation_strength: float = 0.2

    # Dynamics - tuned for fast equilibrium
    decay_min: float = 0.5
    decay_max: float = 0.9
    gate_bias_min: float = -1.0
    gate_bias_max: float = 1.0


class BreathingModel(nn.Module):
    """Breathing language model."""

    def __init__(self, config: BreathingModelConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

        # Breathing layers
        layer_config = BreathingConfig(
            n_neurons=config.n_neurons_per_layer,
            embed_dim=config.embed_dim,
            n_breathe=config.n_breathe,
            decay_min=config.decay_min,
            decay_max=config.decay_max,
            gate_bias_min=config.gate_bias_min,
            gate_bias_max=config.gate_bias_max,
        )

        self.layers = nn.ModuleList(
            [BreathingLayer(layer_config, layer_idx=i) for i in range(config.n_layers)]
        )

        # Output
        self.out_norm = nn.LayerNorm(config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight

        # State
        self.tokens_seen = 0
        self.total_evolutions = 0

        # Trajectory
        self.trajectory = {"generations": [], "decay_spread": [], "mean_fitness": []}
        self.track_trajectory = False

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()

    def set_trajectory_tracking(self, enabled: bool = True):
        self.track_trajectory = enabled
        if enabled:
            self.trajectory = {
                "generations": [],
                "decay_spread": [],
                "mean_fitness": [],
            }

    def forward(self, token_id: int) -> torch.Tensor:
        x = self.embed.weight[token_id]

        for layer in self.layers:
            out = layer(x)
            x = x + out

        x = self.out_norm(x)
        logits = self.out_proj(x)

        self.tokens_seen += 1
        return logits

    def maybe_evolve(self):
        if self.tokens_seen > 0 and self.tokens_seen % self.config.evolve_every == 0:
            self._evolve()

    def _evolve(self):
        # Capture fitness before evolution
        fitnesses = []
        for layer in self.layers:
            layer.population.finalize_fitness()
            fitnesses.append(layer.population.fitness.mean().item())

        for layer in self.layers:
            layer.evolve(
                selection_pressure=self.config.selection_pressure,
                elite_fraction=self.config.elite_fraction,
                mutation_rate=self.config.mutation_rate,
                mutation_strength=self.config.mutation_strength,
            )

        self.total_evolutions += 1

        if self.track_trajectory:
            decays = [layer.population.decay.mean().item() for layer in self.layers]
            self.trajectory["generations"].append(self.total_evolutions)
            self.trajectory["decay_spread"].append(max(decays) - min(decays))
            self.trajectory["mean_fitness"].append(sum(fitnesses) / len(fitnesses))

    def get_hierarchy_summary(self) -> Dict:
        decays = [layer.population.decay.mean().item() for layer in self.layers]
        gates = [layer.population.gate_bias.mean().item() for layer in self.layers]
        return {
            "decays_by_layer": decays,
            "gate_biases_by_layer": gates,
            "decay_spread": max(decays) - min(decays),
        }

    def save_checkpoint(self, path: str):
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
    def load_checkpoint(cls, path: str) -> "BreathingModel":
        ckpt = torch.load(path, weights_only=False)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        model.tokens_seen = ckpt["tokens_seen"]
        model.total_evolutions = ckpt["total_evolutions"]
        model.trajectory = ckpt.get("trajectory", model.trajectory)
        return model


if __name__ == "__main__":
    import pickle
    import time

    print("Testing BreathingModel on WikiText-2...")

    # Load data
    with open("wikitext2_tokens.pkl", "rb") as f:
        tokens = pickle.load(f)[:10000]
    print(f"Loaded {len(tokens)} tokens")

    # Create model
    config = BreathingModelConfig(
        vocab_size=50257,
        embed_dim=128,
        n_layers=4,
        n_neurons_per_layer=128,
        n_breathe=5,
        evolve_every=100,
    )
    model = BreathingModel(config)
    model.set_trajectory_tracking(True)

    print(
        f"Model: {config.n_layers} layers, {config.n_neurons_per_layer} neurons/layer"
    )
    print(f"Breathing: {config.n_breathe} steps/token")
    print(f"Decay range: [{config.decay_min}, {config.decay_max}]")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train
    print("\nTraining...")
    start = time.time()
    total_loss = 0
    correct = 0

    for i in range(len(tokens) - 1):
        logits = model.forward(tokens[i])
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tokens[i + 1]]))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if logits.argmax().item() == tokens[i + 1]:
            correct += 1

        model.maybe_evolve()

        if (i + 1) % 2000 == 0:
            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1) * 100
            ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 50000)
            h = model.get_hierarchy_summary()
            tps = (i + 1) / (time.time() - start)
            print(
                f"Token {i + 1}: Loss={avg_loss:.3f}, PPL={ppl:.0f}, Acc={acc:.2f}%, "
                f"DecaySpread={h['decay_spread']:.3f}, tok/s={tps:.0f}"
            )

    elapsed = time.time() - start
    final_loss = total_loss / (len(tokens) - 1)
    final_ppl = torch.exp(torch.tensor(final_loss)).item()
    final_acc = correct / (len(tokens) - 1) * 100

    print(f"\nDone in {elapsed:.1f}s ({len(tokens) / elapsed:.0f} tok/s)")
    print(f"Final: Loss={final_loss:.3f}, PPL={final_ppl:.0f}, Acc={final_acc:.2f}%")

    # Hierarchy
    h = model.get_hierarchy_summary()
    print(f"\nHierarchy:")
    print(f"  Decays: {[f'{d:.3f}' for d in h['decays_by_layer']]}")
    print(f"  Gates:  {[f'{g:.3f}' for g in h['gate_biases_by_layer']]}")

    # Trajectory
    if model.trajectory["mean_fitness"]:
        print(f"\nTrajectory:")
        print(
            f"  Fitness: {model.trajectory['mean_fitness'][:3]} ... {model.trajectory['mean_fitness'][-3:]}"
        )
        print(
            f"  Decay spread: {model.trajectory['decay_spread'][:3]} ... {model.trajectory['decay_spread'][-3:]}"
        )

    # Quick generation test
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def generate(prompt, max_tokens=20):
        model.reset_state()
        toks = tokenizer.encode(prompt)
        for t in toks[:-1]:
            model.forward(t)
        for _ in range(max_tokens):
            logits = model.forward(toks[-1])
            probs = F.softmax(logits / 0.8, dim=0)
            toks.append(torch.multinomial(probs, 1).item())
        return tokenizer.decode(toks)

    print("\nGeneration:")
    for p in ["The", "In the", "He said"]:
        print(f"  '{p}' -> {generate(p)}")

    print("\nBreathingModel test complete!")
