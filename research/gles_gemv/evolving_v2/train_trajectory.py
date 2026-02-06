"""
Trajectory Analysis Training Script

Tests the Delta Observer hypothesis: Is hierarchy transient scaffolding or stable structure?

Captures hierarchy metrics at EVERY evolution step to track:
- Does hierarchy strength rise then fall? (scaffolding pattern)
- Or does it rise and stabilize? (stable structure)
"""

import torch
import torch.nn.functional as F
import json
import time
import os
import pickle
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict

from model import EvolvingModel, ModelConfig


@dataclass
class TrainConfig:
    """Training configuration for trajectory experiment."""

    max_tokens: int = 50000  # Shorter run, more frequent evolution
    n_epochs: int = 1
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    log_every: int = 5000  # Less frequent logging
    save_trajectory_every: int = 100  # Save trajectory JSON every N evolutions
    output_dir: str = "experiments/trajectory_analysis"


def load_cached_tokens(max_tokens: int = None) -> List[int]:
    """Load pre-cached WikiText-2 tokens."""
    cache_path = "wikitext2_tokens.pkl"

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache not found at {cache_path}. Run prepare_wikitext.py first."
        )

    with open(cache_path, "rb") as f:
        tokens = pickle.load(f)

    if max_tokens:
        tokens = tokens[:max_tokens]

    print(f"Loaded {len(tokens):,} tokens from cache")
    return tokens


def train_with_trajectory(
    model: EvolvingModel,
    tokens: List[int],
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> Dict:
    """Train while capturing full evolution trajectory."""

    # Enable trajectory tracking
    model.set_trajectory_tracking(True)
    model.reset_state()

    # Also track accuracy at regular intervals
    accuracy_log = {
        "tokens": [],
        "accuracy": [],
        "loss": [],
        "generation": [],
    }

    total_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()
    last_trajectory_save = 0

    print("\nStarting trajectory training...")
    print("=" * 60)

    for i in range(len(tokens) - 1):
        token_id = tokens[i]
        target_id = tokens[i + 1]

        # Forward with fitness tracking
        logits = model.forward(token_id, target_id=target_id)

        # Loss for embedding gradient
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id]))

        # Backward and optimize embeddings
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        pred = logits.argmax().item()
        if pred == target_id:
            correct += 1
        total += 1

        # Maybe evolve (this captures trajectory internally)
        model.maybe_evolve()

        # Logging
        if (i + 1) % config.log_every == 0:
            acc = correct / total * 100
            avg_loss = total_loss / total
            tokens_per_sec = total / (time.time() - epoch_start)

            # Record accuracy point
            accuracy_log["tokens"].append(i + 1)
            accuracy_log["accuracy"].append(acc)
            accuracy_log["loss"].append(avg_loss)
            accuracy_log["generation"].append(model.total_evolutions)

            print(
                f"Token {i + 1:,}/{len(tokens) - 1:,} | "
                f"Loss: {avg_loss:.3f} | Acc: {acc:.2f}% | "
                f"Gen: {model.total_evolutions} | "
                f"Decay spread: {model.trajectory['decay_spread'][-1] if model.trajectory['decay_spread'] else 0:.3f} | "
                f"tok/s: {tokens_per_sec:.1f}"
            )

        # Save trajectory periodically
        if (
            model.total_evolutions
            >= last_trajectory_save + config.save_trajectory_every
        ):
            last_trajectory_save = model.total_evolutions
            interim_path = os.path.join(
                config.output_dir, f"trajectory_gen{model.total_evolutions}.json"
            )
            save_trajectory(model.trajectory, accuracy_log, interim_path)
            print(f"  [Saved trajectory at generation {model.total_evolutions}]")

    final_acc = correct / total * 100
    final_loss = total_loss / total
    train_time = time.time() - epoch_start

    print("=" * 60)
    print(f"Training complete: {train_time:.1f}s")
    print(f"Final: Loss={final_loss:.3f}, Acc={final_acc:.2f}%")
    print(f"Total evolutions: {model.total_evolutions}")
    print(f"Trajectory points: {len(model.trajectory['generations'])}")

    return {
        "trajectory": model.trajectory,
        "accuracy_log": accuracy_log,
        "final_acc": final_acc,
        "final_loss": final_loss,
        "total_evolutions": model.total_evolutions,
        "train_time": train_time,
    }


def save_trajectory(trajectory: Dict, accuracy_log: Dict, path: str):
    """Save trajectory data to JSON."""
    data = {
        "trajectory": trajectory,
        "accuracy_log": accuracy_log,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def analyze_trajectory(trajectory: Dict) -> Dict:
    """Analyze trajectory for scaffolding vs stable structure pattern."""

    generations = trajectory["generations"]
    decay_spread = trajectory["decay_spread"]
    diversity = trajectory["diversity"]

    if len(generations) < 10:
        return {"error": "Not enough data points"}

    # Split into thirds
    n = len(generations)
    third = n // 3

    early_spread = sum(decay_spread[:third]) / third
    mid_spread = sum(decay_spread[third : 2 * third]) / third
    late_spread = sum(decay_spread[2 * third :]) / (n - 2 * third)

    early_div = sum(diversity[:third]) / third
    mid_div = sum(diversity[third : 2 * third]) / third
    late_div = sum(diversity[2 * third :]) / (n - 2 * third)

    # Determine pattern
    if mid_spread > early_spread and mid_spread > late_spread:
        pattern = "SCAFFOLDING"
        description = "Hierarchy rises then falls (like Delta Observer clustering)"
    elif late_spread > mid_spread > early_spread:
        pattern = "STABLE_STRUCTURE"
        description = "Hierarchy continues to strengthen"
    elif early_spread > mid_spread and early_spread > late_spread:
        pattern = "EARLY_COLLAPSE"
        description = "Initial hierarchy weakens over time"
    else:
        pattern = "MIXED"
        description = "No clear pattern"

    return {
        "pattern": pattern,
        "description": description,
        "early_spread": early_spread,
        "mid_spread": mid_spread,
        "late_spread": late_spread,
        "early_diversity": early_div,
        "mid_diversity": mid_div,
        "late_diversity": late_div,
        "max_spread": max(decay_spread),
        "max_spread_gen": generations[decay_spread.index(max(decay_spread))],
        "final_spread": decay_spread[-1],
    }


def print_trajectory_analysis(analysis: Dict):
    """Print trajectory analysis results."""
    print("\n" + "=" * 60)
    print("TRAJECTORY ANALYSIS - DELTA OBSERVER HYPOTHESIS")
    print("=" * 60)

    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    print(f"\nPattern detected: {analysis['pattern']}")
    print(f"  {analysis['description']}")

    print(f"\nHierarchy strength (decay spread) by phase:")
    print(f"  Early:  {analysis['early_spread']:.4f}")
    print(f"  Middle: {analysis['mid_spread']:.4f}")
    print(f"  Late:   {analysis['late_spread']:.4f}")

    print(f"\nDiversity by phase:")
    print(f"  Early:  {analysis['early_diversity']:.4f}")
    print(f"  Middle: {analysis['mid_diversity']:.4f}")
    print(f"  Late:   {analysis['late_diversity']:.4f}")

    print(f"\nPeak hierarchy:")
    print(
        f"  Max spread: {analysis['max_spread']:.4f} at generation {analysis['max_spread_gen']}"
    )
    print(f"  Final spread: {analysis['final_spread']:.4f}")

    # Interpretation
    print("\n" + "-" * 40)
    if analysis["pattern"] == "SCAFFOLDING":
        print("INTERPRETATION: Hierarchy is SCAFFOLDING")
        print("  - Like clustering in Delta Observer, hierarchy emerges")
        print("    to help learning then dissolves when encoded in weights")
        print("  - The structure we see mid-training may not persist")
    elif analysis["pattern"] == "STABLE_STRUCTURE":
        print("INTERPRETATION: Hierarchy is STABLE STRUCTURE")
        print("  - Unlike Delta Observer clustering, hierarchy persists")
        print("  - Evolution finds a stable temporal organization")
        print("  - The structure we see is the final architecture")
    else:
        print(f"INTERPRETATION: {analysis['pattern']}")
        print("  - Pattern doesn't clearly match either hypothesis")
        print("  - May need longer training or different analysis")


def main():
    torch.manual_seed(42)

    print("=" * 60)
    print("TRAJECTORY ANALYSIS EXPERIMENT")
    print("Testing: Is hierarchy scaffolding or stable structure?")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Config - more frequent evolution for denser trajectory
    train_config = TrainConfig(
        max_tokens=50000,  # 50k tokens
        log_every=5000,
        save_trajectory_every=50,
    )

    model_config = ModelConfig(
        vocab_size=50257,  # GPT-2
        embed_dim=128,
        n_layers=8,
        n_neurons_per_layer=256,
        evolve_every=50,  # More frequent evolution = more trajectory points
    )

    # Load data
    tokens = load_cached_tokens(train_config.max_tokens)

    # Create output directory
    os.makedirs(train_config.output_dir, exist_ok=True)

    print(f"\nModel Configuration:")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Neurons/layer: {model_config.n_neurons_per_layer}")
    print(f"  Evolve every: {model_config.evolve_every} tokens")
    print(
        f"  Expected evolutions: ~{train_config.max_tokens // model_config.evolve_every}"
    )
    print()

    # Create model
    model = EvolvingModel(model_config)

    # Optimizer for embeddings only
    optimizer = torch.optim.AdamW(
        [
            {"params": model.embed.parameters()},
            {"params": model.out_norm.parameters()},
        ],
        lr=train_config.learning_rate,
    )

    # Train with trajectory tracking
    results = train_with_trajectory(model, tokens, optimizer, train_config)

    # Analyze trajectory
    analysis = analyze_trajectory(results["trajectory"])
    print_trajectory_analysis(analysis)

    # Save final results
    final_results = {
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "trajectory": results["trajectory"],
        "accuracy_log": results["accuracy_log"],
        "analysis": analysis,
        "final_acc": results["final_acc"],
        "final_loss": results["final_loss"],
        "total_evolutions": results["total_evolutions"],
        "train_time": results["train_time"],
    }

    results_path = os.path.join(train_config.output_dir, "trajectory_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved results: {results_path}")

    # Save model
    model_path = os.path.join(train_config.output_dir, "model_final.pt")
    model.save_checkpoint(model_path)
    print(f"Saved model: {model_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
