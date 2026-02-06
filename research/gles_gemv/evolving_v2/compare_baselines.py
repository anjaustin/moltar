"""
Baseline Comparison: Breathing Model vs GRU

This script runs a fair comparison between:
1. Breathing CfC model (with niche-based evolution)
2. GRU baseline (parameter-matched)

Both models are trained on the same WikiText-2 data for the same number of tokens.
"""

import torch
import torch.nn.functional as F
import pickle
import time
import json
from dataclasses import asdict

from breathing_model import BreathingModel, BreathingModelConfig
from gru_baseline import GRUBaseline, GRUBaselineConfig


def train_model(model, tokens, model_name: str, log_every: int = 1000):
    """Train a model and return metrics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = {
        "model": model_name,
        "n_params": sum(p.numel() for p in model.parameters()),
        "loss_history": [],
        "ppl_history": [],
        "acc_history": [],
        "tokens_per_step": [],
    }

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

        # Evolution for Breathing model
        if hasattr(model, "maybe_evolve"):
            model.maybe_evolve()

        if (i + 1) % log_every == 0:
            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1) * 100
            ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 50000)
            tps = (i + 1) / (time.time() - start)

            metrics["loss_history"].append(avg_loss)
            metrics["ppl_history"].append(ppl)
            metrics["acc_history"].append(acc)
            metrics["tokens_per_step"].append(i + 1)

            print(
                f"  [{model_name}] Token {i + 1}: Loss={avg_loss:.3f}, PPL={ppl:.0f}, Acc={acc:.2f}%, tok/s={tps:.0f}"
            )

    elapsed = time.time() - start
    final_loss = total_loss / (len(tokens) - 1)
    final_ppl = torch.exp(torch.tensor(final_loss)).item()
    final_acc = correct / (len(tokens) - 1) * 100

    metrics["final_loss"] = final_loss
    metrics["final_ppl"] = final_ppl
    metrics["final_acc"] = final_acc
    metrics["elapsed_time"] = elapsed
    metrics["tokens_per_second"] = len(tokens) / elapsed

    return metrics


def main():
    print("=" * 70)
    print("BASELINE COMPARISON: Breathing Model vs GRU")
    print("=" * 70)

    # Configuration
    N_TOKENS = 10000  # Training tokens
    VOCAB_SIZE = 50257
    EMBED_DIM = 128

    # Load data
    print("\nLoading WikiText-2...")
    with open("wikitext2_tokens.pkl", "rb") as f:
        tokens = pickle.load(f)[:N_TOKENS]
    print(f"Loaded {len(tokens)} tokens")

    # Create Breathing model
    print("\n" + "-" * 70)
    print("Creating Breathing Model...")
    breathing_config = BreathingModelConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        n_layers=4,
        n_neurons_per_layer=128,
        n_breathe=5,
        evolve_every=200,  # Evolve every 200 tokens
        decay_min=0.5,
        decay_max=0.9,
    )
    breathing_model = BreathingModel(breathing_config)
    breathing_params = sum(p.numel() for p in breathing_model.parameters())
    print(f"  Layers: {breathing_config.n_layers}")
    print(f"  Neurons/layer: {breathing_config.n_neurons_per_layer}")
    print(f"  Breathing steps: {breathing_config.n_breathe}")
    print(f"  Parameters: {breathing_params:,}")

    # Create GRU baseline (parameter-matched)
    print("\n" + "-" * 70)
    print("Creating GRU Baseline (parameter-matched)...")

    # Find best GRU config
    best_gru_config = None
    best_diff = float("inf")
    for hidden_dim in range(64, 512, 16):
        for n_layers in [1, 2, 3]:
            config = GRUBaselineConfig(
                vocab_size=VOCAB_SIZE,
                embed_dim=EMBED_DIM,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
            )
            model = GRUBaseline(config)
            params = sum(p.numel() for p in model.parameters())
            diff = abs(params - breathing_params)
            if diff < best_diff:
                best_diff = diff
                best_gru_config = config
                best_gru_params = params

    gru_model = GRUBaseline(best_gru_config)
    print(f"  Hidden dim: {best_gru_config.hidden_dim}")
    print(f"  Layers: {best_gru_config.n_layers}")
    print(f"  Parameters: {best_gru_params:,}")
    print(
        f"  Parameter diff: {best_diff:,} ({100 * best_diff / breathing_params:.1f}%)"
    )

    # Train both
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    print("\nTraining Breathing Model...")
    breathing_metrics = train_model(
        breathing_model, tokens, "Breathing", log_every=2000
    )

    print("\nTraining GRU Baseline...")
    gru_metrics = train_model(gru_model, tokens, "GRU", log_every=2000)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Breathing':>15} {'GRU':>15} {'Winner':>10}")
    print("-" * 70)

    # Final metrics comparison
    metrics_to_compare = [
        (
            "Parameters",
            f"{breathing_metrics['n_params']:,}",
            f"{gru_metrics['n_params']:,}",
            "tie",
        ),
        (
            "Final Loss",
            f"{breathing_metrics['final_loss']:.3f}",
            f"{gru_metrics['final_loss']:.3f}",
            "Breathing"
            if breathing_metrics["final_loss"] < gru_metrics["final_loss"]
            else "GRU",
        ),
        (
            "Final PPL",
            f"{breathing_metrics['final_ppl']:.0f}",
            f"{gru_metrics['final_ppl']:.0f}",
            "Breathing"
            if breathing_metrics["final_ppl"] < gru_metrics["final_ppl"]
            else "GRU",
        ),
        (
            "Final Accuracy",
            f"{breathing_metrics['final_acc']:.2f}%",
            f"{gru_metrics['final_acc']:.2f}%",
            "Breathing"
            if breathing_metrics["final_acc"] > gru_metrics["final_acc"]
            else "GRU",
        ),
        (
            "Time (s)",
            f"{breathing_metrics['elapsed_time']:.1f}",
            f"{gru_metrics['elapsed_time']:.1f}",
            "Breathing"
            if breathing_metrics["elapsed_time"] < gru_metrics["elapsed_time"]
            else "GRU",
        ),
        (
            "Speed (tok/s)",
            f"{breathing_metrics['tokens_per_second']:.0f}",
            f"{gru_metrics['tokens_per_second']:.0f}",
            "Breathing"
            if breathing_metrics["tokens_per_second"] > gru_metrics["tokens_per_second"]
            else "GRU",
        ),
    ]

    for name, breathing_val, gru_val, winner in metrics_to_compare:
        print(f"{name:<25} {breathing_val:>15} {gru_val:>15} {winner:>10}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ppl_improvement = (
        (gru_metrics["final_ppl"] - breathing_metrics["final_ppl"])
        / gru_metrics["final_ppl"]
        * 100
    )
    speed_ratio = (
        breathing_metrics["tokens_per_second"] / gru_metrics["tokens_per_second"]
    )

    if breathing_metrics["final_ppl"] < gru_metrics["final_ppl"]:
        print(
            f"\nBreathing model achieves {abs(ppl_improvement):.1f}% lower perplexity than GRU."
        )
    else:
        print(
            f"\nGRU baseline achieves {abs(ppl_improvement):.1f}% lower perplexity than Breathing model."
        )

    print(
        f"Speed ratio: Breathing is {speed_ratio:.2f}x {'faster' if speed_ratio > 1 else 'slower'} than GRU."
    )

    # Breathing model specific metrics
    if hasattr(breathing_model, "get_hierarchy_summary"):
        h = breathing_model.get_hierarchy_summary()
        print(f"\nBreathing model hierarchy:")
        print(f"  Decay spread: {h['decay_spread']:.3f}")
        print(f"  Evolutions: {breathing_model.total_evolutions}")

        # Kuramoto check
        r_values = [
            layer.population.kuramoto_order_parameter(t=50.0)
            for layer in breathing_model.layers
        ]
        print(f"  Kuramoto r (per layer): {[f'{r:.3f}' for r in r_values]}")

    # Save results
    results = {
        "breathing": breathing_metrics,
        "gru": gru_metrics,
        "config": {
            "n_tokens": N_TOKENS,
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
        },
    }

    with open("experiments/baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to experiments/baseline_comparison.json")

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
