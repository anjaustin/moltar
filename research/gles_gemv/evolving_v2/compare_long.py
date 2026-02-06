"""
Long Training Comparison: Breathing Model vs GRU (100k tokens)

Uses cleaned WikiText-2 data for better signal.
"""

import torch
import torch.nn.functional as F
import pickle
import time
import json
from pathlib import Path

from breathing_model import BreathingModel, BreathingModelConfig
from gru_baseline import GRUBaseline, GRUBaselineConfig


def train_model(
    model, tokens, model_name: str, log_every: int = 10000, save_every: int = 25000
):
    """Train a model and return metrics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Learning rate warmup + decay
    def get_lr(step, warmup=1000, total=100000):
        if step < warmup:
            return 1e-3 * step / warmup
        # Cosine decay
        progress = (step - warmup) / (total - warmup)
        return 1e-3 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    metrics = {
        "model": model_name,
        "n_params": sum(p.numel() for p in model.parameters()),
        "checkpoints": [],
    }

    start = time.time()
    total_loss = 0
    correct = 0

    # For windowed metrics
    window_loss = 0
    window_correct = 0
    window_start = 0

    n_tokens = len(tokens) - 1
    for i in range(n_tokens):
        # LR schedule
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        logits = model.forward(tokens[i])
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tokens[i + 1]]))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        window_loss += loss.item()
        if logits.argmax().item() == tokens[i + 1]:
            correct += 1
            window_correct += 1

        # Evolution for Breathing model
        if hasattr(model, "maybe_evolve"):
            model.maybe_evolve()

        if (i + 1) % log_every == 0:
            # Cumulative metrics
            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1) * 100
            ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 50000)

            # Window metrics (last log_every tokens)
            window_size = i + 1 - window_start
            window_avg_loss = window_loss / window_size
            window_acc = window_correct / window_size * 100
            window_ppl = min(torch.exp(torch.tensor(window_avg_loss)).item(), 50000)

            tps = (i + 1) / (time.time() - start)
            eta = (n_tokens - i - 1) / tps / 60  # minutes remaining

            checkpoint = {
                "step": i + 1,
                "cumulative_loss": avg_loss,
                "cumulative_ppl": ppl,
                "cumulative_acc": acc,
                "window_loss": window_avg_loss,
                "window_ppl": window_ppl,
                "window_acc": window_acc,
                "lr": lr,
                "tok_per_sec": tps,
            }

            # Add breathing-specific metrics
            if hasattr(model, "total_evolutions"):
                checkpoint["evolutions"] = model.total_evolutions
                h = model.get_hierarchy_summary()
                checkpoint["decay_spread"] = h["decay_spread"]
                # Kuramoto
                r_values = [
                    layer.population.kuramoto_order_parameter(t=50.0)
                    for layer in model.layers
                ]
                checkpoint["kuramoto_mean"] = sum(r_values) / len(r_values)

            metrics["checkpoints"].append(checkpoint)

            print(
                f"  [{model_name}] {i + 1:,}/{n_tokens:,} | "
                f"Loss={window_avg_loss:.3f} PPL={window_ppl:.0f} Acc={window_acc:.1f}% | "
                f"Cumul: PPL={ppl:.0f} Acc={acc:.1f}% | "
                f"{tps:.0f} tok/s, ETA {eta:.1f}m"
            )

            # Reset window
            window_loss = 0
            window_correct = 0
            window_start = i + 1

        # Save checkpoint
        if save_every and (i + 1) % save_every == 0:
            ckpt_path = f"experiments/long_{model_name.lower()}_step{i + 1}.pt"
            if hasattr(model, "save_checkpoint"):
                model.save_checkpoint(ckpt_path)
            else:
                torch.save(model.state_dict(), ckpt_path)
            print(f"    Saved checkpoint: {ckpt_path}")

    elapsed = time.time() - start
    final_loss = total_loss / n_tokens
    final_ppl = torch.exp(torch.tensor(final_loss)).item()
    final_acc = correct / n_tokens * 100

    metrics["final_loss"] = final_loss
    metrics["final_ppl"] = final_ppl
    metrics["final_acc"] = final_acc
    metrics["elapsed_time"] = elapsed
    metrics["tokens_per_second"] = n_tokens / elapsed
    metrics["total_tokens"] = n_tokens

    return metrics


def main():
    print("=" * 70)
    print("LONG TRAINING: Breathing Model vs GRU (100k tokens, clean data)")
    print("=" * 70)

    # Configuration
    N_TOKENS = 20000  # 20k for quick comparison (~7 min total)
    VOCAB_SIZE = 50257
    EMBED_DIM = 128

    Path("experiments").mkdir(exist_ok=True)

    # Load cleaned data
    print("\nLoading cleaned WikiText-2...")
    with open("wikitext2_clean_tokens.pkl", "rb") as f:
        all_tokens = pickle.load(f)
    tokens = all_tokens[: N_TOKENS + 1]  # +1 for target
    print(f"Using {len(tokens) - 1:,} tokens for training")

    # Create Breathing model
    print("\n" + "-" * 70)
    print("Creating Breathing Model...")
    breathing_config = BreathingModelConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        n_layers=4,
        n_neurons_per_layer=128,
        n_breathe=5,
        evolve_every=500,  # Less frequent evolution for longer training
        decay_min=0.5,
        decay_max=0.9,
    )
    breathing_model = BreathingModel(breathing_config)
    breathing_params = sum(p.numel() for p in breathing_model.parameters())
    print(f"  Parameters: {breathing_params:,}")

    # Create GRU baseline
    print("\nCreating GRU Baseline...")
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

    gru_model = GRUBaseline(best_gru_config)
    gru_params = sum(p.numel() for p in gru_model.parameters())
    print(
        f"  Parameters: {gru_params:,} (diff: {abs(gru_params - breathing_params):,})"
    )

    # Train both
    print("\n" + "=" * 70)
    print("TRAINING (this will take ~15-20 minutes)")
    print("=" * 70)

    print("\n[1/2] Training Breathing Model...")
    breathing_metrics = train_model(
        breathing_model, tokens, "Breathing", log_every=5000, save_every=None
    )

    print("\n[2/2] Training GRU Baseline...")
    gru_metrics = train_model(gru_model, tokens, "GRU", log_every=5000, save_every=None)

    # Results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Breathing':>15} {'GRU':>15} {'Delta':>12}")
    print("-" * 70)

    metrics_to_compare = [
        ("Parameters", breathing_metrics["n_params"], gru_metrics["n_params"]),
        ("Final Loss", breathing_metrics["final_loss"], gru_metrics["final_loss"]),
        ("Final PPL", breathing_metrics["final_ppl"], gru_metrics["final_ppl"]),
        ("Final Accuracy %", breathing_metrics["final_acc"], gru_metrics["final_acc"]),
        ("Time (s)", breathing_metrics["elapsed_time"], gru_metrics["elapsed_time"]),
        (
            "Speed (tok/s)",
            breathing_metrics["tokens_per_second"],
            gru_metrics["tokens_per_second"],
        ),
    ]

    for name, b_val, g_val in metrics_to_compare:
        if isinstance(b_val, float):
            delta = b_val - g_val
            print(f"{name:<25} {b_val:>15.2f} {g_val:>15.2f} {delta:>+12.2f}")
        else:
            print(f"{name:<25} {b_val:>15,} {g_val:>15,}")

    # PPL improvement
    ppl_improvement = (
        (gru_metrics["final_ppl"] - breathing_metrics["final_ppl"])
        / gru_metrics["final_ppl"]
        * 100
    )
    print(
        f"\nPPL Improvement: {ppl_improvement:+.1f}% ({'Breathing wins' if ppl_improvement > 0 else 'GRU wins'})"
    )

    # Learning curves
    print("\n" + "-" * 70)
    print("Learning Curves (Window PPL)")
    print("-" * 70)
    print(f"{'Step':>10} {'Breathing':>12} {'GRU':>12} {'Gap':>10}")

    for b_ckpt, g_ckpt in zip(
        breathing_metrics["checkpoints"], gru_metrics["checkpoints"]
    ):
        step = b_ckpt["step"]
        b_ppl = b_ckpt["window_ppl"]
        g_ppl = g_ckpt["window_ppl"]
        gap = g_ppl - b_ppl
        print(f"{step:>10,} {b_ppl:>12.0f} {g_ppl:>12.0f} {gap:>+10.0f}")

    # Breathing-specific
    if (
        breathing_metrics["checkpoints"]
        and "kuramoto_mean" in breathing_metrics["checkpoints"][-1]
    ):
        print("\n" + "-" * 70)
        print("Breathing Model Dynamics")
        print("-" * 70)
        last = breathing_metrics["checkpoints"][-1]
        print(f"  Total evolutions: {last.get('evolutions', 'N/A')}")
        print(f"  Final decay spread: {last.get('decay_spread', 'N/A'):.4f}")
        print(f"  Final Kuramoto r (mean): {last.get('kuramoto_mean', 'N/A'):.4f}")

    # Save results
    results = {
        "breathing": breathing_metrics,
        "gru": gru_metrics,
        "config": {
            "n_tokens": N_TOKENS,
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
            "data": "wikitext2_clean",
        },
    }

    results_path = "experiments/long_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 70)
    print("Long training comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
