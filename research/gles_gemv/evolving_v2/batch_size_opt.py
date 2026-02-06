#!/usr/bin/env python3
"""
Batch Size Optimization Test

Test different batch sizes to find optimal training configuration.
Each test runs 10k tokens and tracks convergence curves.
"""

import torch
import torch.nn.functional as F
import pickle
import time
import json
import gc
from pathlib import Path
from breathing_model import BreathingModel, BreathingModelConfig


def train_with_batch_size(batch_size, tokens, log_every=1000):
    """Train model with specific batch size."""
    print(f"\n{'=' * 70}")
    print(f"Testing batch_size={batch_size}")
    print(f"{'=' * 70}")

    # Create model
    config = BreathingModelConfig(
        vocab_size=50257,
        embed_dim=128,
        n_layers=4,
        n_neurons_per_layer=128,
        n_breathe=5,
        evolve_every=1000,
        decay_min=0.5,
        decay_max=0.9,
    )
    model = BreathingModel(config)

    # Adjust learning rate for batch size (linear scaling rule)
    # Base LR 1e-3 for batch_size=1
    lr = 1e-3 * batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_tokens = len(tokens) - 1
    n_batches = n_tokens // batch_size

    # Metrics
    history = {
        "batch_size": batch_size,
        "lr": lr,
        "checkpoints": [],
    }

    start_time = time.time()
    total_loss = 0
    total_correct = 0
    tokens_processed = 0

    print(f"  LR: {lr:.4f} (scaled for batch_size)")
    print(f"  Total batches: {n_batches}")
    print(f"  Training...")

    for batch_idx in range(n_batches):
        model.train()

        # Accumulate gradients over batch
        batch_loss = 0
        batch_correct = 0

        for i in range(batch_size):
            idx = batch_idx * batch_size + i
            if idx >= n_tokens:
                break

            logits = model.forward(tokens[idx])
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tokens[idx + 1]]))

            # Scale loss for gradient accumulation
            scaled_loss = loss / batch_size
            scaled_loss.backward()

            batch_loss += loss.item()
            if logits.argmax().item() == tokens[idx + 1]:
                batch_correct += 1

            # Evolution (only on first item of batch to maintain frequency)
            if i == 0:
                model.maybe_evolve()

        # Update weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Stats
        total_loss += batch_loss
        total_correct += batch_correct
        tokens_processed += batch_size

        # Logging
        if (batch_idx + 1) % (log_every // batch_size) == 0:
            avg_loss = total_loss / tokens_processed
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            acc = total_correct / tokens_processed * 100
            elapsed = time.time() - start_time
            tps = tokens_processed / elapsed

            # Window metrics (last 1000 tokens)
            window_tokens = min(tokens_processed, 1000)
            window_loss = 0  # Would need to track window separately

            history["checkpoints"].append(
                {
                    "step": tokens_processed,
                    "batch": batch_idx + 1,
                    "loss": avg_loss,
                    "ppl": ppl,
                    "acc": acc,
                    "evolutions": model.total_evolutions,
                    "elapsed": elapsed,
                    "tok_per_sec": tps,
                }
            )

            print(
                f"    {tokens_processed:>6,}/{n_tokens:,} | "
                f"Loss: {avg_loss:.3f} | PPL: {ppl:>6.0f} | "
                f"Acc: {acc:>5.1f}% | {tps:>5.0f} tok/s"
            )

    # Final stats
    final_loss = total_loss / tokens_processed
    final_ppl = torch.exp(torch.tensor(final_loss)).item()
    final_acc = total_correct / tokens_processed * 100
    total_time = time.time() - start_time

    history["final"] = {
        "loss": final_loss,
        "ppl": final_ppl,
        "acc": final_acc,
        "time": total_time,
        "tok_per_sec": tokens_processed / total_time,
    }

    print(
        f"\n  Final: PPL={final_ppl:.0f}, Acc={final_acc:.2f}%, "
        f"Time={total_time:.1f}s, Speed={tokens_processed / total_time:.0f} tok/s"
    )

    # Cleanup
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return history


def main():
    print("=" * 70)
    print("BATCH SIZE OPTIMIZATION TEST")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    with open("wikitext2_clean_tokens.pkl", "rb") as f:
        all_tokens = pickle.load(f)
    tokens = all_tokens[:10001]  # 10k tokens
    print(f"Using {len(tokens) - 1:,} tokens")

    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16, 32]
    results = {}

    Path("experiments/batch_size_opt").mkdir(parents=True, exist_ok=True)

    for bs in batch_sizes:
        try:
            history = train_with_batch_size(bs, tokens)
            results[f"batch_{bs}"] = history

            # Save incremental results
            with open(f"experiments/batch_size_opt/results_batch{bs}.json", "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"\n  ERROR with batch_size={bs}: {e}")
            results[f"batch_{bs}"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Batch Size':<12} {'LR':<8} {'Final PPL':<12} {'Final Acc':<12} {'Time (s)':<10} {'Speed (tok/s)':<15}"
    )
    print("-" * 70)

    for bs in batch_sizes:
        key = f"batch_{bs}"
        if key in results and "error" not in results[key]:
            final = results[key]["final"]
            print(
                f"{bs:<12} {results[key]['lr']:<8.4f} {final['ppl']:<12.0f} "
                f"{final['acc']:<12.2f} {final['time']:<10.1f} {final['tok_per_sec']:<15.0f}"
            )

    # Find best
    valid_results = [
        (bs, results[f"batch_{bs}"])
        for bs in batch_sizes
        if f"batch_{bs}" in results and "error" not in results[f"batch_{bs}"]
    ]

    if valid_results:
        best_ppl = min(valid_results, key=lambda x: x[1]["final"]["ppl"])
        best_acc = max(valid_results, key=lambda x: x[1]["final"]["acc"])
        best_speed = max(valid_results, key=lambda x: x[1]["final"]["tok_per_sec"])

        print("\n" + "-" * 70)
        print("BEST CONFIGURATIONS:")
        print(
            f"  Best PPL:    batch_size={best_ppl[0]} (PPL={best_ppl[1]['final']['ppl']:.0f})"
        )
        print(
            f"  Best Acc:    batch_size={best_acc[0]} (Acc={best_acc[1]['final']['acc']:.2f}%)"
        )
        print(
            f"  Best Speed:  batch_size={best_speed[0]} ({best_speed[1]['final']['tok_per_sec']:.0f} tok/s)"
        )

        # Balanced recommendation
        print("\n  RECOMMENDATION:")
        # Score each config (lower is better)
        scores = {}
        for bs, hist in valid_results:
            ppl_score = hist["final"]["ppl"] / best_ppl[1]["final"]["ppl"]
            acc_score = best_acc[1]["final"]["acc"] / hist["final"]["acc"]  # Inverted
            speed_score = (
                best_speed[1]["final"]["tok_per_sec"] / hist["final"]["tok_per_sec"]
            )  # Inverted
            scores[bs] = ppl_score + acc_score + speed_score

        best_balanced = min(scores.items(), key=lambda x: x[1])
        print(
            f"    Balanced best: batch_size={best_balanced[0]} (score={best_balanced[1]:.2f})"
        )

    # Save all results
    with open("experiments/batch_size_opt/all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to experiments/batch_size_opt/")
    print("=" * 70)


if __name__ == "__main__":
    main()
