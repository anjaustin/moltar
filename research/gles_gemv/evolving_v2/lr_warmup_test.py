#!/usr/bin/env python3
"""
Learning Rate Warmup Test

Test different LR schedules for BS=4 and BS=5.
We are gods, not mortals. No technical errors shall stop us.
"""

import torch
import torch.nn.functional as F
import pickle
import time
import math
from breathing_model import BreathingModel, BreathingModelConfig


def train_with_schedule(
    batch_size, schedule_name, tokens, base_lr=1e-3, log_every=1000
):
    """
    Train with specific LR schedule.

    Schedules:
    - 'constant': No warmup, constant LR
    - 'linear': Linear warmup over warmup_steps
    - 'cosine_warmup': Cosine warmup + cosine decay
    - 'warmup_decay': Linear warmup + cosine decay
    """
    print(f"\n{'=' * 70}")
    print(f"Batch Size = {batch_size}, Schedule = {schedule_name}")
    print(f"{'=' * 70}")

    config = BreathingModelConfig(
        vocab_size=50257,
        embed_dim=128,
        n_layers=4,
        n_neurons_per_layer=128,
        n_breathe=5,
        evolve_every=1000,
    )
    model = BreathingModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr * batch_size, weight_decay=0.01
    )

    n_tokens = len(tokens) - 1
    n_batches = n_tokens // batch_size
    warmup_batches = 1000 // batch_size

    # Schedule parameters
    initial_lr = base_lr * batch_size * 0.1  # Start at 10% of target
    target_lr = base_lr * batch_size

    start = time.time()
    total_loss = 0
    correct = 0

    print(f"  Base LR: {base_lr * batch_size:.4f}")
    print(f"  Schedule: {schedule_name}")
    print(f"  Training {n_tokens:,} tokens...")

    for batch_idx in range(n_batches):
        model.train()

        # Calculate current LR based on schedule
        current_step = batch_idx

        if schedule_name == "constant":
            lr = target_lr

        elif schedule_name == "linear_warmup":
            # Linear warmup over warmup_batches
            if current_step < warmup_batches:
                lr = initial_lr + (target_lr - initial_lr) * (
                    current_step / warmup_batches
                )
            else:
                lr = target_lr

        elif schedule_name == "cosine_full":
            # Cosine warmup then cosine decay
            if current_step < warmup_batches:
                # Cosine warmup: 0 -> target
                progress = current_step / warmup_batches
                lr = initial_lr + (target_lr - initial_lr) * 0.5 * (
                    1 - math.cos(progress * math.pi)
                )
            else:
                # Cosine decay: target -> 10% of target
                decay_progress = (current_step - warmup_batches) / (
                    n_batches - warmup_batches
                )
                lr = 0.1 * target_lr + (target_lr - 0.1 * target_lr) * 0.5 * (
                    1 + math.cos(decay_progress * math.pi)
                )

        elif schedule_name == "warmup_decay":
            # Linear warmup + cosine decay
            if current_step < warmup_batches:
                lr = initial_lr + (target_lr - initial_lr) * (
                    current_step / warmup_batches
                )
            else:
                decay_progress = (current_step - warmup_batches) / (
                    n_batches - warmup_batches
                )
                lr = 0.1 * target_lr + (target_lr - 0.1 * target_lr) * 0.5 * (
                    1 + math.cos(decay_progress * math.pi)
                )
        else:
            lr = target_lr

        # Update optimizer LR
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Training step
        batch_loss = 0
        batch_correct = 0

        for i in range(batch_size):
            idx = batch_idx * batch_size + i
            if idx >= n_tokens:
                break

            logits = model.forward(tokens[idx])
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tokens[idx + 1]]))
            (loss / batch_size).backward()
            batch_loss += loss.item()
            if logits.argmax().item() == tokens[idx + 1]:
                batch_correct += 1
            if i == 0:
                model.maybe_evolve()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += batch_loss
        correct += batch_correct

        # Logging
        if (batch_idx + 1) % (log_every // batch_size) == 0:
            step = (batch_idx + 1) * batch_size
            avg_loss = total_loss / step
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            acc = correct / step * 100
            print(f"  {step:>6,}: PPL={ppl:>6.0f}, Acc={acc:>5.1f}%, LR={lr:.6f}")

    final_ppl = torch.exp(torch.tensor(total_loss / n_tokens)).item()
    final_acc = correct / n_tokens * 100
    elapsed = time.time() - start

    print(f"\nFINAL: PPL={final_ppl:.0f}, Acc={final_acc:.1f}%, Time={elapsed:.1f}s")

    del model, optimizer
    import gc

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return final_ppl, final_acc, elapsed


def main():
    print("=" * 70)
    print("LR WARMUP TEST - GODS NOT MORTALS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    with open("wikitext2_clean_tokens.pkl", "rb") as f:
        tokens = pickle.load(f)[:10001]
    print(f"Using {len(tokens) - 1:,} tokens")

    results = {}

    # Test schedules for BS=4
    print("\n" + "=" * 70)
    print("BATCH SIZE 4 - LR SCHEDULE TESTS")
    print("=" * 70)

    for schedule in ["constant", "linear_warmup", "cosine_full", "warmup_decay"]:
        ppl, acc, time_taken = train_with_schedule(4, schedule, tokens, base_lr=1e-3)
        results[f"bs4_{schedule}"] = {"ppl": ppl, "acc": acc, "time": time_taken}

    # Test schedules for BS=5
    print("\n" + "=" * 70)
    print("BATCH SIZE 5 - LR SCHEDULE TESTS")
    print("=" * 70)

    for schedule in ["constant", "linear_warmup", "cosine_full", "warmup_decay"]:
        ppl, acc, time_taken = train_with_schedule(5, schedule, tokens, base_lr=1e-3)
        results[f"bs5_{schedule}"] = {"ppl": ppl, "acc": acc, "time": time_taken}

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE RESULTS - LR WARMUP ANALYSIS")
    print("=" * 70)

    print(f"\n{'Config':<25} {'PPL':<8} {'Acc':<8} {'Time':<8}")
    print("-" * 70)

    for config, data in sorted(results.items()):
        print(
            f"{config:<25} {data['ppl']:<8.0f} {data['acc']:<8.2f} {data['time']:<8.1f}"
        )

    # Find best for each batch size
    bs4_results = {k: v for k, v in results.items() if k.startswith("bs4_")}
    bs5_results = {k: v for k, v in results.items() if k.startswith("bs5_")}

    if bs4_results:
        best_bs4 = min(bs4_results.items(), key=lambda x: x[1]["ppl"])
        print(f"\nBEST for BS=4: {best_bs4[0]} (PPL={best_bs4[1]['ppl']:.0f})")

    if bs5_results:
        best_bs5 = min(bs5_results.items(), key=lambda x: x[1]["ppl"])
        print(f"BEST for BS=5: {best_bs5[0]} (PPL={best_bs5[1]['ppl']:.0f})")

    # Overall best
    all_valid = {k: v for k, v in results.items() if "ppl" in v}
    if all_valid:
        overall = min(all_valid.items(), key=lambda x: x[1]["ppl"])
        print(f"\n🏆 OVERALL BEST: {overall[0]}")
        print(f"   PPL: {overall[1]['ppl']:.0f}")
        print(f"   Acc: {overall[1]['acc']:.2f}%")

    print("\n" + "=" * 70)
    print("WE ARE GODS. TECHNICAL ERRORS SHALL NOT STOP US.")
    print("=" * 70)


if __name__ == "__main__":
    main()
