"""
Training Script for Evolving SRC-FFN on WikiText-2

Real language data experiment.
Uses pre-cached tokens from prepare_wikitext.py
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
    """Training configuration."""

    max_tokens: int = 100000
    n_epochs: int = 1
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    log_every: int = 2000
    save_every: int = 20000
    output_dir: str = "experiments/wikitext_real"


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


def train_epoch(
    model: EvolvingModel,
    tokens: List[int],
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    metrics: Dict,
) -> float:
    """Train for one epoch."""

    model.reset_state()

    total_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()

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

        # Maybe evolve
        model.maybe_evolve()

        # Logging
        if (i + 1) % config.log_every == 0:
            acc = correct / total * 100
            avg_loss = total_loss / total
            ppl = min(
                torch.exp(torch.tensor(avg_loss)).item(), 10000
            )  # Cap perplexity display
            tokens_per_sec = total / (time.time() - epoch_start)

            hierarchy = model.get_hierarchy_summary()

            print(
                f"  [{epoch + 1}] Token {i + 1:,}/{len(tokens) - 1:,} | "
                f"Loss: {avg_loss:.3f} | PPL: {ppl:.1f} | Acc: {acc:.2f}% | "
                f"Gen: {model.total_evolutions} | "
                f"tok/s: {tokens_per_sec:.1f}"
            )

            # Log hierarchy evolution
            print(
                f"       Decays: {' '.join(f'{d:.2f}' for d in hierarchy['decays_by_layer'])}"
            )
            print(
                f"       Gates:  {' '.join(f'{g:+.1f}' for g in hierarchy['gate_biases_by_layer'])}"
            )

            # Record metrics
            metrics["steps"].append(
                {
                    "epoch": epoch,
                    "token": i + 1,
                    "loss": avg_loss,
                    "perplexity": ppl,
                    "accuracy": acc,
                    "generation": model.total_evolutions,
                    "decays": hierarchy["decays_by_layer"],
                    "gate_biases": hierarchy["gate_biases_by_layer"],
                    "total_births": hierarchy["total_births"],
                    "total_deaths": hierarchy["total_deaths"],
                }
            )

        # Checkpointing
        if config.save_every and (i + 1) % config.save_every == 0:
            ckpt_path = os.path.join(
                config.output_dir, f"checkpoint_e{epoch}_t{i + 1}.pt"
            )
            model.save_checkpoint(ckpt_path)
            print(f"       Saved: {ckpt_path}")

    final_acc = correct / total * 100
    final_loss = total_loss / total
    final_ppl = torch.exp(torch.tensor(final_loss)).item()
    epoch_time = time.time() - epoch_start

    print(
        f"\n  Epoch {epoch + 1}: Loss={final_loss:.3f}, PPL={final_ppl:.1f}, Acc={final_acc:.2f}%, Time={epoch_time:.1f}s"
    )

    return final_acc, final_loss, final_ppl


def analyze_hierarchy(model: EvolvingModel) -> Dict:
    """Analyze the evolved hierarchy."""
    print("\n" + "=" * 60)
    print("HIERARCHY ANALYSIS - REAL LANGUAGE")
    print("=" * 60)

    analysis = {"layers": []}

    for i, layer in enumerate(model.layers):
        stats = layer.get_stats()

        print(f"\nLayer {i}:")
        print(f"  Generation: {stats['generation']}")
        print(
            f"  Decay:      {stats['mean_decay']:.3f} +/- {stats['std_decay']:.3f} "
            f"[{stats['min_decay']:.3f}, {stats['max_decay']:.3f}]"
        )
        print(
            f"  Gate bias:  {stats['mean_gate_bias']:.2f} +/- {stats['std_gate_bias']:.2f} "
            f"[{stats['min_gate_bias']:.2f}, {stats['max_gate_bias']:.2f}]"
        )
        print(f"  Births/Deaths: {stats['total_births']}/{stats['total_deaths']}")

        analysis["layers"].append(stats)

    # Hierarchy assessment
    decays = [l["mean_decay"] for l in analysis["layers"]]
    gate_biases = [l["mean_gate_bias"] for l in analysis["layers"]]

    analysis["hierarchy_score"] = max(decays) - min(decays)
    analysis["gate_spread"] = max(gate_biases) - min(gate_biases)

    print(f"\n--- Summary ---")
    print(f"Decay spread: {analysis['hierarchy_score']:.3f}")
    print(f"Gate spread:  {analysis['gate_spread']:.2f}")

    # Find fast and slow layers
    sorted_by_decay = sorted(enumerate(decays), key=lambda x: x[1])
    fast_layers = [i for i, d in sorted_by_decay[:2]]
    slow_layers = [i for i, d in sorted_by_decay[-2:]]

    print(f"Fastest layers: {fast_layers} (decay ~ {decays[fast_layers[0]]:.3f})")
    print(f"Slowest layers: {slow_layers} (decay ~ {decays[slow_layers[0]]:.3f})")

    analysis["fast_layers"] = fast_layers
    analysis["slow_layers"] = slow_layers

    return analysis


def main():
    torch.manual_seed(42)

    print("=" * 60)
    print("EVOLVING SRC-FFN - WIKITEXT-2 EXPERIMENT")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Config
    train_config = TrainConfig(
        max_tokens=100000,  # ~100k tokens for reasonable runtime
        n_epochs=1,
        log_every=2000,
        save_every=20000,
    )

    model_config = ModelConfig(
        vocab_size=50257,  # GPT-2
        embed_dim=128,
        n_layers=8,
        n_neurons_per_layer=256,
        evolve_every=100,
    )

    # Load data
    tokens = load_cached_tokens(train_config.max_tokens)

    # Create output directory
    os.makedirs(train_config.output_dir, exist_ok=True)

    print(f"\nModel Configuration:")
    print(f"  Vocab size: {model_config.vocab_size:,}")
    print(f"  Embed dim: {model_config.embed_dim}")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Neurons/layer: {model_config.n_neurons_per_layer}")
    print(
        f"  Total neurons: {model_config.n_layers * model_config.n_neurons_per_layer:,}"
    )
    print(f"  Evolve every: {model_config.evolve_every} tokens")
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

    # Metrics
    metrics = {
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "data": "wikitext-2",
        "steps": [],
        "final": {},
    }

    # Training
    print("=" * 60)
    print("TRAINING ON REAL LANGUAGE")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(train_config.n_epochs):
        print(f"\n=== Epoch {epoch + 1}/{train_config.n_epochs} ===")
        acc, loss, ppl = train_epoch(
            model, tokens, optimizer, train_config, epoch, metrics
        )
        metrics["final"][f"epoch_{epoch}"] = {
            "accuracy": acc,
            "loss": loss,
            "perplexity": ppl,
        }
        model.reset_state()

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time / 60:.1f} minutes")

    # Analyze
    analysis = analyze_hierarchy(model)
    metrics["hierarchy_analysis"] = analysis

    # Save
    final_model_path = os.path.join(train_config.output_dir, "model_final.pt")
    model.save_checkpoint(final_model_path)
    print(f"\nSaved model: {final_model_path}")

    metrics_path = os.path.join(train_config.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
