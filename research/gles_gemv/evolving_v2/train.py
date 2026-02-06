"""
Training Script for Evolving SRC-FFN

The scaling experiment:
- 8 layers x 512 neurons = 4,096 evolving neurons
- WikiText-2 dataset (~2M tokens)
- Evolution + gradient descent on embeddings
"""

import torch
import torch.nn.functional as F
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import argparse

from model import EvolvingModel, ModelConfig


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    data_source: str = "wikitext"  # "wikitext" or "simple" for testing
    max_tokens: int = 100000  # Limit for testing

    # Training
    n_epochs: int = 3
    learning_rate: float = 1e-3
    grad_clip: float = 1.0

    # Logging
    log_every: int = 1000
    save_every: int = 10000

    # Output
    output_dir: str = "experiments/scaling_001"


def load_wikitext_tokens(max_tokens: int = None) -> List[int]:
    """Load WikiText-2 and tokenize with GPT-2 tokenizer."""
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
    except ImportError:
        print("Installing required packages...")
        import subprocess

        subprocess.run(["pip", "install", "datasets", "transformers"], check=True)
        from datasets import load_dataset
        from transformers import GPT2Tokenizer

    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)

    print("Tokenizing...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Concatenate all text
    texts = dataset["train"]["text"]
    full_text = "\n".join([t for t in texts if t.strip()])

    # Tokenize
    tokens = tokenizer.encode(full_text)

    if max_tokens:
        tokens = tokens[:max_tokens]

    print(f"Total tokens: {len(tokens):,}")
    return tokens


def load_simple_tokens(max_tokens: int = None) -> List[int]:
    """Simple test data - "hello world" pattern."""
    # Character-level for simplicity
    text = "hello world " * 10000
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    tokens = [char_to_id[c] for c in text]

    if max_tokens:
        tokens = tokens[:max_tokens]

    print(f"Simple data: {len(tokens):,} tokens, vocab size: {len(chars)}")
    return tokens


def train_epoch(
    model: EvolvingModel,
    tokens: List[int],
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

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
        if (i + 1) % train_config.log_every == 0:
            acc = correct / total * 100
            avg_loss = total_loss / total
            tokens_per_sec = total / (time.time() - epoch_start)

            hierarchy = model.get_hierarchy_summary()

            print(
                f"  [{epoch + 1}] Token {i + 1:,}/{len(tokens) - 1:,} | "
                f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | "
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
                    "accuracy": acc,
                    "generation": model.total_evolutions,
                    "decays": hierarchy["decays_by_layer"],
                    "gate_biases": hierarchy["gate_biases_by_layer"],
                    "total_births": hierarchy["total_births"],
                    "total_deaths": hierarchy["total_deaths"],
                }
            )

        # Checkpointing
        if train_config.save_every and (i + 1) % train_config.save_every == 0:
            ckpt_path = os.path.join(
                train_config.output_dir, f"checkpoint_e{epoch}_t{i + 1}.pt"
            )
            model.save_checkpoint(ckpt_path)
            print(f"       Saved checkpoint: {ckpt_path}")

    final_acc = correct / total * 100
    final_loss = total_loss / total
    epoch_time = time.time() - epoch_start

    print(
        f"\n  Epoch {epoch + 1} complete: Loss={final_loss:.4f}, Acc={final_acc:.2f}%, Time={epoch_time:.1f}s"
    )

    return final_acc


def analyze_hierarchy(model: EvolvingModel) -> Dict:
    """Analyze the evolved hierarchy."""
    print("\n" + "=" * 60)
    print("HIERARCHY ANALYSIS")
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
        print(f"  Max age: {stats['max_age']}")

        analysis["layers"].append(stats)

    # Overall hierarchy assessment
    decays = [l["mean_decay"] for l in analysis["layers"]]
    hierarchy_score = max(decays) - min(decays)

    print(f"\nHierarchy Score (decay spread): {hierarchy_score:.3f}")
    print(f"  {'Strong hierarchy' if hierarchy_score > 0.1 else 'Weak hierarchy'}")

    # Check if early layers are faster
    early_decay = sum(decays[: len(decays) // 2]) / (len(decays) // 2)
    late_decay = sum(decays[len(decays) // 2 :]) / (len(decays) // 2)

    if late_decay > early_decay + 0.02:
        print(
            f"  Pattern: Early=fast ({early_decay:.3f}), Late=slow ({late_decay:.3f}) [EXPECTED]"
        )
    elif early_decay > late_decay + 0.02:
        print(
            f"  Pattern: Early=slow ({early_decay:.3f}), Late=fast ({late_decay:.3f}) [INVERTED]"
        )
    else:
        print(f"  Pattern: Uniform ({early_decay:.3f} ~ {late_decay:.3f})")

    analysis["hierarchy_score"] = hierarchy_score
    analysis["early_decay"] = early_decay
    analysis["late_decay"] = late_decay

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Train Evolving SRC-FFN")
    parser.add_argument(
        "--simple", action="store_true", help="Use simple test data instead of WikiText"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100000, help="Maximum tokens to train on"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--neurons", type=int, default=512, help="Neurons per layer")
    parser.add_argument(
        "--embed-dim", type=int, default=256, help="Embedding dimension"
    )
    parser.add_argument(
        "--evolve-every", type=int, default=100, help="Tokens between evolution"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/scaling_001",
        help="Output directory",
    )
    args = parser.parse_args()

    # Setup
    torch.manual_seed(42)

    print("=" * 60)
    print("EVOLVING SRC-FFN - SCALING EXPERIMENT")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    train_config = TrainConfig(
        data_source="simple" if args.simple else "wikitext",
        max_tokens=args.max_tokens,
        n_epochs=args.epochs,
        output_dir=args.output_dir,
    )

    if args.simple:
        tokens = load_simple_tokens(args.max_tokens)
        vocab_size = max(tokens) + 1
    else:
        tokens = load_wikitext_tokens(args.max_tokens)
        vocab_size = 50257  # GPT-2

    # Create output directory
    os.makedirs(train_config.output_dir, exist_ok=True)

    # Model config
    model_config = ModelConfig(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        n_layers=args.layers,
        n_neurons_per_layer=args.neurons,
        evolve_every=args.evolve_every,
    )

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

    # Only train embeddings with gradient descent
    optimizer = torch.optim.AdamW(
        [
            {"params": model.embed.parameters()},
            {"params": model.out_norm.parameters()},
        ],
        lr=train_config.learning_rate,
    )

    # Metrics tracking
    metrics = {
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "steps": [],
        "epoch_accuracies": [],
    }

    # Training loop
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(train_config.n_epochs):
        print(f"\n=== Epoch {epoch + 1}/{train_config.n_epochs} ===")
        acc = train_epoch(model, tokens, optimizer, train_config, epoch, metrics)
        metrics["epoch_accuracies"].append(acc)

        # Reset state between epochs
        model.reset_state()

    total_time = time.time() - start_time

    print(f"\nTotal training time: {total_time / 60:.1f} minutes")
    print(f"Total evolutions: {model.total_evolutions}")

    # Analyze hierarchy
    analysis = analyze_hierarchy(model)
    metrics["hierarchy_analysis"] = analysis

    # Save final model and metrics
    final_model_path = os.path.join(train_config.output_dir, "model_final.pt")
    model.save_checkpoint(final_model_path)
    print(f"\nSaved final model: {final_model_path}")

    metrics_path = os.path.join(train_config.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Final accuracy: {metrics['epoch_accuracies'][-1]:.2f}%")
    print(f"Hierarchy score: {analysis['hierarchy_score']:.3f}")
    print(f"Total generations: {model.total_evolutions}")
    print(
        f"Total births/deaths: {analysis['layers'][0]['total_births'] * len(model.layers)}"
    )


if __name__ == "__main__":
    main()
