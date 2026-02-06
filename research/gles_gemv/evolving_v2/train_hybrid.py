"""
Training Script for Hybrid Evolving Model on WikiText-2

Weights trained with gradients, dynamics evolved.
Should produce coherent output while allowing hierarchy to emerge.
"""

import torch
import torch.nn.functional as F
import json
import time
import os
import pickle
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict

from hybrid_model import HybridModel, HybridModelConfig


@dataclass
class TrainConfig:
    """Training configuration."""

    max_tokens: int = 50000
    n_epochs: int = 1
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    log_every: int = 2000
    save_every: int = 10000
    output_dir: str = "experiments/hybrid_v1"


def load_tokens(max_tokens: int = None) -> List[int]:
    """Load WikiText-2 tokens."""
    cache_path = "wikitext2_tokens.pkl"

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Run prepare_wikitext.py first")

    with open(cache_path, "rb") as f:
        tokens = pickle.load(f)

    if max_tokens:
        tokens = tokens[:max_tokens]

    print(f"Loaded {len(tokens):,} tokens")
    return tokens


def train(
    model: HybridModel,
    tokens: List[int],
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> Dict:
    """Train the model."""

    model.set_trajectory_tracking(True)
    model.reset_state()

    metrics = {
        "steps": [],
        "config": asdict(config),
    }

    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    print("\n" + "=" * 60)
    print("TRAINING HYBRID MODEL")
    print("=" * 60)

    for i in range(len(tokens) - 1):
        token_id = tokens[i]
        target_id = tokens[i + 1]

        # Forward
        logits = model.forward(token_id)
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id]))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Track
        total_loss += loss.item()
        if logits.argmax().item() == target_id:
            correct += 1
        total += 1

        # Maybe evolve
        model.maybe_evolve()

        # Logging
        if (i + 1) % config.log_every == 0:
            acc = correct / total * 100
            avg_loss = total_loss / total
            ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 50000)
            tps = total / (time.time() - start_time)

            hierarchy = model.get_hierarchy_summary()

            print(
                f"Token {i + 1:,}/{len(tokens) - 1:,} | "
                f"Loss: {avg_loss:.3f} | PPL: {ppl:.0f} | Acc: {acc:.2f}% | "
                f"Gen: {model.total_evolutions} | "
                f"DecaySpread: {hierarchy['decay_spread']:.3f} | "
                f"tok/s: {tps:.0f}"
            )

            # Per-layer info
            print(
                f"  Decays: {' '.join(f'{d:.2f}' for d in hierarchy['decays_by_layer'])}"
            )
            print(
                f"  Gates:  {' '.join(f'{g:+.1f}' for g in hierarchy['gate_biases_by_layer'])}"
            )

            metrics["steps"].append(
                {
                    "token": i + 1,
                    "loss": avg_loss,
                    "ppl": ppl,
                    "accuracy": acc,
                    "generation": model.total_evolutions,
                    "decay_spread": hierarchy["decay_spread"],
                    "decays": hierarchy["decays_by_layer"],
                    "gates": hierarchy["gate_biases_by_layer"],
                }
            )

        # Checkpointing
        if config.save_every and (i + 1) % config.save_every == 0:
            ckpt_path = os.path.join(config.output_dir, f"checkpoint_t{i + 1}.pt")
            model.save_checkpoint(ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final stats
    final_acc = correct / total * 100
    final_loss = total_loss / total
    final_ppl = torch.exp(torch.tensor(final_loss)).item()
    train_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"  Loss: {final_loss:.3f}")
    print(f"  PPL: {final_ppl:.0f}")
    print(f"  Accuracy: {final_acc:.2f}%")
    print(f"  Time: {train_time:.0f}s ({total / train_time:.0f} tok/s)")
    print(f"  Evolutions: {model.total_evolutions}")
    print("=" * 60)

    metrics["final"] = {
        "loss": final_loss,
        "ppl": final_ppl,
        "accuracy": final_acc,
        "train_time": train_time,
        "total_evolutions": model.total_evolutions,
    }

    return metrics


def test_generation(model: HybridModel, tokenizer):
    """Test generation quality."""
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    def generate(prompt: str, max_tokens: int = 30, temperature: float = 0.8) -> str:
        model.reset_state()
        tokens = tokenizer.encode(prompt)
        generated = tokens[:]

        # Feed prompt
        for t in tokens[:-1]:
            _ = model.forward(t)

        # Generate
        for _ in range(max_tokens):
            logits = model.forward(generated[-1])
            probs = F.softmax(logits / temperature, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

        return tokenizer.decode(generated)

    prompts = [
        "The",
        "In the",
        "He said",
        "The president",
        "Scientists discovered",
    ]

    for p in prompts:
        print(f'\nPrompt: "{p}"')
        result = generate(p)
        print(f"Output: {result}")


def main():
    torch.manual_seed(42)

    print("=" * 60)
    print("HYBRID EVOLVING MODEL - WIKITEXT-2")
    print("Weights: gradient descent | Dynamics: evolution")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Config
    train_config = TrainConfig(
        max_tokens=50000,
        log_every=2000,
        save_every=10000,
    )

    model_config = HybridModelConfig(
        vocab_size=50257,
        embed_dim=128,
        n_layers=8,
        n_neurons_per_layer=256,
        evolve_every=100,
    )

    # Load data
    tokens = load_tokens(train_config.max_tokens)

    # Create output dir
    os.makedirs(train_config.output_dir, exist_ok=True)

    print(f"\nModel config:")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Neurons/layer: {model_config.n_neurons_per_layer}")
    print(f"  Embed dim: {model_config.embed_dim}")
    print(f"  Evolve every: {model_config.evolve_every}")

    # Create model
    model = HybridModel(model_config)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} (trainable: {trainable:,})")

    # Optimizer - train everything
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    # Train
    metrics = train(model, tokens, optimizer, train_config)

    # Save final model and metrics
    model.save_checkpoint(os.path.join(train_config.output_dir, "model_final.pt"))

    metrics["model_config"] = asdict(model_config)
    metrics["trajectory"] = model.trajectory

    with open(os.path.join(train_config.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Test generation
    try:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        test_generation(model, tokenizer)
    except Exception as e:
        print(f"\nSkipping generation test: {e}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
