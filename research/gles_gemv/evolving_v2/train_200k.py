#!/usr/bin/env python3
"""
200k Token Training - Quality Assessment

Train the Breathing model for 200k tokens and evaluate if it produces
useful text or just "sprinkler play".

Key questions:
1. Does PPL keep improving or plateau?
2. Does generated text become coherent?
3. Are we learning language structure or just memorizing?
"""

import torch
import torch.nn.functional as F
import pickle
import time
import json
from pathlib import Path
from transformers import GPT2Tokenizer

from breathing_model import BreathingModel, BreathingModelConfig


def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.8):
    """Generate text from prompt."""
    model.reset_state()
    model.eval()

    toks = tokenizer.encode(prompt)
    with torch.no_grad():
        # Feed prompt
        for t in toks[:-1]:
            model.forward(t)

        # Generate
        generated = []
        for _ in range(max_tokens):
            logits = model.forward(toks[-1])
            probs = F.softmax(logits / temperature, dim=0)
            next_tok = torch.multinomial(probs, 1).item()
            generated.append(next_tok)
            toks.append(next_tok)

            # Stop at EOS or newline sometimes
            if next_tok == tokenizer.eos_token_id:
                break

    full_text = tokenizer.decode(toks)
    generated_text = tokenizer.decode(generated)
    return full_text, generated_text


def train_and_evaluate():
    N_TOKENS = 200000
    CHECKPOINT_EVERY = 25000
    GENERATE_EVERY = 25000
    LOG_EVERY = 5000

    print("=" * 70)
    print("200K TOKEN TRAINING - QUALITY ASSESSMENT")
    print("=" * 70)

    # Setup
    Path("experiments/200k_training").mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load data
    print("\nLoading cleaned WikiText-2...")
    with open("wikitext2_clean_tokens.pkl", "rb") as f:
        all_tokens = pickle.load(f)

    if len(all_tokens) < N_TOKENS + 1:
        print(f"Warning: Only {len(all_tokens)} tokens available, need {N_TOKENS + 1}")
        N_TOKENS = len(all_tokens) - 1

    tokens = all_tokens[: N_TOKENS + 1]
    print(f"Training on {N_TOKENS:,} tokens")

    # Create model
    print("\nCreating Breathing Model...")
    config = BreathingModelConfig(
        vocab_size=50257,
        embed_dim=128,
        n_layers=4,
        n_neurons_per_layer=128,
        n_breathe=5,
        evolve_every=1000,  # Evolve every 1000 tokens
        decay_min=0.5,
        decay_max=0.9,
    )
    model = BreathingModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(
        f"  Architecture: {config.n_layers} layers, {config.n_neurons_per_layer} neurons/layer"
    )
    print(f"  Breathing: {config.n_breathe} steps/token")
    print(f"  Evolution: every {config.evolve_every} tokens")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Metrics
    history = {
        "config": {"n_tokens": N_TOKENS, "n_params": n_params},
        "checkpoints": [],
        "generations": [],
    }

    # Initial generation
    print("\n" + "-" * 70)
    print("INITIAL STATE (0 tokens)")
    print("-" * 70)
    test_prompts = ["The", "In the", "He said", "The company"]
    for prompt in test_prompts:
        full, generated = generate_text(model, tokenizer, prompt, max_tokens=30)
        print(f"\n  '{prompt}' -> {full}")

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    start_time = time.time()
    total_loss = 0
    correct = 0

    for i in range(N_TOKENS):
        model.train()

        # Forward
        logits = model.forward(tokens[i])
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tokens[i + 1]]))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Stats
        total_loss += loss.item()
        if logits.argmax().item() == tokens[i + 1]:
            correct += 1

        # Evolution
        model.maybe_evolve()

        # Logging
        if (i + 1) % LOG_EVERY == 0:
            avg_loss = total_loss / (i + 1)
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            acc = correct / (i + 1) * 100
            elapsed = time.time() - start_time
            tps = (i + 1) / elapsed
            eta = (N_TOKENS - i - 1) / tps / 60

            # Kuramoto
            r_mean = (
                sum(
                    layer.population.kuramoto_order_parameter(t=50.0)
                    for layer in model.layers
                )
                / config.n_layers
            )

            print(
                f"  {i + 1:>7,}/{N_TOKENS:,} | Loss: {avg_loss:.3f} | PPL: {ppl:>6.0f} | "
                f"Acc: {acc:>5.1f}% | Evol: {model.total_evolutions:>3} | "
                f"Kura: {r_mean:.3f} | {tps:>5.0f} tok/s | ETA: {eta:>5.1f}m"
            )

            history["checkpoints"].append(
                {
                    "step": i + 1,
                    "loss": avg_loss,
                    "ppl": ppl,
                    "acc": acc,
                    "evolutions": model.total_evolutions,
                    "kuramoto_mean": r_mean,
                    "elapsed": elapsed,
                }
            )

        # Generation check
        if (i + 1) % GENERATE_EVERY == 0:
            print("\n" + "-" * 70)
            print(f"GENERATION CHECK ({i + 1:,} tokens)")
            print("-" * 70)

            for prompt in ["The", "In the", "He said", "The company"]:
                full, generated = generate_text(model, tokenizer, prompt, max_tokens=40)
                print(f"\n  '{prompt}' -> {full}")

            # Save generation sample
            history["generations"].append(
                {
                    "step": i + 1,
                    "samples": [
                        {
                            "prompt": p,
                            "output": generate_text(model, tokenizer, p, max_tokens=40)[
                                0
                            ],
                        }
                        for p in ["The", "In 1999,", "Scientists have discovered"]
                    ],
                }
            )
            print()

        # Checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = f"experiments/200k_training/checkpoint_step{i + 1}.pt"
            model.save_checkpoint(ckpt_path)
            print(f"\n  [Saved checkpoint: {ckpt_path}]\n")

    # Final stats
    final_loss = total_loss / N_TOKENS
    final_ppl = torch.exp(torch.tensor(final_loss)).item()
    final_acc = correct / N_TOKENS * 100
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Final Loss: {final_loss:.3f}")
    print(f"  Final PPL: {final_ppl:.0f}")
    print(f"  Final Accuracy: {final_acc:.2f}%")
    print(f"  Total Evolutions: {model.total_evolutions}")
    print(f"  Total Time: {total_time / 60:.1f} minutes")
    print(f"  Average Speed: {N_TOKENS / total_time:.0f} tok/s")

    # Final generation
    print("\n" + "-" * 70)
    print("FINAL GENERATION (200k tokens)")
    print("-" * 70)

    final_samples = []
    for prompt in [
        "The",
        "In the year 2024,",
        "Scientists have discovered that",
        "The president announced",
        "Once upon a time",
        "In conclusion, the results show",
    ]:
        full, generated = generate_text(
            model, tokenizer, prompt, max_tokens=50, temperature=0.8
        )
        final_samples.append({"prompt": prompt, "output": full})
        print(f"\n  '{prompt}'")
        print(f"  -> {full}")

    history["final"] = {
        "loss": final_loss,
        "ppl": final_ppl,
        "acc": final_acc,
        "time": total_time,
        "samples": final_samples,
    }

    # Save history
    with open("experiments/200k_training/history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Save final model
    model.save_checkpoint("experiments/200k_training/final_model.pt")

    # Quality assessment
    print("\n" + "=" * 70)
    print("QUALITY ASSESSMENT")
    print("=" * 70)

    # Heuristics for "sprinkler play" vs useful
    ppl_target = 100  # GPT-2 small is ~20, tiny models should beat 100
    acc_target = 15  # Reasonable for small model

    print(f"\nMetrics vs Targets:")
    print(f"  Perplexity: {final_ppl:.0f} (target: <{ppl_target})")
    print(f"  Accuracy:   {final_acc:.1f}% (target: >{acc_target}%)")
    print(f"  PPL Trend:  {history['checkpoints'][0]['ppl']:.0f} → {final_ppl:.0f}")

    # Trend analysis
    if len(history["checkpoints"]) >= 2:
        first_ppl = history["checkpoints"][0]["ppl"]
        last_ppl = history["checkpoints"][-1]["ppl"]
        improvement = (first_ppl - last_ppl) / first_ppl * 100
        print(f"  PPL Improvement: {improvement:.1f}%")

        if improvement > 50:
            trend = "STRONG improvement - learning"
        elif improvement > 20:
            trend = "MODERATE improvement - some learning"
        elif improvement > 0:
            trend = "WEAK improvement - plateauing"
        else:
            trend = "NO improvement - stuck"
        print(f"  Trend: {trend}")

    # Verdict
    print("\n" + "-" * 70)
    print("VERDICT")
    print("-" * 70)

    useful = False
    reasons = []

    if final_ppl < ppl_target:
        useful = True
        reasons.append(f"PPL {final_ppl:.0f} < {ppl_target} target")
    else:
        reasons.append(f"PPL {final_ppl:.0f} > {ppl_target} target")

    if final_acc > acc_target:
        useful = True
        reasons.append(f"Accuracy {final_acc:.1f}% > {acc_target}% target")
    else:
        reasons.append(f"Accuracy {final_acc:.1f}% < {acc_target}% target")

    if model.total_evolutions > 50:
        reasons.append(f"Evolution active ({model.total_evolutions} cycles)")

    if useful:
        print("  Status: USEFUL ✓")
        print(f"  This model shows learning and could be useful for:")
        print(f"    - Small language tasks")
        print(f"    - Proof of concept")
        print(f"    - Further scaling")
    else:
        print("  Status: SPRINKLER PLAY ✗")
        print(f"  This model is not yet useful. Issues:")
        for r in reasons:
            print(f"    - {r}")

    print(f"\n  Recommendation: {'Scale up' if useful else 'Debug and fix'}")

    print("\n" + "=" * 70)
    print("All results saved to experiments/200k_training/")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
