"""
Analyze a checkpoint to see what hierarchy has evolved.
"""

import torch
import sys
from model import EvolvingModel


def analyze(checkpoint_path: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    print(f"\nModel Config:")
    config = checkpoint["config"]
    print(f"  Layers: {config.n_layers}")
    print(f"  Neurons/layer: {config.n_neurons_per_layer}")
    print(f"  Embed dim: {config.embed_dim}")

    print(f"\nTraining State:")
    print(f"  Tokens seen: {checkpoint['tokens_seen']:,}")
    print(f"  Evolutions: {checkpoint['total_evolutions']}")

    # Load model
    model = EvolvingModel(config)
    model.load_state_dict(checkpoint["model_state"])

    print(f"\n{'=' * 60}")
    print("HIERARCHY ANALYSIS")
    print("=" * 60)

    decays = []
    gates = []

    for i, layer in enumerate(model.layers):
        stats = layer.get_stats()
        decays.append(stats["mean_decay"])
        gates.append(stats["mean_gate_bias"])

        print(f"\nLayer {i}:")
        print(f"  Decay:     {stats['mean_decay']:.3f} +/- {stats['std_decay']:.3f}")
        print(
            f"  Gate bias: {stats['mean_gate_bias']:.2f} +/- {stats['std_gate_bias']:.2f}"
        )
        print(f"  Generation: {stats['generation']}")
        print(f"  Births: {stats['total_births']:,}")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    print(f"\nDecay by layer:     {' '.join(f'{d:.3f}' for d in decays)}")
    print(f"Gate bias by layer: {' '.join(f'{g:+.2f}' for g in gates)}")

    decay_spread = max(decays) - min(decays)
    gate_spread = max(gates) - min(gates)

    print(f"\nDecay spread: {decay_spread:.3f}")
    print(f"Gate spread:  {gate_spread:.2f}")

    # Find patterns
    sorted_by_decay = sorted(enumerate(decays), key=lambda x: x[1])
    fast_layers = [i for i, _ in sorted_by_decay[:2]]
    slow_layers = [i for i, _ in sorted_by_decay[-2:]]

    print(f"\nFastest layers: {fast_layers}")
    print(f"Slowest layers: {slow_layers}")

    # Check for early-fast-late-slow pattern
    early_avg = sum(decays[: len(decays) // 2]) / (len(decays) // 2)
    late_avg = sum(decays[len(decays) // 2 :]) / (len(decays) // 2)

    if late_avg > early_avg + 0.01:
        print(f"\nPattern: Early=fast ({early_avg:.3f}), Late=slow ({late_avg:.3f})")
    elif early_avg > late_avg + 0.01:
        print(
            f"\nPattern: Early=slow ({early_avg:.3f}), Late=fast ({late_avg:.3f}) [INVERTED]"
        )
    else:
        print(f"\nPattern: No clear early/late differentiation")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to most recent checkpoint
        import glob

        checkpoints = sorted(glob.glob("experiments/*/checkpoint*.pt"))
        if checkpoints:
            path = checkpoints[-1]
        else:
            print("Usage: python analyze_checkpoint.py <checkpoint_path>")
            sys.exit(1)
    else:
        path = sys.argv[1]

    analyze(path)
