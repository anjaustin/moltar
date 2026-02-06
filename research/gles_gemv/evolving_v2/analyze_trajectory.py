"""
Analyze trajectory data from evolution experiment.

Tests the Delta Observer hypothesis: Is hierarchy scaffolding or stable structure?
"""

import json
import sys
from typing import Dict, List


def load_trajectory(path: str) -> Dict:
    """Load trajectory data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def analyze_trajectory(data: Dict) -> Dict:
    """Analyze trajectory for scaffolding vs stable structure pattern."""

    trajectory = data["trajectory"]
    generations = trajectory["generations"]
    decay_spread = trajectory["decay_spread"]
    diversity = trajectory["diversity"]
    per_layer_decays = trajectory["per_layer_decays"]

    n = len(generations)
    print(f"Analyzing {n} trajectory points (generations 1-{generations[-1]})")

    if n < 10:
        return {"error": "Not enough data points"}

    # Split into thirds
    third = n // 3

    early_spread = sum(decay_spread[:third]) / third
    mid_spread = sum(decay_spread[third : 2 * third]) / third
    late_spread = sum(decay_spread[2 * third :]) / (n - 2 * third)

    early_div = sum(diversity[:third]) / third
    mid_div = sum(diversity[third : 2 * third]) / third
    late_div = sum(diversity[2 * third :]) / (n - 2 * third)

    # Per-layer analysis
    n_layers = len(per_layer_decays[0])

    # Track each layer's decay over time
    layer_trajectories = [[] for _ in range(n_layers)]
    for point in per_layer_decays:
        for i, decay in enumerate(point):
            layer_trajectories[i].append(decay)

    # Layer trends (early vs late)
    layer_trends = []
    for layer_idx, decays in enumerate(layer_trajectories):
        early_decay = sum(decays[:third]) / third
        late_decay = sum(decays[2 * third :]) / (n - 2 * third)
        trend = late_decay - early_decay  # positive = slowing, negative = speeding up
        layer_trends.append(
            {
                "layer": layer_idx,
                "early_decay": early_decay,
                "late_decay": late_decay,
                "trend": trend,
                "direction": "SLOWING"
                if trend > 0.01
                else ("SPEEDING" if trend < -0.01 else "STABLE"),
            }
        )

    # Determine pattern
    if mid_spread > early_spread * 1.1 and mid_spread > late_spread * 1.1:
        pattern = "SCAFFOLDING"
        description = "Hierarchy rises then falls (like Delta Observer clustering)"
    elif late_spread > mid_spread > early_spread:
        pattern = "STABLE_STRUCTURE"
        description = "Hierarchy continues to strengthen"
    elif late_spread > early_spread and late_spread > mid_spread:
        pattern = "LATE_STRUCTURE"
        description = "Hierarchy strengthens late (not scaffolding)"
    elif early_spread > mid_spread and early_spread > late_spread:
        pattern = "EARLY_COLLAPSE"
        description = "Initial hierarchy weakens over time"
    else:
        pattern = "MIXED"
        description = "No clear pattern"

    # Find max spread point
    max_spread = max(decay_spread)
    max_spread_gen = generations[decay_spread.index(max_spread)]

    return {
        "pattern": pattern,
        "description": description,
        "n_points": n,
        "early_spread": early_spread,
        "mid_spread": mid_spread,
        "late_spread": late_spread,
        "early_diversity": early_div,
        "mid_diversity": mid_div,
        "late_diversity": late_div,
        "max_spread": max_spread,
        "max_spread_gen": max_spread_gen,
        "max_spread_position": max_spread_gen / generations[-1],  # 0=early, 1=late
        "final_spread": decay_spread[-1],
        "layer_trends": layer_trends,
        "first_5_spreads": decay_spread[:5],
        "last_5_spreads": decay_spread[-5:],
    }


def print_analysis(analysis: Dict):
    """Print analysis results."""
    print("\n" + "=" * 60)
    print("TRAJECTORY ANALYSIS - DELTA OBSERVER HYPOTHESIS")
    print("=" * 60)

    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    print(f"\n*** PATTERN: {analysis['pattern']} ***")
    print(f"    {analysis['description']}")

    print(f"\n--- Hierarchy Strength (Decay Spread) by Phase ---")
    print(
        f"  Early (gen 1-{analysis['n_points'] // 3}):     {analysis['early_spread']:.4f}"
    )
    print(f"  Middle:                      {analysis['mid_spread']:.4f}")
    print(
        f"  Late (to gen {analysis['n_points']}):         {analysis['late_spread']:.4f}"
    )

    print(f"\n--- Diversity by Phase ---")
    print(f"  Early:  {analysis['early_diversity']:.4f}")
    print(f"  Middle: {analysis['mid_diversity']:.4f}")
    print(f"  Late:   {analysis['late_diversity']:.4f}")

    print(f"\n--- Peak Analysis ---")
    print(
        f"  Max spread: {analysis['max_spread']:.4f} at generation {analysis['max_spread_gen']}"
    )
    print(f"  Position:   {analysis['max_spread_position']:.1%} through training")
    print(f"  Final:      {analysis['final_spread']:.4f}")

    print(f"\n--- First 5 spreads: {[f'{s:.4f}' for s in analysis['first_5_spreads']]}")
    print(f"--- Last 5 spreads:  {[f'{s:.4f}' for s in analysis['last_5_spreads']]}")

    print(f"\n--- Per-Layer Trends (Early → Late) ---")
    for lt in analysis["layer_trends"]:
        print(
            f"  Layer {lt['layer']}: {lt['early_decay']:.3f} → {lt['late_decay']:.3f} ({lt['direction']})"
        )

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if analysis["pattern"] == "SCAFFOLDING":
        print("\n✓ Hierarchy IS transient scaffolding!")
        print("  - Like clustering in Delta Observer")
        print("  - Hierarchy emerges to help learning then dissolves")
        print("  - The structure we see mid-training may not persist")
        print("  - Semantic info encoded in weights, not hierarchy")
    elif analysis["pattern"] in ["STABLE_STRUCTURE", "LATE_STRUCTURE"]:
        print("\n✗ Hierarchy is NOT scaffolding - it's stable structure")
        print("  - Unlike Delta Observer clustering")
        print("  - Evolution finds and maintains temporal organization")
        print("  - The hierarchy we observe is the final architecture")
    else:
        print(f"\n? Pattern is {analysis['pattern']}")
        print("  - Doesn't clearly match either hypothesis")
        print("  - May need longer training or different analysis")

    # Delta Observer comparison
    print("\n--- Delta Observer Comparison ---")
    print("  Delta Observer silhouette: -0.02 → 0.33 → -0.02")
    print(
        f"  Our decay spread:         {analysis['first_5_spreads'][0]:.3f} → {analysis['max_spread']:.3f} → {analysis['final_spread']:.3f}"
    )

    ratio = analysis["max_spread"] / max(analysis["final_spread"], 0.001)
    if ratio > 1.5:
        print(f"  *** Peak is {ratio:.1f}x final - SCAFFOLDING SIGNATURE ***")
    else:
        print(f"  *** Peak is only {ratio:.1f}x final - STABLE STRUCTURE ***")


def main():
    if len(sys.argv) < 2:
        # Default to latest trajectory file
        import glob

        files = sorted(
            glob.glob("experiments/trajectory_analysis/trajectory_gen*.json")
        )
        if not files:
            print("No trajectory files found. Run train_trajectory.py first.")
            return
        path = files[-1]
        print(f"Using latest trajectory: {path}")
    else:
        path = sys.argv[1]

    data = load_trajectory(path)
    analysis = analyze_trajectory(data)
    print_analysis(analysis)

    # Save analysis
    output_path = path.replace(".json", "_analysis.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to: {output_path}")


if __name__ == "__main__":
    main()
