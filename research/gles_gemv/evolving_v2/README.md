# Evolving SRC-FFN v2

Vectorized implementation of evolving neural populations for sequence modeling.

## Overview

This implements the scaled version of PRD-003 (Evolving Neural Populations). Key improvements over v1:
- **50-100x faster** through vectorized batch computation
- **Configurable scale** (tested up to 8 layers × 512 neurons)
- **Full training pipeline** with metrics and checkpointing

## Quick Start

```bash
# Simple test (fast)
python train.py --simple --max-tokens 10000 --epochs 1 --layers 4 --neurons 64

# Medium scale (recommended for testing)
python train.py --simple --max-tokens 50000 --epochs 3 --layers 8 --neurons 256

# Full scale with WikiText-2 (requires datasets/transformers)
python train.py --max-tokens 100000 --epochs 3 --layers 8 --neurons 512
```

## Results

### Scaling Experiment (scaling_002)

| Metric | Value |
|--------|-------|
| Peak Accuracy | **97.9%** |
| Configuration | 8 layers × 256 neurons |
| Total Neurons | 2,048 |
| Generations | 1,499 per layer |
| Total Births/Deaths | 1.5 million |
| Speed | 758 tokens/sec |

### Evolved Hierarchy

The system evolved an unexpected structure:

```
Layer | Decay | Gate  | Interpretation
------|-------|-------|---------------
  0   | 0.987 | +1.74 | Stable input
  1   | 0.987 | +1.76 | Stable input
  2   | 0.980 | +1.76 | Stable input
  3   | 0.898 | +1.70 | FAST intermediate
  4   | 0.992 | +1.79 | Memory bank
  5   | 0.909 | +1.82 | FAST intermediate
  6   | 0.991 | +1.73 | Memory bank
  7   | 0.988 | +1.04 | Specialized output (different gate)
```

Key finding: Middle layers (3, 5) evolved to be "fast" with lower decay, not early layers as initially expected.

## Files

```
evolving_v2/
├── vectorized_population.py  # Core: batched neuron computation
├── model.py                  # Full model assembly
├── train.py                  # Training script
└── experiments/              # Results
    ├── test_001/             # Quick test run
    ├── scaling_002/          # 97.9% accuracy experiment
    └── wikitext_001/         # WikiText experiment (partial)
```

## Architecture

### VectorizedPopulation

All neurons computed as a single batch operation:

```python
# Instead of:
for neuron in neurons:
    output = neuron.forward(x)

# We do:
up = torch.mv(w_up, x)           # [n_neurons]
g = torch.sigmoid(up + ...)       # [n_neurons]
h = (1-g) * h * decay + g * tanh(up)  # [n_neurons]
output = torch.mv(w_down.t(), h)  # [embed_dim]
```

### Evolution

Each generation:
1. **Finalize fitness** - Average contribution over tokens seen
2. **Sort by fitness** - Rank all neurons
3. **Selection** - Top 50% survive
4. **Reproduction** - Survivors produce offspring with mutation
5. **Reset** - Clear fitness for next generation

### Fitness Function

```python
contribution = neuron_output * dot(w_down, target_embedding)
fitness = mean(contributions over window)
```

Neurons that push the output toward the correct next token get positive fitness.

## Configuration

### ModelConfig

```python
ModelConfig(
    vocab_size=50257,        # GPT-2 tokenizer
    embed_dim=256,           # Embedding dimension
    n_layers=8,              # Number of layers
    n_neurons_per_layer=512, # Neurons per layer
    evolve_every=100,        # Tokens between evolution
    selection_pressure=0.5,  # Fraction surviving
    elite_fraction=0.1,      # Protected from mutation
    mutation_rate=0.3,       # Probability of mutation
    mutation_strength=0.1,   # Scale of mutations
    sexual_rate=0.3,         # Crossover vs cloning
)
```

### CLI Arguments

```
--simple          Use simple test data instead of WikiText
--max-tokens N    Limit training tokens
--epochs N        Number of epochs
--layers N        Number of layers
--neurons N       Neurons per layer
--embed-dim N     Embedding dimension
--evolve-every N  Tokens between evolution
--output-dir DIR  Output directory
```

## Dependencies

- PyTorch >= 2.0
- For WikiText: `datasets`, `transformers`

## Key Insights

1. **Evolution works at scale** - 97.9% accuracy with 2,048 neurons
2. **Hierarchy is task-specific** - Simple patterns produce different structure than expected
3. **Selection does credit assignment** - No backprop through neuron parameters
4. **Diversity maintenance is crucial** - Without it, populations collapse

## Next Steps

1. Run on WikiText-2 to see what hierarchy emerges for real language
2. Compare to gradient-descent baseline on same architecture
3. Analyze WHY layers 3 and 5 became fast
4. Test continuous adaptation to distribution shift

## References

- PRD-001: SRC-FFN Architecture
- PRD-002: Equilibrium Propagation Training
- PRD-003: Evolving Neural Populations
- DISCOVERY-SUMMARY.md: Full project overview
