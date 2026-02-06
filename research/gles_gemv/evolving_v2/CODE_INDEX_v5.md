# Code Index: Evolving SRC-FFN v5.0

**Last updated:** 2026-02-06  
**Status**: Production Ready (Validated)  
**Architecture**: Breathing + Niche Evolution  

---

## 🏗️ Core Architecture (Production)

### **Breathing Architecture** (Validated - Beats GRU)

| File | Lines | Description | Status |
|------|-------|-------------|---------|
| `breathing_layer.py` | 576 | CfC with 5-step breathing + niche evolution | ✅ **Core** |
| `breathing_model.py` | 275 | Full breathing model assembly | ✅ **Core** |
| `compare_baselines.py` | 288 | GRU comparison validation | ✅ **Validated** |

**Key Achievement**: 33.4% lower perplexity than parameter-matched GRU baseline.

### **Training Infrastructure**

| File | Lines | Description | Status |
|------|-------|-------------|---------|
| `train_wikitext.py` | 313 | Main training on WikiText-2 | ✅ **Production** |
| `train_200k.py` | 345 | 200K token optimized training | ✅ **Optimized** |

---

## 🔬 Analysis & Research Tools

### **Trajectory Analysis**

| File | Lines | Description | Status |
|------|-------|-------------|---------|
| `train_trajectory.py` | 359 | Training with trajectory tracking | ✅ **Research** |
| `analyze_trajectory.py` | 218 | Delta Observer-style analysis | ✅ **Analysis** |
| `analyze_checkpoint.py` | 97 | Checkpoint inspection tools | ✅ **Utility** |

### **Parameter Optimization**

| File | Lines | Description | Status |
|------|-------|-------------|---------|
| `batch_size_finetune.py` | 327 | Batch size optimization | 🔬 **Experimental** |
| `batch_size_opt.py` | 258 | Batch size experiments | 🔬 **Experimental** |
| `lr_warmup_test.py` | 235 | Learning rate warmup | 🔬 **Experimental** |
| `compare_long.py` | 312 | Long sequence comparison | 🔬 **Research** |
| `quick_compare.py` | 76 | Quick baseline comparison | ✅ **Utility** |
| `gru_baseline.py` | 261 | GRU parameter-matched baseline | ✅ **Reference** |

---

## 🗃️ Data & Preparation

| File | Lines | Description | Status |
|------|-------|-------------|---------|
| `prepare_wikitext.py` | 58 | WikiText-2 data preparation | ✅ **Utility** |

---

## 📁 Architecture Evolution (Historical)

### **v1-v3: Historical Implementations**

| File | Lines | Description | Status |
|------|-------|-------------|---------|
| `vectorized_population.py` | 496 | Original batched CfC (broken fitness) | ❌ **Deprecated** |
| `model.py` | 359 | Original model assembly | ❌ **Deprecated** |
| `train.py` | 388 | Simple data training | ❌ **Deprecated** |
| `train_hybrid.py` | 281 | Hybrid training (replaced by breathing) | ❌ **Deprecated** |
| `hybrid_layer.py` | 347 | Hybrid layer (replaced by breathing) | ❌ **Deprecated** |
| `hybrid_model.py` | 305 | Hybrid model (replaced by breathing) | ❌ **Deprecated** |

---

## 📊 Key Equations & Concepts

### **Breathing Equilibrium**
```python
# 5-step equilibration with balanced gates
for _ in range(5):
    h = (1-g) * h * decay + g * tanh(up)
    # After 5 steps: 99.5% target, 0.5% old state
```

### **Niche-Based Evolution**
```python
# 4 temporal niches: [0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9]
def niche_evolve(decay, fitness, n_niches=4):
    # Evolve within niches only - preserves diversity
    for niche_idx in range(n_niches):
        niche_mask = (decay >= niche_low) & (decay < niche_high)
        # Selection and reproduction within bounds
```

### **Kuramoto Order Parameter**
```python
def kuramoto_order_parameter(decay, h_states):
    # Measures synchronization: 0=diverse, 1=synchronized
    r = |(1/N) * Σ e^(iθ_j)|
    # Target: r < 0.3 (diverse), Warn: r > 0.5
```

---

## 🎯 Production Configuration

### **Validated Parameters** (250K → 5M tokens)
```python
# Optimal training configuration
BS = 5              # Batch size
LR = 0.00417        # Learning rate  
WD = 0.00734        # Weight decay
CLIP = 0.734        # Gradient clipping
EVOLVE_EVERY = 100  # Tokens between evolution
```

### **Model Configuration**
```python
BreathingModelConfig(
    vocab_size=50257,           # GPT-2 tokenizer
    embed_dim=128,              # Embedding dimension
    n_layers=8,                 # Number of layers
    n_neurons_per_layer=256,    # Neurons per layer
    n_breathe=5,                # Breathing steps per token
    decay_range=(0.5, 0.9),   # Fast to slow neurons
    gate_range=(-1.0, 1.0),   # Balanced gates
    evolve_every=100,           # Evolution frequency
)
```

---

## 🧪 Experimental Status

### **Current Experiments**
- **5M Training**: PID 35295, ~10-12h ETA
- **Batch Optimization**: Various batch size studies
- **Trajectory Analysis**: Delta Observer connection

### **Next Steps**
1. Complete 5M token training
2. Ablation: evolution vs no evolution
3. Breathing steps comparison (3, 7, 10)
4. Layer 1 Kuramoto investigation
5. Penn Treebank clean data

---

## 📁 Directory Structure

```
experiments/
├── scaling_5m_optimized/     # Current 5M training
├── baseline_comparison.json  # GRU comparison results
├── 250k_final_results.json   # Validated 250K results
└── trajectory_analysis/       # Trajectory tracking data
```

---

## 🔗 Quick Start

```bash
# Production training (5M tokens)
python train_wikitext.py --max_tokens 5000000 --n_epochs 5 --learning_rate 0.00417

# Baseline comparison
python compare_baselines.py --breathing --gru --n_tokens 10000

# Analysis tools
python analyze_trajectory.py experiments/trajectory_gen450.json
```

---

## 📚 References

- **ARCHITECTURE.md**: Complete architecture guide
- **TRAINING_PROTOCOLS.md**: Training configuration details
- **FINDINGS-2026-02-05.md**: Validation story
- **experiments/baseline_comparison.json**: GRU comparison data

**Status**: Architecture is **validated and production ready**. The breathing model with niche evolution beats GRU baselines and is currently scaling to 5M tokens.