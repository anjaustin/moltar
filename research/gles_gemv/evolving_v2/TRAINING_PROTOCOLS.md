# Training Protocols: Evolving SRC-FFN

**Protocol Version**: 1.0  
**Last Updated**: 2026-02-06  
**Status**: Production Ready  

---

## 🎯 Quick Reference

### **Validated Parameters** (250K → 5M tokens)
```bash
# Optimal configuration (validated on 250K tokens)
BS=5              # Batch size  
LR=0.00417        # Learning rate
WD=0.00734        # Weight decay  
CLIP=0.734        # Gradient clipping
EVOLVE_EVERY=100  # Tokens between evolution
```

### **Training Commands**
```bash
# Small test (fast validation)
python train_wikitext.py --max_tokens 10000 --n_epochs 1 --learning_rate 0.00417

# Medium scale (development)
python train_wikitext.py --max_tokens 100000 --n_epochs 3 --learning_rate 0.00417

# Production scale (current)
python train_wikitext.py --max_tokens 5000000 --n_epochs 5 --learning_rate 0.00417
```

---

## 📊 Parameter Evolution

### **Historical Progression**
| Training | BS | LR | WD | Clip | Result | Status |
|----------|----|----|----|----|--------|---------|
| 10K test | 2 | 0.002 | - | - | PPL=2,108 | Initial |
| 50K test | 2 | 0.002 | - | - | PPL=2,086 | Baseline |
| **250K** | **5** | **0.00417** | **0.00734** | **0.734** | **PPL=924** | ✅ **Validated** |
| 5M (current) | 5 | 0.00417 | 0.00734 | 0.734 | *In Progress* | 🔄 Running |

### **Parameter Rationale**
- **BS=5**: Balance between gradient noise and memory efficiency
- **LR=0.00417**: Higher than standard (0.001) due to evolution regularization
- **WD=0.00734**: Moderate regularization for 6.6M parameters
- **Clip=0.734**: Prevents gradient explosions during breathing steps

---

## 🏗️ Model Configuration

### **Breathing Architecture** (Recommended)
```python
from breathing_model import BreathingModelConfig

config = BreathingModelConfig(
    vocab_size=50257,           # GPT-2 tokenizer (fixed)
    embed_dim=128,              # Embedding dimension
    n_layers=8,                 # Number of layers
    n_neurons_per_layer=256,    # Neurons per layer (2,048 total)
    n_breathe=5,                # Breathing steps per token
    decay_range=(0.5, 0.9),   # Fast (0.5) to slow (0.9) neurons
    gate_range=(-1.0, 1.0),   # Balanced gates
    evolve_every=100,           # Evolution frequency
    
    # Evolution parameters
    selection_pressure=0.5,      # Top 50% survive
    elite_fraction=0.1,       # Protected from mutation
    mutation_rate=0.3,        # Probability of mutation
    mutation_strength=0.1,    # Scale of mutations
    sexual_rate=0.3,          # Crossover vs cloning
)
```

### **Scaling Guidelines**
| Scale | Layers | Neurons/Layer | Total Neurons | Embed Dim | Use Case |
|-------|--------|---------------|---------------|-----------|----------|
| **Small** | 4 | 64 | 256 | 64 | Testing |
| **Medium** | 8 | 128 | 1,024 | 128 | Development |
| **Large** | 8 | 256 | 2,048 | 128 | Production |
| **XL** | 8 | 512 | 4,096 | 256 | Research |

---

## ⏱️ Training Schedule

### **Standard Protocol** (Recommended)
```python
# 5M token training schedule
def training_schedule():
    max_tokens = 5_000_000
    n_epochs = 5
    
    # Progress tracking every 1000 tokens
    log_every = 1000
    save_every = 20_000
    
    # Evolution every 100 tokens  
    evolve_every = 100
    
    return {
        'max_tokens': max_tokens,
        'n_epochs': n_epochs,
        'log_every': log_every,
        'save_every': save_every,
        'evolve_every': evolve_every
    }
```

### **Checkpoint Schedule**
- **Every 20K tokens**: Save checkpoint
- **Every 100K tokens**: Full evaluation
- **Every 1M tokens**: Major milestone analysis

---

## 📈 Monitoring Protocol

### **Real-time Metrics**
```bash
# Monitor training progress
tail -f experiments/scaling_5m_optimized.log | grep -E "^[0-9]K:|T=|PPL=|Acc=|Evols="

# Check process health
ps -p 35295 -o pid,pcpu,pmem,etime
```

### **Key Indicators**
| Metric | Target | Warning | Action |
|--------|--------|---------|---------|
| **Perplexity** | < 1000 | > 2000 | Check learning rate |
| **Accuracy** | > 10% | < 5% | Increase training |
| **Kuramoto r** | < 0.3 | > 0.5 | Check diversity |
| **Speed** | > 100 tok/s | < 50 tok/s | Check batch size |
| **Memory** | < 80% | > 95% | Reduce scale |

### **Evolution Health**
- **Evolutions**: ~1 per 100 tokens (target: 50K for 5M tokens)
- **Diversity**: Maintain decay spread across niches
- **Birth/Death**: Should see continuous turnover

---

## 🔄 Recovery Protocols

### **Training Divergence**
```bash
# If perplexity explodes (> 5000)
1. Reduce learning rate: LR *= 0.5
2. Increase gradient clip: clip *= 1.5  
3. Restart from last checkpoint
```

### **Evolution Collapse**
```bash
# If Kuramoto r > 0.5 (synchronized)
1. Increase mutation rate: 0.3 → 0.5
2. Add diversity injection
3. Check niche balance
```

### **Memory Issues**
```bash
# If GPU memory > 95%
1. Reduce batch size: BS=5 → BS=3
2. Reduce embedding dim: 128 → 96
3. Reduce neurons: 256 → 192
```

---

## 🧪 Experimental Variants

### **Ablation Studies**
```bash
# Evolution vs no evolution
python compare_baselines.py --with_evolution --without_evolution

# Breathing steps comparison  
python compare_baselines.py --breathe_steps 3,5,7,10

# Niche count comparison
python compare_baselines.py --niches 2,4,6,8
```

### **Data Variants**
```bash
# Clean data (recommended)
python train_wikitext.py --data_clean --max_tokens 100000

# Penn Treebank
python train_ptb.py --max_tokens 100000
```

---

## 📊 Expected Results

### **5M Token Targets**
| Metric | Current (250K) | Target (5M) | Improvement |
|--------|-----------------|-------------|-------------|
| **Perplexity** | 924 | < 800 | > 13% |
| **Accuracy** | 12.4% | > 15% | > 20% |
| **Evolutions** | 250 | ~5000 | 20x |

### **Timeline Estimates**
| Training Size | Time | Checkpoints | Status |
|---------------|------|-------------|---------|
| 100K tokens | 15 min | 5 | ✅ Available |
| 250K tokens | 35 min | 12 | ✅ Completed |
| 1M tokens | 2.5 hours | 50 | 🔄 Ready |
| **5M tokens** | **12 hours** | **250** | **🔄 Running** |

---

## 🔧 Troubleshooting

### **Common Issues**
1. **"Loaded 0 tokens from cache"** → Run `prepare_wikitext.py` first
2. **"CUDA out of memory"** → Reduce batch size or model scale
3. **"Evolution not happening"** → Check `evolve_every` parameter
4. **"No diversity"** → Monitor Kuramoto order parameter

### **Debug Commands**
```bash
# Check training status
ps aux | grep train_wikitext

# Check log for errors  
tail -50 experiments/scaling_5m_optimized.log | grep -i error

# Check checkpoint health
python analyze_checkpoint.py experiments/scaling_5m_optimized/checkpoint_e0_t100000.pt
```

---

## 📚 References

- **ARCHITECTURE.md**: Complete architecture guide
- **experiments/250k_final_results.json**: Validated 250K results
- **experiments/baseline_comparison.json**: GRU comparison data
- **PID 35295**: Current 5M training process

**Status**: Protocol is **validated and production ready**. 5M training currently running with expected completion in ~10 hours.