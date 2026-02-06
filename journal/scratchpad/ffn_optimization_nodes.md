# Nodes of Interest: FFN Optimization for LFM2-350M

## Node 1: The Vec_Dot Gap

23% of MUL_MAT time goes to vec_dot fallback instead of KleidiAI. This represents ~21% of total inference time. KleidiAI handles 99% of calls but only 77% of time.

**Why it matters:** This is the clearest remaining optimization target. If we could route ALL matmuls through KleidiAI, we'd get ~1.3x speedup with no quality loss.

**Key question:** What dimensions/alignments cause vec_dot fallback?

---

## Node 2: Memory Bandwidth Ceiling

We're at 71% of theoretical DRAM bandwidth (9.2 GB/s of 13 GB/s). The missing 29% could be:
- Cache line waste
- Non-sequential access patterns
- Overhead between loads
- Hardware limitations

**Why it matters:** If we're already at hardware limits, further optimization is futile. If there's real headroom, we can target it.

**Tension with Node 1:** Vec_dot might not be the bandwidth issue - it might be accessing the same memory but slower.

---

## Node 3: The Prefetch Opportunity

While computing layer N, the weights for layer N+1 are known but not loaded. Modern CPUs have prefetch instructions. Pipeline: compute N while loading N+1.

**Why it matters:** Could hide memory latency behind compute, approaching 100% bandwidth utilization.

**Key question:** Does KleidiAI/llama.cpp already do this? If not, why not?

---

## Node 4: Alternative Accelerators

The phone has:
- Mali-G610 GPU (Vulkan support)
- MediaTek APU (NPU)

llama.cpp has Vulkan backend but it's often slower for small models. NPU would require custom work.

**Why it matters:** If CPU has hit limits, other processors might help.

**Tension with Node 2:** GPU memory bandwidth might be worse, NPU might not support Q4_0.

---

## Node 5: Model Architecture Efficiency

LFM2 has unique structure: 10 shortconv layers, 6 attention layers, 16 FFN layers. ShortConv is cheap (depthwise). FFN is expensive.

**Why it matters:** If we could shift work from FFN to shortconv (even post-training), we'd win.

**Key insight:** ShortConv does token mixing, FFN does channel mixing. They're complementary.

---

## Node 6: The Dimension Alignment Problem

KleidiAI likely requires specific tile sizes (powers of 2, multiples of NEON vector width). LFM2 dimensions:
- hidden_dim = 1024 (good)
- ffn_dim = 4608 (= 1024 * 4.5 = problematic?)
- n_heads = 16 (good)
- head_dim = 64 (good)

4608 is weird: 4608 = 512 * 9 = 576 * 8. Not a power of 2.

**Why it matters:** The "odd" dimensions might explain the vec_dot fallback.

**Key question:** What happens if we pad 4608 to 4096 or 5120?

---

## Node 7: Kernel Fusion Opportunity

Current FFN flow:
1. gate = matmul(x, W_gate)
2. up = matmul(x, W_up)
3. hidden = silu(gate) * up
4. out = matmul(hidden, W_down)

That's 3 separate matmul kernels + 2 elementwise. Each kernel has launch overhead and memory round-trips.

**Why it matters:** Fusing gate+up into one kernel, or fusing silu+multiply, could reduce overhead.

**Tension:** KleidiAI is optimized for individual matmuls. Fusion might lose those optimizations.

---

## Node 8: The Q4_0 Packing Question

Q4_0 packs 2 weights per byte (4 bits each). Each block has 32 weights + 1 FP16 scale. How does this interact with cache lines (64 bytes)?

Block size: 32 * 0.5 + 2 = 18 bytes. Not aligned to cache line.

**Why it matters:** Cache line waste could explain some of the bandwidth gap.

**Key question:** Is there a better quantization format that's more cache-friendly?

---

## Node 9: The Two-Thread Insight Revisited

We got 2.6x by using 2 threads instead of 8. But WHY does it help?

Hypothesis 1: Big cores are faster (yes, but that's not 2.6x)
Hypothesis 2: Barrier sync overhead (measured at 0.19% - not it)
Hypothesis 3: Memory bandwidth contention (8 cores fighting for 13 GB/s)
Hypothesis 4: Cache thrashing (8 cores evicting each other's L2)

**Why it matters:** Understanding the WHY might reveal further optimizations.

**New idea:** What if 1 thread is even better? Or 3 threads?

---

## Node 10: The Native Code Path

llama.cpp compiles to native ARM64. But the model weights are interpreted (dequantized at runtime). What if we compiled the model itself to native code?

Like TVM, TensorRT, or ExecuTorch - compile the model graph to optimized assembly.

**Why it matters:** Eliminates interpretation overhead, enables whole-model optimization.

**Tension:** Huge engineering effort. Requires new toolchain.

---

## Node 11: The Root Access Opportunity

We have Magisk but su isn't accessible from shell. With root we could:
- Lock CPU to max frequency (2.2 GHz instead of 1.3-1.5 GHz)
- Pin process to big cores
- Disable thermal throttling (dangerous)
- Access GPU/NPU directly

**Why it matters:** Could be another 1.5x just from frequency.

**Key question:** How to get shell root access via Magisk?

---

## Node 12: The Batch Size Question

We're running batch=1 (single token). What if we batched multiple sequences?

Batch=1: 209 MB model, 1 token output
Batch=8: 209 MB model, 8 tokens output (same bandwidth, 8x throughput)

**Why it matters:** Throughput scales with batch size (if you have parallel requests).

**Tension:** Latency stays the same. Only helps throughput-oriented workloads.

---

## Node 13: The Speculative Decoding Option

Use a tiny model (50M params?) to draft tokens, then verify with 350M model. If draft is often correct, you skip most 350M compute.

**Why it matters:** Could be 2-3x speedup if draft model is accurate.

**Key question:** Does a tiny LFM2 exist? Could we create one via distillation?

---

## Node 14: The GGML vs Other Runtimes Question

We're using llama.cpp/GGML. Other options:
- MLC-LLM (Apache TVM-based)
- ExecuTorch (PyTorch/Meta)
- ONNX Runtime
- TensorRT (NVIDIA, not applicable)

**Why it matters:** Different runtimes have different optimizations.

**Key question:** Has anyone benchmarked LFM2 on other runtimes for mobile?

---

## Node 15: The Model Surgery Option

Instead of optimizing inference, change the model:
- Remove layers (less bandwidth)
- Shrink FFN (narrower = faster)
- Replace attention with linear attention
- Share weights across layers

All require retraining or fine-tuning.

**Why it matters:** Algorithmic changes can give 2-10x. Inference optimization gives 1.1-1.5x.

**Tension:** We want POST-TRAINING optimization. Model surgery needs training.
