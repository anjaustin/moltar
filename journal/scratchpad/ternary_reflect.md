# Reflections: Ternary/Sparse Kernels for LFM2 On-Device

## The Structure Beneath

Looking at the 12 nodes, three clusters emerge:

**Cluster A: The Physics (Nodes 1, 2, 8)**
The device is memory-bandwidth-bound. The weights are peaked at zero. We can't profile. These are the immovable constraints — the grain of the wood.

**Cluster B: The Approaches (Nodes 4, 5, 6, 9, 11)**
Multiple paths from "do nothing new" to "full ternary revolution." Each trades engineering effort for potential speedup. The tension is: which path maximizes learning per unit of effort?

**Cluster C: The Purpose (Nodes 7, 10, 12)**
What are we actually trying to do? Probe first. Deploy via LD_PRELOAD. Validate Yinsen's architecture.

## Asking "Why" Three Times

**Why do we want ternary kernels?**
Because 35% of Q4_0 weights are already ternary-compatible, and ternary means zero multiplies.

**Why does zero-multiply matter?**
Because... wait. On a memory-bandwidth-bound device, the ALU is already waiting. Making the ALU faster doesn't help if it's waiting for data. Zero-multiply saves power but not wall-clock time on this device.

**Then why does this matter at all?**
Because ternary encoding is 2 bits, not 4. That means **half the data to read from memory.** The zero-multiply aspect is a bonus — the real win is that ternary-encoded models are half the size.

This is the core insight. The zero-multiply is not the point. The 2-bit encoding is the point. Everything flows from Node 1: bandwidth is the bottleneck.

## Resolving the Key Tensions

### Node 3 vs Node 11: Re-quantize vs Use Existing Q2_K?

This is the central tension. llama.cpp already has 2-bit quantization formats (IQ2_XXS at 2.06 bpw, Q2_K at 2.63 bpw). They were designed by smart people with importance matrices and careful calibration. Why would our ternary approach be better?

**Resolution:** They might NOT be better in quality. But they're different in two ways:
1. **Compute path:** Q2_K still dequantizes to float and multiplies. Ternary adds/subtracts. On the A78, this difference is small because we're bandwidth-bound. But it's non-zero — fewer cycles per element means the core finishes each memory-fetch-worth of work faster, keeping the memory bus maximally utilized.
2. **Yinsen's kernel architecture:** The blocked layouts, prefetch patterns, and ghost-stream LDNP are all designed for ternary data. They're battle-tested on Apple Silicon at 186 GOP/s. Q2_K through KleidiAI uses whatever Arm decided to do, which may or may not be optimal for this specific SoC.

**The real resolution:** Do BOTH. Quantize to Q2_K and benchmark. Quantize to ternary and benchmark. The device will tell us which wins. Probe first.

### Node 5 vs Node 1: Yinsen's Format vs Bandwidth Reduction

Yinsen's kernels expect ternary or int8. Q4_0 is neither. Converting Q4_0 to int8 at load time doubles the data. Converting to ternary loses quality.

**Resolution:** Don't convert at inference time. Convert at **preparation time.** Create a new model file:
1. Read the Q4_0 GGUF
2. For each block of 32 weights, dequantize to float
3. Re-quantize to ternary using the block's scale: threshold = scale * T (where T is a tunable parameter)
4. Pack as 2-bit (Yinsen's format)
5. Write a new GGUF (or custom binary) with 2-bit packed weights

This is a one-time offline cost. The on-device model file is half the size. Yinsen's kernels consume it natively. No runtime format conversion.

### Node 8: Blind Tuning?

We can't profile. But we CAN measure wall-clock matvec time with high precision using `clock_gettime(CLOCK_MONOTONIC)`. And we can vary one thing at a time (block size, prefetch distance, LDNP vs LDR). We're not blind — we just have coarser instruments.

### Node 12: Speed vs Validation?

**Resolution:** They're the same thing if we do it right. If Yinsen's ternary kernels are faster than KleidiAI's Q2_K on this device, that IS the validation. If they're slower, that's also information. The probe gives us both.

## What Would This Look Like If It Were Easy?

1. Take LFM2-350M-Q4_0.gguf
2. Run a script that converts every Q4_0 tensor to ternary-packed (2-bit)
3. Load the ternary model into a standalone benchmark harness
4. Run a single matvec (e.g., ffn_gate: 1024 x 4608) using Yinsen's `neon_ternary_matvec_blocked8`
5. Compare wall-clock to the same matvec using ggml's Q4_0 path
6. The numbers speak

That's actually... not hard. The conversion is straightforward. Yinsen's kernels already exist. The benchmark is a single function call.

## What I Now Understand

The path is a **three-way matvec shootout on the actual device:**

1. **Q4_0 via KleidiAI** (current baseline)
2. **Q2_K via KleidiAI** (llama.cpp's own aggressive quant)
3. **Ternary via Yinsen's NEON kernels** (2-bit packed, blocked layout)

All three are ~2-bit effective. The question is which combination of encoding + kernel + cache strategy wins on this specific memory subsystem.

The probe is a single C file. It loads one tensor from the GGUF, converts it to all three formats, and times each matvec 1000 times. No llama.cpp involved. Pure matvec throughput measurement.

**Then** — armed with numbers — we decide whether to build the full LD_PRELOAD library or just use Q2_K through the existing pipeline.

## Remaining Questions

- What threshold T gives the best quality when converting Q4_0 to ternary? The sparsity data suggests T=1.5 (map quant values 7,8,9 to {-1,0,+1}) captures 35% perfectly, but what about 6,10 (|val|=2)?
- Does Q2_K even work with LFM2 in llama.cpp? Need to check if the quantization tool supports the lfm2 architecture.
- How does Yinsen's `neon_ternary_matvec_sdot` compare to `neon_ternary_matvec_blocked8`? The blocked variant should win on this device because of its cache optimization, but the sdot variant is simpler.
- For the ghost-stream LDNP kernels: does Android's kernel even allow non-temporal load hints on this SoC, or does it ignore them?

## The Surprise

The surprise is that this isn't really about ternary at all. It's about **bits per weight on a bandwidth-bound device.** Ternary is one way to get to 2 bpw. Q2_K is another. The right answer might be Q2_K through KleidiAI with zero new code. But the probe will tell us if Yinsen's cache-aware kernels can beat Arm's own library — and if they can, THAT is the validation of the entire Yinsen architecture.
