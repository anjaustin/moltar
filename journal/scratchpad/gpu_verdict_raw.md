# Raw Thoughts: GPU Verdict on Dimensity 930

## Stream of Consciousness

We spent 3 probes and a full pipeline prototype trying to find a use for the PowerVR BXM-8-256 GPU in our LFM2 inference pipeline. The results are brutally clear: the GPU has no profitable role in single-token generation for this model on this hardware.

My gut reaction when we started was excitement — "there's a whole GPU sitting idle, surely we can use it." The Probe A results (160us timeline dispatch) made it look promising. Probe B (93.9% hidden) looked like a slam dunk. But I was fooling myself. I was measuring the GPU in isolation, not in the context of the actual computation graph.

The fundamental problem is that CPU activations (SwiGLU, RMSNorm) are trivially cheap — 25us and 1us respectively — while GPU dispatch overhead is 330us minimum. The GPU is 13x slower at SwiGLU than the CPU. Even with perfect overlap, you're replacing a 25us CPU op with a 330us GPU op that STILL has to complete before the next matvec can start.

What scares me: we might be done with hardware optimization. The fabric v0.4 already captures ~99% of the theoretical throughput — 44 tok/s vs 44.1 tok/s bare taskset. The remaining 0.1 tok/s is noise. Where do you go from here?

What about prefill? Probe C killed that too — GPU matmul is 19x slower than CPU at every batch size. No crossover. Ever.

The only remaining hardware lever is memory bandwidth. CPU matvec at 758us for a 2.65MB weight tensor = 3.5 GB/s effective bandwidth per core. With 2 cores RAID 0 = 6.6 GB/s. Device LPDDR4X theoretical is ~13 GB/s. We're at 51% utilization. Could we squeeze more? Maybe, but KleidiAI + SDOT is already the state of the art for this ISA.

What if the model changes? LFM2 is a hybrid — 10 ShortConv + 6 Attention layers. The attention layers have different compute patterns (KV cache, softmax). But the attention layers are still dominated by matvecs. Same bottleneck.

What about other models? If someone runs a model with massive activation functions (huge MLP, expensive normalization), the calculus might change. But for LFM2 architecture, activations are < 2% of compute.

The fabric daemon is doing what it should — it's a hardware optimizer. Core pinning, memory prefaulting, adaptive tick rate, Q15 CfC neural controller. These are real, measurable wins. The GPU investigation was the right thing to do (probe first!), and the answer was "no."

## Questions Arising

- Is the GPU useful for ANYTHING on this device? (Not inference, maybe other tasks?)
- What about the TMU/texture path we probed earlier? Linear interpolation for LUT activations?
- Should we accept 44 tok/s as the ceiling and shift focus to quality/features?
- What about the 700M and 1.2B models? Different compute/memory ratios?
- Is there a way to reduce DRAM bandwidth demand? (Better quantization, weight reordering, cache-aware tiling?)
- What about the ShortConv layers? They're 1D convolutions — different compute pattern than matvecs.
- The daemon CfC controller doesn't actually DO anything yet beyond initial scheduling. Can it be smarter?
- Should we focus on multi-model support (switching between 350M/700M/1.2B based on context)?

## First Instincts

- Accept the GPU verdict: it's useless for this workload on this hardware
- The remaining optimization space is narrow: memory bandwidth, cache behavior, quantization format
- Maybe shift from "go faster" to "use less" — power/thermal optimization for sustained performance
- The CfC controller could learn thermal throttling patterns and pre-emptively adjust
- The real product is the daemon — reliable, measurable, does-one-thing-well hardware optimization
