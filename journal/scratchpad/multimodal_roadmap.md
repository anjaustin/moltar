# Multi-Modal Roadmap: Liquid AI LFM2 Family on TriX Fabric

## The Landscape

Liquid AI ships a complete model family, all sharing the LFM2 1.2B backbone we already optimize:

```
                     LFM2 1.2B Backbone
                    (hybrid conv+attention)
                           |
              ┌────────────┼────────────────┐
              |            |                |
         LFM2-Text    LFM2-Audio       LFM2-VL
         (350M-1.2B)  (1.5B total)     (1.6B total)
              |            |                |
         Pure LLM     Speech ↔ Speech   Image → Text
         Instruct     ASR + TTS + Chat  Vision-Language
         Thinking     Interleaved       llama.cpp native
```

All use the same `lfm2` GGUF architecture. All run through llama.cpp.

---

## Model Inventory + Device Memory Budget

### Device: 3.6 GB RAM total (~2.5 GB usable after OS)

| Model | Files | Q4_0 Size | Q8_0 Size | F16 Size |
|-------|-------|-----------|-----------|----------|
| **LFM2.5 Text (1.2B)** | 1 GGUF | 696 MB | 1.25 GB | 2.34 GB |
| **LFM2.5 Audio (1.5B)** | 4 GGUFs (see below) | **856 MB** | 1.83 GB | 3.33 GB |
| **LFM2.5 VL (1.6B)** | 1 GGUF | 696 MB | 1.25 GB | 2.34 GB |
| **LFM2 Text (350M)** | 1 GGUF | 209 MB | — | — |

### LFM2.5-Audio-1.5B GGUF Components (Q4_0)

| File | Purpose | Size |
|------|---------|------|
| `LFM2.5-Audio-1.5B-Q4_0.gguf` | LLM backbone (text+audio token generation) | 696 MB |
| `mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf` | Audio encoder (FastConformer) | 220 MB |
| `vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf` | Audio output (Mimi decoder) | 109 MB |
| `tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf` | Audio tokenizer (speaker file) | 50.5 MB |
| **Total** | | **1,076 MB** |

**Correction**: Q4_0 mmproj is 220 MB, not 50 MB from the earlier card listing. Total audio Q4_0 = 1,076 MB. Still fits.

### Feasible Configurations on 2.5 GB Usable

| Configuration | Total RAM | Fits? | Use Case |
|--------------|-----------|-------|----------|
| Audio Q4_0 alone | 1,076 MB | YES | Voice assistant |
| Text 350M + Audio Q4_0 | 1,285 MB | YES | Fast text + voice |
| Text 1.2B Q4_0 alone | 696 MB | YES | Smart text (current) |
| VL Q4_0 alone | 696 MB | YES | Image understanding |
| Audio Q4_0 + VL Q4_0 | 1,772 MB | TIGHT | Voice + vision |
| All three Q4_0 | 2,468 MB | NO | Too much RAM |

**Key insight**: We can run any TWO modalities concurrently at Q4_0. All three is too tight.

---

## Runtime Architecture

### Two Paths to Audio

**Path A: Prebuilt Runner (Fast, No Build)**
Liquid AI ships `llama-liquid-audio-android-arm64.zip` — prebuilt binaries for Android ARM64.
- `llama-liquid-audio-cli` — CLI for ASR, TTS, interleaved chat
- `llama-liquid-audio-server` — HTTP server with audio I/O

These are built from llama.cpp PR #18641 (draft, not merged upstream yet).
No compilation needed. Download, unzip, deploy.

**Path B: Build from PR #18641 (Full Control)**
Checkout the WIP branch, cross-compile with our NDK toolchain.
Our existing build flags (`armv8.2-a+dotprod`, KleidiAI, Vulkan off) apply.
This gives us full control for fabric integration.

### Text + VL Path (Already Working)

Our existing llama.cpp build already has:
- `lfm2.cpp` model support (confirmed in source tree)
- `conformer.cpp` audio encoder (merged in PR #18106, Dec 2025)
- `mtmd` multimodal tools (vision + audio preprocessing)

For text-only upgrades (LFM2.5-1.2B-Instruct), our existing `llama-completion` binary should work as-is — just swap the GGUF file.

For VL, we need to build `llama-mtmd-cli` from the tools directory.

---

## Capabilities by Model

### LFM2.5-Audio-1.5B — Speech-to-Speech

Three generation modes, all from one model:

| Mode | System Prompt | Input | Output |
|------|--------------|-------|--------|
| **ASR** | "Perform ASR." | Audio WAV | Text (capitalized, punctuated) |
| **TTS** | "Perform TTS. Use the US male voice." | Text | Audio WAV (24kHz) |
| **Interleaved** | "Respond with interleaved text and audio." | Audio or text | Text + audio (real-time) |

Four TTS voices: US male, US female, UK male, UK female.

Competitive ASR: 7.24 WER average (vs Whisper-large-V3's 7.93).
At 1.5B params — smaller than Whisper-large but better accuracy.

### LFM2.5-VL-1.6B — Vision-Language

Image input → text output. Works with `llama-cli -hf LiquidAI/LFM2.5-VL-1.6B-GGUF:Q4_0`.

### LFM2.5-1.2B-Instruct/Thinking — Upgraded Text

Drop-in replacement for our LFM2 text models. Same architecture, better training.
"Thinking" variant: chain-of-thought reasoning mode.

---

## The Fabric Integration Plan

### Phase 1: Prove Audio Works (Priority 1)

**Goal**: LFM2.5-Audio running on device through the TriX fabric.

1. Download prebuilt `llama-liquid-audio-android-arm64.zip`
2. Download Q4_0 audio GGUFs (4 files, 1,076 MB total)
3. Deploy to device
4. Test ASR: feed WAV → get text
5. Test TTS: feed text → get WAV  
6. Test interleaved: feed WAV → get text + WAV
7. Measure: tokens/sec, latency to first audio, total generation time
8. Run through fabric daemon: pin to big cores, pre-fault models

**Success criteria**: ASR produces correct text. TTS produces audible speech. Fabric improves speed.

### Phase 2: Upgrade Text Model (Priority 2)

**Goal**: Replace LFM2-350M with LFM2.5-1.2B-Instruct for smarter text generation.

1. Download LFM2.5-1.2B-Instruct GGUF (Q4_0, 696 MB)
2. Deploy, test with existing `llama-completion` binary
3. Benchmark: generation tok/s, prompt eval, quality
4. Compare against 350M baseline (speed vs intelligence tradeoff)
5. Update fabric daemon to manage model selection

**Note**: 1.2B is 3.3x larger than 350M. Expect ~15 tok/s (from Phase 4 testing) vs 44 tok/s. The tradeoff is quality.

### Phase 3: Multi-Model Fabric Daemon v0.5 (Priority 3)

**Goal**: Fabric manages multiple models — hot-swap, memory-aware scheduling.

The fabric daemon evolves from "optimize one model" to "orchestrate many":

```
Fabric Daemon v0.5
├── Memory Manager
│   ├── mmap all models (lazy load)
│   ├── mlock active model(s)
│   ├── madvise(DONTNEED) for cold models
│   └── Track RSS, predict pressure
├── Model Scheduler  
│   ├── Active model selection (text vs audio vs VL)
│   ├── Core pinning per model
│   ├── Pipeline: STT → LLM → TTS
│   └── Pre-warm next model on context switch
├── CfC Controller (Q15, existing)
│   ├── System state observer (temps, freqs)
│   ├── Throughput tracker per model
│   └── Adaptive tick rate (10-100 Hz)
└── Hardware Layer (existing)
    ├── Big core pinning (A78 × 2)
    ├── Pre-fault + mlock
    └── sysfs monitoring
```

### Phase 4: Vision (Priority 4)

**Goal**: Add image understanding.

1. Build `llama-mtmd-cli` from our llama.cpp source
2. Download LFM2.5-VL-1.6B-GGUF (Q4_0, 696 MB)
3. Deploy, test image → text
4. Integrate into fabric model scheduler

### Phase 5: Full Pipeline Demo (Stretch)

**Goal**: Voice in → understand → reason → voice out, managed by the fabric.

```
User speaks → [Mic capture]
    → LFM2.5-Audio ASR → text
    → LFM2.5-1.2B-Instruct → reasoning text  
    → LFM2.5-Audio TTS → speech
    → [Speaker output]

OR (simpler, single model):

User speaks → [Mic capture]
    → LFM2.5-Audio interleaved → text + speech simultaneously
    → [Speaker output]
```

The single-model path (interleaved) is actually simpler and lower latency.

---

## Critical Technical Questions

### Q1: Does the prebuilt Android runner work on our SoC?

The runner targets `android-arm64`. Our device is `aarch64` with `armv8.2-a+dotprod`. Should work, but they removed "buggy armv9 backends" — suggesting they've tested on multiple ARM variants. Need to verify.

### Q2: Does the audio backbone share weights with text?

The audio model's LLM backbone IS an LFM2 1.2B. If we're running audio, we get text capability for free — the model handles both. We may not need a separate text model at all for the basic pipeline.

### Q3: What's the latency budget for real-time conversation?

Interleaved mode needs to generate first audio token fast. On server hardware this is sub-second. On our A78 cores at ~15 tok/s (1.2B estimate), first audio token might take 100-200ms. Acceptable for voice assistant, but not for real-time conversation feel.

### Q4: Can we build from the WIP PR ourselves?

PR #18641 is draft but the source is available. If the prebuilt runner has issues on our SoC, we can build from source with our proven NDK + KleidiAI toolchain.

### Q5: Can audio and text models share the backbone?

Both use LFM2 1.2B GGUF. If the weights are identical (same backbone), we could theoretically mmap the same file and save 696 MB. Need to verify file hashes.

---

## What NOT to Do

1. **Don't build GPU support for any of this.** GPU investigation is closed.
2. **Don't try to run all three modalities simultaneously.** RAM won't fit.
3. **Don't start with building from source.** Try the prebuilt runner first.
4. **Don't design the multi-model fabric before proving single-model audio works.**
5. **Don't forget the fabric.** Every model runs THROUGH the daemon. The daemon makes it fast.

---

## Immediate Next Action

**Download and deploy LFM2.5-Audio prebuilt runner + Q4_0 GGUFs to device.**

Files needed:
```
# Runner (4.88 MB)
runners/llama-liquid-audio-android-arm64.zip

# Model files (1,076 MB total)
LFM2.5-Audio-1.5B-Q4_0.gguf          (696 MB)
mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf   (220 MB)  
vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf  (109 MB)
tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf (50.5 MB)
```

Total download: ~1.08 GB. Device has space (we have ~2 GB free after existing models).
