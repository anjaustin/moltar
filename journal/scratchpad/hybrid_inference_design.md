# Hybrid Inference Architecture: A78 Responders + A55 Thinkers

## The Insight

Human cognitive tasks have different latency requirements:

| Task Type | Latency Need | Example |
|-----------|--------------|---------|
| **Reactive** | <100ms | Tool calls, confirmations, UI response |
| **Interactive** | <500ms | Short answers, completions |
| **Generative** | >1s ok | Long explanations, creative writing |
| **Background** | Minutes ok | Summarization, batch processing |

## Hardware Mapping

| Core | First Token | Throughput | Power | Best For |
|------|-------------|------------|-------|----------|
| A78 | **23ms** | 44 tok/s | High | Reactive, Interactive |
| A55 | 590ms | 10 tok/s | Low | Generative, Background |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY CLASSIFIER                            │
│   (Could be rule-based or tiny classifier model)                │
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   REACTIVE  │    │ INTERACTIVE │    │  GENERATIVE │
    │   A78 x 2   │    │   A78 x 1   │    │   A55 x 2   │
    │   48 tok/s  │    │   44 tok/s  │    │   18 tok/s  │
    │   ~23ms TTF │    │   ~23ms TTF │    │  ~590ms TTF │
    └─────────────┘    └─────────────┘    └─────────────┘
           │                  │                  │
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    RESPONSE STREAMER                         │
    │   (Unified output regardless of which path generated it)    │
    └─────────────────────────────────────────────────────────────┘
```

## Classification Heuristics

### REACTIVE (A78 fast path)
- Query contains: "what time", "call", "send", "open", "set"
- Expected output: <20 tokens
- Contains tool/function call patterns
- User waiting for immediate response

### INTERACTIVE (A78 standard)
- Query is a question expecting short answer
- Expected output: 20-100 tokens
- Conversational turn-taking

### GENERATIVE (A55 efficient path)
- Query contains: "explain", "write", "describe", "list"
- Expected output: >100 tokens
- User will be reading for a while anyway
- Creative/long-form tasks

### BACKGROUND (A55 batch)
- Summarization of documents
- Batch processing
- Pre-computation for later
- Screen off / idle processing

## Handoff Strategy

For some queries, **start fast, continue slow**:

```
User: "Explain how photosynthesis works"

1. A78 FAST START (23ms to first token)
   Generates: "Photosynthesis is the process by which plants"
   
2. HANDOFF after ~10 tokens (user is now reading)
   
3. A55 CONTINUES (power efficient)
   Generates: "convert light energy into chemical energy..."
   
Result: Snappy first response + efficient long generation
```

This gives **perceived low latency** with **actual high efficiency**.

## Implementation Options

### Option A: Dual Model Instances
- Keep model loaded on both A78 and A55 pools
- Route queries to appropriate pool
- Requires 2x model memory (~380MB)

### Option B: Single Model, Core Migration  
- Start on A78
- After N tokens, migrate thread to A55
- Tricky: state handoff, scheduler cooperation

### Option C: Speculative Dual-Path
- Start both paths simultaneously
- A78 serves while A55 warms up
- Kill A55 if query is short
- Handoff to A55 if query is long

### Option D: Predictive Warm-up
- Classify query while user is typing
- Pre-warm appropriate cores
- Route when query is submitted

## Power Budget Analysis

Assume continuous assistant usage:
- 50% reactive/interactive, 50% generative
- Current: All A78 = 100% power baseline

With hybrid:
- 50% on A78 @ 100% power
- 50% on A55 @ ~40% power
- Blended: 50% + 20% = **70% power** 

**30% battery savings with minimal UX impact!**

## The Reading Speed Reality Check

| Output Length | A78 Time | A55 Time | Human Read Time |
|---------------|----------|----------|-----------------|
| 50 tokens | 1.1s | 2.8s | **10s** |
| 200 tokens | 4.5s | 11s | **40s** |
| 500 tokens | 11s | 28s | **100s** |

For 200+ token outputs, the human is the bottleneck, not the model!

A55 at 18 tok/s still **outruns human reading by 3.6x**.

## Next Steps

1. **Measure actual power draw** (need root or external meter)
2. **Implement query classifier** (can be simple regex/heuristics)
3. **Test handoff latency** (thread migration cost)
4. **User study**: Do users notice A55 path for long content?

## The Moneyball Reframe

Traditional thinking: "Make everything as fast as possible"

Moneyball thinking: "Match processing speed to consumption speed"

Nobody needs a 48 tok/s firehose when they're reading at 4 tok/s.
But everyone notices a 600ms delay on "What's 2+2?"

**Use the right tool for the right job.**
