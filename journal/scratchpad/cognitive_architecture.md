# Cognitive Architecture: Heterogeneous Inference System

## Hardware Mapping

**Dimensity 7020 Resources:**
- **2x Cortex-A78** @ 2.2GHz (big cores)
- **6x Cortex-A55** @ 2.0GHz (little cores)  
- **IMG BXM-8-256 GPU** (PowerVR, not Mali)
- **4GB LPDDR4X** @ ~10 GB/s

## The Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COGNITIVE ARCHITECTURE                              │
│                        Moto G Power 5G (2023)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              ┌──────────────┐                                │
│                              │    A78-0     │                                │
│                              │  EXECUTIVE   │                                │
│                              │              │                                │
│                              │ - Routing    │                                │
│                              │ - Planning   │                                │
│                              │ - Attention  │                                │
│                              │ - Scheduling │                                │
│                              └──────┬───────┘                                │
│                                     │                                        │
│         ┌───────────────────────────┼───────────────────────────┐            │
│         │                           │                           │            │
│         ▼                           ▼                           ▼            │
│  ┌─────────────┐            ┌─────────────┐            ┌──────────────┐      │
│  │   A78-1     │            │  A55 POOL   │            │  PowerVR GPU │      │
│  │ RESPONDER   │            │  (6 cores)  │            │  BXM-8-256   │      │
│  │             │            │             │            │              │      │
│  │ - LLM Gen   │            │ - Memory    │            │ - Embeddings │      │
│  │ - 48 tok/s  │            │ - Graph DB  │            │ - Vector ops │      │
│  │ - Tools     │            │ - Zep-like  │            │ - Similarity │      │
│  │ - <25ms TTF │            │ - RAG prep  │            │ - Batch GEMV │      │
│  │             │            │ - Summaries │            │              │      │
│  │ FAST PATH   │            │ SLOW PATH   │            │ PARALLEL PATH│      │
│  └─────────────┘            └─────────────┘            └──────────────┘      │
│         │                           │                           │            │
│         │                    ┌──────┴──────┐                    │            │
│         │                    │             │                    │            │
│         │                    ▼             ▼                    │            │
│         │            ┌───────────┐ ┌───────────┐                │            │
│         │            │ A55 0-1   │ │ A55 2-3   │                │            │
│         │            │ Graph Ops │ │ Vector DB │                │            │
│         │            └───────────┘ └───────────┘                │            │
│         │                    │             │                    │            │
│         │            ┌───────────┐         │                    │            │
│         │            │ A55 4-5   │         │                    │            │
│         │            │ Summaries │         │                    │            │
│         │            └───────────┘         │                    │            │
│         │                    │             │                    │            │
│         └────────────────────┴─────────────┴────────────────────┘            │
│                                     │                                        │
│                                     ▼                                        │
│                          ┌──────────────────┐                                │
│                          │   SHARED STATE   │                                │
│                          │                  │                                │
│                          │ - Context Window │                                │
│                          │ - Memory Graph   │                                │
│                          │ - Embedding Cache│                                │
│                          │ - Conversation   │                                │
│                          └──────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### A78-0: Executive (The Coordinator)

**Role:** Prefrontal cortex - planning, routing, attention management

**Tasks:**
- Classify incoming queries (reactive vs generative vs memory)
- Dispatch to appropriate subsystem
- Manage attention/priority
- Aggregate results from subsystems
- Handle interrupts and context switches

**Characteristics:**
- Always ready (low sleep latency)
- Lightweight inference for routing decisions
- State machine for conversation flow

### A78-1: Responder (The Speaker)

**Role:** Working memory and speech production - user-facing generation

**Tasks:**
- LLM token generation at 48 tok/s
- Tool calling and execution
- Real-time response streaming
- First-token latency critical (<25ms)

**Characteristics:**
- Full bandwidth for generation
- Preemptable only for higher priority
- Hot path optimization

### A55 Pool: Memory System (The Hippocampus)

**Role:** Long-term memory, knowledge consolidation, background processing

**Tasks by Core Pair:**

| Cores | Function | Example |
|-------|----------|---------|
| A55 0-1 | Graph Operations | Entity relationships, concept links |
| A55 2-3 | Vector Database | Similarity search, RAG retrieval |
| A55 4-5 | Summarization | Compress conversations, extract facts |

**Characteristics:**
- Latency tolerant (background)
- Power efficient
- Can process during screen-off
- 18 tok/s aggregate for LLM tasks

### PowerVR GPU: Vector Engine (Sensory Processing)

**Role:** Parallel numerical operations

**Tasks:**
- Embedding generation (text → vector)
- Batch cosine similarity
- Matrix operations for retrieval
- Voxel/spatial indexing (if needed)

**Characteristics:**
- High parallelism
- Efficient for batch operations
- Shared memory with CPU

## Data Flow Examples

### Example 1: Simple Question
```
User: "What's 2+2?"

A78-0 (EXEC): Classify → REACTIVE
A78-0 (EXEC): Dispatch → A78-1
A78-1 (RESP): Generate "4" → User
              [23ms total]
```

### Example 2: Memory Query
```
User: "What did we discuss about neural networks?"

A78-0 (EXEC): Classify → MEMORY_REQUIRED
A78-0 (EXEC): Dispatch query to GPU + A55 pool

GPU:          Embed query → [0.1, -0.3, 0.8, ...]
A55 2-3:      Vector search → top 5 relevant chunks
A55 0-1:      Graph traverse → related entities
A78-0 (EXEC): Aggregate context

A78-0 (EXEC): Dispatch to A78-1 with context
A78-1 (RESP): Generate "Last time we discussed backprop..."
              [~200ms total, feels instant]
```

### Example 3: Long-Form Generation
```
User: "Explain how transformers work in detail"

A78-0 (EXEC): Classify → GENERATIVE_LONG
A78-0 (EXEC): Check memory for prior context
A55 2-3:      Quick vector lookup (parallel)

A78-1 (RESP): Generate first 10 tokens FAST
              "Transformers are a type of..."
              [25ms to first token - feels snappy]

A78-0 (EXEC): Handoff to A55 pool for continuation
A55 4-5:      Continue generation at 18 tok/s
              (User is reading, doesn't notice slower rate)

A55 0-1:      Meanwhile, update knowledge graph
              (Async, preparing for future queries)
```

### Example 4: Background Consolidation
```
[Screen off, charging]

A78-0 (EXEC): Low-power monitoring

A55 Pool:     - Summarize today's conversations
              - Update entity relationships
              - Compress old memories
              - Pre-compute common embeddings
              - Index new knowledge

GPU:          Batch embedding generation

[No user-facing latency requirements]
[Power efficient, thermal headroom]
```

## Memory Hierarchy

```
┌─────────────────────────────────────────┐
│         IMMEDIATE (Hot Cache)           │  A78 L1/L2
│   - Current conversation turn           │  ~128KB
│   - Active tool results                 │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         WORKING (Session)               │  Shared L3 / RAM
│   - Conversation history                │  ~1-4MB
│   - Retrieved context                   │
│   - Loaded memory chunks                │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         LONG-TERM (Persistent)          │  Flash storage
│   - Vector database                     │  ~100MB+
│   - Knowledge graph                     │
│   - Summarized conversations            │
│   - Entity memory                       │
└─────────────────────────────────────────┘
```

## Power States

| State | Active Components | Power | Latency to Full |
|-------|-------------------|-------|-----------------|
| IDLE | A78-0 (low freq) | ~0.1W | 50ms |
| READY | A78-0 + A78-1 (med) | ~0.5W | 10ms |
| RESPOND | A78-0 + A78-1 (high) | ~2W | 0ms |
| THINK | A78-0 + A55 pool | ~1W | 100ms |
| FULL | All cores + GPU | ~4W | 0ms |
| BACKGROUND | A55 pool only | ~0.5W | 200ms |

## The Moneyball Insight

Traditional: "Run everything on fastest cores"
→ 48 tok/s, ~3W continuous, thermal throttle in 30s

Heterogeneous: "Match processing to requirements"
→ Same perceived speed, ~1W average, runs indefinitely

**The cognitive architecture isn't about going faster - it's about being smarter about when speed matters.**

## Implementation Phases

### Phase 1: Routing
- Query classifier (can be regex/heuristics initially)
- Taskset-based core pinning
- Measure power delta

### Phase 2: Memory System
- SQLite + embeddings on A55 cores
- Background summarization
- Entity extraction

### Phase 3: GPU Integration
- OpenCL/Vulkan compute for embeddings
- Batch similarity operations

**GPU Findings (Feb 2026):**
```
PowerVR BXM-8-256 via OpenCL:
- 1 Compute Unit, 1024 max workgroup
- 390 MHz, 1.8GB accessible memory
- FP16 support (cl_khr_fp16)
- 28KB local memory

Performance (512-dim embeddings):
- GPU: 0.5 GFLOPS (31ms for 16K vectors)
- CPU: 2.3 GFLOPS (7.3ms for 16K vectors)
- Verdict: A78 wins on raw throughput

Key Insight: GPU value is PARALLELISM, not speed
- GPU search overlaps with CPU inference
- 31ms search happens while generating first 1-2 tokens
- "Free" compute from CPU perspective
```

### Phase 4: Full Integration
- State machine for conversation flow
- Dynamic power management
- Thermal-aware scheduling
