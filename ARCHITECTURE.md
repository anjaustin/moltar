# Architecture

Moltar has two compute engines — LLM inference and vector search — both optimized for the MT6855V's memory hierarchy.

## Target Hardware

```
MediaTek MT6855V (Dimensity 930)
├── CPU Cluster
│   ├── 2x Cortex-A78 @ 2.2 GHz (big)     ← all inference runs here
│   │   ├── L1D: 64 KB, L1I: 64 KB
│   │   ├── L2: 256 KB per core
│   │   └── Cache line: 64 bytes
│   ├── 6x Cortex-A55 (little)              ← unused for inference
│   └── L3 (DSU): ~1 MB shared
├── GPU: PowerVR BXM-8-256
│   ├── OpenCL 3.0, 128-wide SIMD
│   ├── 28 KB local memory
│   └── 950 MHz max
└── DRAM: LPDDR4X @ 4266 MHz
    ├── 6 GB total
    └── 15.5 GB/s sustained (Android stopped)
```

ISA: ARMv8.2-a with DOTPROD (`vdotq_laneq_s32`), FP16, NEON. No I8MM, no SVE.

## LLM Inference Engine

Located in `research/llama.cpp/`. A fork of llama.cpp with custom kernels in `ggml/src/ggml-cpu/`.

### Quantization Formats

**Row-Scaled Q4_0** (`block_q4_0x4_rs`, 64 bytes = 1 cache line):
- 4 interleaved Q4_0 rows per block
- Pure-integer SDOT accumulation (no float until final reduction)
- Power-of-2 shift activation quantization via IEEE754 bit manipulation

**Row-Scaled Q8_0** (`block_q8_0x4_rs`, 128 bytes = 2 cache lines):
- 4 interleaved Q8_0 rows per block
- Same integer accumulation path

### GEMV Kernel (Token Generation)

Token generation is DRAM-bandwidth-bound. Each token reads the full weight matrix once.

```
Inner loop (NEON DOTPROD):
  vdotq_laneq_s32  v_acc, v_weights, v_activations, lane
```

- Processes 4 output elements per iteration (4 rows interleaved in block)
- Activation quantized once, cached across all weight blocks in a row
- Single `addv` horizontal reduction per output element

### GEMM Kernel (Prompt Processing)

Prompt processing has activation reuse across multiple tokens.

- 4 activation rows processed simultaneously
- Same weight block read serves all 4 activations
- 4x reduction in weight memory traffic vs sequential GEMV

### Dispatch Optimizations

- **Barrier-skip**: skip pthread barriers when only 1 thread has work
- **Graph dispatch fast path**: bypass scheduler overhead for simple graphs
- **Thread 1 fast-forward**: main thread starts compute before worker is signaled
- **Activation quant caching**: quantize activations once, reuse across GEMV calls

### Modified Files

```
ggml/src/ggml-cpu/
├── repack.h              # block_q4_0x4_rs, block_q8_0x4_rs structs
├── repack.cpp            # repack functions, tensor_traits, forward_mul_mat dispatch
└── arch/arm/
    └── repack.cpp        # NEON DOTPROD GEMV/GEMM implementations
```

## L-Cache Vector Database

Located in `research/lcvdb/`. An HNSW vector database designed to fit entirely in CPU cache.

### Split Storage Layout

Three separate arrays, each with different access patterns:

```
Topology Array (hot during graph traversal):
  lcvdb_topo_t, 32 bytes per node, 2 nodes per cache line
  ┌──────────────────────┬────┬────┬───────┬──────────┬──────────┐
  │ neighbors (u16 x 8)  │ nc │ ml │ flags │ payload  │ reserved │
  │      16 bytes        │ 1  │ 1  │  2    │    4     │    8     │
  └──────────────────────┴────┴────┴───────┴──────────┴──────────┘

Vector Array (touched only for distance computation):
  lcvdb_vec_t, 64 bytes per node, 1 cache line
  ┌──────────────────────────────┬──────────┐
  │ int8 vector (48 dimensions)  │ padding  │
  │          48 bytes            │ 16 bytes │
  └──────────────────────────────┴──────────┘

DB Header (1 cache line):
  lcvdb_t, 64 bytes
  ┌───────┬────┬────┬───┬───────────┬──────────┬───────────┬──────┬──────────┐
  │ count │ ep │ ml │ M │ topo_ptr  │ vec_ptr  │ max_nodes │ prng │ reserved │
  │  4B   │ 2B │ 1B │1B │    8B     │    8B    │    4B     │  4B  │   32B    │
  └───────┴────┴────┴───┴───────────┴──────────┴───────────┴──────┴──────────┘
```

### Why Split Storage

With the old unified layout (64 bytes/node), traversing the graph loaded full vectors even when only checking neighbor lists. With split storage:

- **Graph traversal** reads only topology: 32 bytes/node instead of 64
- **Distance computation** loads vectors on-demand: only for candidates being scored
- At N=256: topology = 8 KB (fits L1D), vectors = 16 KB (fits L1D), total = 24 KB
- At N=1024: topology = 32 KB (fits L2), search touches only a fraction of vectors

### HNSW Algorithm

**Insert** (build_ref.c):
1. Assign random layer via geometric distribution (xorshift32 PRNG)
2. Greedy descent through upper layers to target layer
3. Brute-force collect top-16 candidates (2*M) from all existing nodes
4. Diversity selection: keep candidate C only if `dot(query, C) > dot(S, C)` for all selected S
5. Connect forward edges (new node -> selected neighbors)
6. Connect reverse edges with replacement when full

**Search** (search_ref.c):
1. Greedy descent through upper layers
2. Beam search on layer 0 with ef_search=64 beam width
3. Two-list approach: candidate pool (unordered, pick best) + result list (sorted)
4. Early termination: stop when best candidate worse than worst result
5. Skip deleted nodes (tombstone flag check)

**Distance** (distance.S):
- NEON int8 dot product for 48 dimensions
- 3 chunks of 16 elements: SMULL -> SADDLP/SADALP (widen to int32 per chunk to avoid int16 overflow)
- ADDV horizontal reduction
- ~12 cycles fully pipelined

### Files

```
research/lcvdb/
├── lcvdb.h           # Structs (lcvdb_t, lcvdb_topo_t, lcvdb_vec_t) + API
├── distance.S        # NEON: lcvdb_dot_i8, lcvdb_dot_i8_preloaded, lcvdb_dot_i8_batch4
├── init_ref.c        # C: lcvdb_init()
├── build_ref.c       # C: lcvdb_insert(), lcvdb_delete()
├── search_ref.c      # C: lcvdb_search() — uses NEON dot products from distance.S
├── test_lcvdb.c      # Correctness: distance, insert, vector integrity, payload, connectivity, search, delete
├── test_recall.c     # Recall@k at N=32..1024
├── test_bench.c      # Latency + throughput at N=32..1024
└── Makefile          # Cross-compile with aarch64-linux-gnu-gcc
```

Old assembly files (`init.S`, `build.S`, `search.S`) exist but are not used — they target the pre-split-storage layout.

## Device Setup

### Boot Script

`/data/adb/service.d/moltar_perf.sh` (installed via Magisk):
- Stops Android framework on boot (`stop`)
- Sets big core governors to `performance`
- Frees ~4 GB/s DRAM bandwidth for inference

### Runtime

```bash
# Set governors (if not done by boot script)
su -c 'echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor'
su -c 'echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor'

# Stop Android framework
su -c 'stop'

# Pin to big cores
taskset c0 /data/local/tmp/test_lcvdb
```

## GPU

PowerVR BXM-8-256, confirmed working via OpenCL probe (`gpu_probe.c`). Int8 dot product kernel written and benchmarked (`gpu_bench.c` + `gpu_dot.cl`). Correct at all N.

GPU dispatch overhead is ~40 us, which makes it slower than CPU for standalone VDB queries. Useful for async search while CPU does LLM inference (dispatch overhead is free when overlapped with 12-50 ms token generation).
