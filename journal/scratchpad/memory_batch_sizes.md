# Memory Batch Sizes: L1 ↔ L2 ↔ L3 ↔ DRAM

## The Hierarchy on Dimensity 930

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY HIERARCHY                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ NEON Registers: 32 × 128-bit = 512 bytes                            │    │
│  │ Access: 1 cycle, 0.45ns @ 2.2GHz                                    │    │
│  │ Batch: 16 bytes per LOAD/STORE (LDR Q / STR Q)                      │    │
│  │        64 bytes per LD4 (4 × 16-byte interleaved)                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↕ 16-64 bytes                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ L1D Cache: 64 KB (A78) / 32 KB (A55)                                │    │
│  │ Line size: 64 bytes                                                  │    │
│  │ Access: 4 cycles, ~1.8ns                                             │    │
│  │ Batch: 64 bytes per cache line fill                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↕ 64 bytes                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ L2 Cache: 256-512 KB per core                                       │    │
│  │ Line size: 64 bytes                                                  │    │
│  │ Access: 10-15 cycles, ~5-7ns                                         │    │
│  │ Batch: 64 bytes per cache line                                       │    │
│  │ Note: Can prefetch multiple lines with PRFM                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↕ 64 bytes                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ L3 / SLC (System Level Cache): ~1 MB shared                         │    │
│  │ Line size: 64 bytes                                                  │    │
│  │ Access: 20-30 cycles, ~10-15ns                                       │    │
│  │ Batch: 64 bytes per cache line                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↕ 64 bytes (but DRAM burst is larger)          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ DRAM: LPDDR4X 4 GB                                                   │    │
│  │ Burst length: BL16 = 16 × 16-bit = 32 bytes per channel             │    │
│  │ Two channels: 32 × 2 = 64 bytes per burst                           │    │
│  │ Row buffer: 8 KB (open row = fast column access)                    │    │
│  │ Access: Row hit ~10ns, Row miss ~20-25ns                            │    │
│  │ Batch: 64 bytes minimum, but can request more                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Numbers

| Transfer | Batch Size | Latency | Bandwidth |
|----------|------------|---------|-----------|
| Register ↔ L1 | 16-64 bytes | 1-4 cycles | ~50 GB/s |
| L1 ↔ L2 | 64 bytes | 10-15 cycles | ~30 GB/s |
| L2 ↔ L3 | 64 bytes | 20-30 cycles | ~20 GB/s |
| L3 ↔ DRAM | 64 bytes | 50-100 cycles | 8-10 GB/s |

## DRAM Burst Details (LPDDR4X)

```
LPDDR4X Burst:
- Burst Length (BL): 16 (fixed for LPDDR4)
- Bus Width: 16 bits per channel × 2 channels = 32 bits
- Burst Size: BL16 × 32 bits = 512 bits = 64 bytes

One DRAM access = 64 bytes minimum
```

But the memory controller can issue **multiple bursts** back-to-back:

```
Prefetch / Streaming:
- Hardware prefetcher can request 4-8 cache lines ahead
- 4 lines × 64 bytes = 256 bytes
- 8 lines × 64 bytes = 512 bytes

Maximum useful batch:
- L1 can hold 64 KB
- Prefetch stream: ~256-512 bytes per "request train"
- Row buffer: 8 KB (entire open row)
```

## The Answer: Maximum Batch per "Round Trip"

**Single request**: 64 bytes (one cache line)

**Prefetch stream**: 256-512 bytes (4-8 lines, hardware prefetch depth)

**Row buffer exploit**: 8 KB (if accessing sequentially within one row)

**Practical maximum for one "operation"**:
- If you can keep accessing within one DRAM row
- And you're streaming sequentially
- You can move ~8 KB with only ONE row activation penalty
- Subsequent accesses within that row are ~5ns each

## For Spline Tables

| Table Size | Fits Where | Access Pattern |
|------------|------------|----------------|
| 16 bytes | NEON register | TBL instruction |
| 64 bytes | 1 cache line | Single load |
| 256 bytes | 4 cache lines | Prefetch-friendly |
| 4 KB | 64 cache lines | Half a DRAM row |
| 8 KB | 128 cache lines | One DRAM row |
| 64 KB | L1 cache | Full L1 capacity |

## The Sweet Spots

**16 bytes**: TBL - one instruction, 16 lookups
**64 bytes**: One cache line - minimal overhead
**8 KB**: One DRAM row - row buffer magic

If spline table = 8 KB:
- First access opens row (~20ns)
- All 4096 subsequent 2-byte accesses are row hits (~5ns each)
- Total: 20ns + 4096 × 5ns = 20.5 µs for 4096 lookups

Wait, that's slow. Let me recalculate...

Actually, with **burst access**:
- 8 KB / 64 bytes per burst = 128 bursts
- Row open: 20ns
- 128 bursts at ~5ns each = 640ns
- Total: ~660ns for 8 KB

That's ~12 GB/s effective bandwidth, which matches our measurements!

## For Maximum Throughput

To move the most data in one round-trip:

1. **Align to 8 KB** (DRAM row size)
2. **Access sequentially** (maximize row buffer hits)
3. **Use prefetch hints** (PRFM) to keep pipeline full
4. **Exploit both channels** (interleaved addressing)

**Theoretical max per "round trip"**:
- One row activation + full row read = 8 KB in ~660ns
- That's 12 GB/s, close to peak

**Practical max for random access**:
- Each random access = new row = 64 bytes in 20ns
- That's 3.2 GB/s, much lower
