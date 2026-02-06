# DRAM as Compute: Spline Lookup via Row Buffer

## The Idea

DRAM row buffers are essentially large SRAM arrays (~8KB per bank). When you "open" a row, the entire row is loaded into the sense amplifiers.

What if we:
1. Organize spline coefficients as DRAM rows
2. Use the row address as the spline index
3. Use the column address to select the coefficient
4. The DRAM controller does the "lookup" via its normal addressing

**The address becomes the computation.**

## DRAM Architecture Refresher

```
LPDDR4X Bank Structure:
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DRAM BANK                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Row Address (14-16 bits) ──► Row Decoder ──► Opens one row                │
│                                      │                                       │
│                                      ▼                                       │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                     ROW BUFFER (~8KB)                               │    │
│   │   Entire row is loaded into sense amplifiers                        │    │
│   │   [col0][col1][col2][col3]...[col1023]                             │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│   Column Address (10 bits) ──────────┴──► Selects which columns to read     │
│                                                                              │
│   Burst: 16 bytes (8 × 16-bit) per access                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Row Buffer as Lookup Table

If we organize data so that:
- Row N contains spline coefficients for input value N
- Columns contain the actual coefficient values

Then:
```
address = (input_value << 10) | coefficient_offset
data = DRAM[address]
```

The DRAM controller:
1. Opens row = input_value (if not already open)
2. Reads column = coefficient_offset
3. Returns the spline coefficient

**Row buffer hit = 5ns. Row buffer miss = 15-20ns.**

## The Shift-Add Pattern

For a piecewise linear spline: `y = a + b*t` where t is the fractional part.

If we store [a, b] in adjacent columns:
```
Row N: [a_N, b_N, a_N, b_N, ...]  (repeated for burst alignment)

To evaluate spline(x):
  index = floor(x * 256)      // Integer part → row address
  frac = fract(x * 256)       // Fractional part
  
  [a, b] = DRAM[index]        // Row buffer lookup
  y = a + b * frac            // One multiply-add
```

The DRAM does the table lookup. The CPU does one FMA.

## Going Further: DRAM-side Interpolation?

What if we could get DRAM to do the interpolation too?

Modern LPDDR4X doesn't compute. But we can fake it:

**Precompute all fractional values:**
```
For 8-bit input (256 levels) and 4-bit fractional (16 sub-levels):
  Total entries = 256 × 16 = 4096

Store: spline_table[4096] where each entry is the FINAL output value

Lookup:
  full_index = (int)(x * 4096)
  y = DRAM[full_index]  // No math needed, just load!
```

4096 × 2 bytes = 8KB = exactly one DRAM row!

**One activation function = one row buffer load.**

## The Full Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DRAM-AS-SPLINE ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   DRAM Layout:                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ Bank 0: SwiGLU spline (4096 entries × 2 bytes = 8KB = 1 row)         │  │
│   │ Bank 1: Sigmoid spline (4096 entries × 2 bytes = 8KB = 1 row)        │  │
│   │ Bank 2: Tanh spline (4096 entries × 2 bytes = 8KB = 1 row)           │  │
│   │ Bank 3: RMSNorm rsqrt spline (4096 entries × 2 bytes = 8KB = 1 row)  │  │
│   │ Banks 4-7: Available for other functions or per-layer variants       │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   Access Pattern:                                                           │
│   1. Quantize activation to 12-bit (4096 levels)                           │
│   2. Construct address: bank_id | row_0 | column_from_index                │
│   3. Issue DRAM read → hits row buffer (row always open!)                  │
│   4. Result is final spline output, no CPU math needed                     │
│                                                                              │
│   Key Insight:                                                              │
│   If we keep the spline row OPEN, every lookup is a row buffer HIT (~5ns)  │
│   No row activation penalty, just column decode                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Row Buffer Hit Trick

LPDDR4X has 8 banks (or 16 with bank groups). Each bank has its own row buffer.

If we:
1. Pin one row per bank for spline tables
2. Never close those rows
3. All spline accesses hit the row buffer

Then spline lookup = **column access only** = ~5ns per lookup.

But wait - can we actually keep rows open?

## DRAM Row Policy

Memory controllers have row policies:
- **Open page**: Keep row open after access, hope for row buffer hits
- **Close page**: Close row after access, optimize for random access

Mobile SoCs typically use adaptive policies. We can't directly control row buffer state.

BUT: If we access the spline table frequently enough, it stays open naturally.

## The Bandwidth Math

```
Spline table: 8KB per function
Activation vector: 4096 × fp16 = 8KB

To apply spline to all activations:
  - 4096 lookups
  - Each lookup: 2 bytes read
  - Total: 8KB read

At row buffer speed (5ns per 16-byte burst):
  - 8KB / 16 bytes = 512 bursts
  - 512 × 5ns = 2.56 µs

Compare to CPU TBL:
  - 4096 / 16 = 256 TBL instructions
  - 256 × 0.5ns (2 GHz) = 128 ns

DRAM is 20x slower than CPU TBL!
```

Hmm, the raw bandwidth doesn't win...

## The Twist: Streaming Access

What if we interleave spline lookups with GEMM?

```
Traditional:
  GEMM → all outputs ready → apply activation to all → next layer

Interleaved:
  GEMM partial output[0:15] ready →
    Issue spline lookup for [0:15] (prefetch) →
  GEMM partial output[16:31] ready →
    Receive spline result for [0:15], issue lookup for [16:31] →
  ...

The spline lookup happens IN PARALLEL with GEMM compute!
```

This is **latency hiding** - DRAM spline lookups happen during GEMM compute cycles.

## Even Wilder: Address as Computation

What if the address calculation IS the computation?

```
Suppose we want: y = a*x + b (affine transform)

If we lay out memory as:
  DRAM[a][x] = a*x  (precomputed for all a, x combinations)

Then:
  address = (a << 16) | x
  result = DRAM[address]  // The "multiplication" is the address decode!
```

For 8-bit a and 8-bit x:
  - Table size = 256 × 256 × 2 bytes = 128KB
  - Fits in DRAM, cacheable in L2

This is a **multiplication lookup table** addressed by the operands.

## The Most Practical Version

Forget exotic DRAM tricks. The simple version:

```c
// Precompute at init
alignas(4096) fp16 swiglu_lut[4096];  // 8KB, page-aligned
for (int i = 0; i < 4096; i++) {
    float x = (i - 2048) / 256.0f;  // Map to [-8, 8]
    swiglu_lut[i] = (fp16)(x / (1.0f + expf(-x)));
}

// At runtime
void apply_swiglu_dram(fp16* out, fp16* in, int n) {
    for (int i = 0; i < n; i++) {
        int idx = (int)((in[i] + 8.0f) * 256.0f);  // Quantize to 12-bit
        idx = clamp(idx, 0, 4095);
        out[i] = swiglu_lut[idx];  // Memory load = spline evaluation
    }
}
```

This will:
1. Hit L2 cache (8KB table fits)
2. Or hit DRAM row buffer (if L2 misses)
3. Be slower than TBL (memory vs register)

But could be pipelined with GEMM...

## The Real Insight

**DRAM row buffers are too slow for spline lookup compared to CPU TBL.**

- DRAM row buffer: ~5ns per access
- CPU TBL: ~0.5ns per lookup (2 GHz, assuming 1-cycle)

BUT:

**DRAM has massive parallelism potential:**
- 8 banks × independent row buffers = 8 parallel lookups
- With bank interleaving, could sustain 8 lookups per 5ns = 1.6 billion lookups/sec

**CPU TBL is sequential:**
- One TBL per cycle = 2 billion lookups/sec at 2 GHz
- But only 16 elements per TBL

So:
- DRAM: 1.6B scalar lookups/sec
- CPU: 2B scalar lookups, but 32B vector lookups/sec (16 elements × 2 GHz)

**CPU TBL wins for vectorizable workloads.**
**DRAM might win for random/scattered lookups.**

## Where DRAM-as-LUT Could Win

1. **Very large tables** that don't fit in L1/L2
2. **Random access patterns** where cache is useless
3. **Latency-hidden** scenarios where lookup overlaps with compute
4. **Multi-bank parallel** lookups

For activation functions: Tables are small (8KB), access is sequential, CPU wins.

For KV cache attention: Large (MBs), random access... DRAM might help?

## Conclusion

The DRAM-as-spline idea is clever but probably doesn't beat CPU TBL for activation functions.

However, the **address-as-computation** concept is powerful:
- The memory system already does address decode (muxing)
- That decode IS a form of computation
- We can encode lookups/functions into memory layout

The practical win might be:
1. Keep CPU TBL for activations (fast, small)
2. Use DRAM-organized LUTs for larger tables (embeddings, KV cache patterns?)
3. Exploit bank parallelism for scattered accesses
