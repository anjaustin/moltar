# The Q4_0 Nibble Bug: A Postmortem

**Date:** February 2, 2026
**Duration:** ~8 hours across two sessions
**Root Cause:** Single nibble-to-element mapping error in Q4_0 dequantization
**Impact:** Every Q4_0 matvec computed wrong dot products, producing garbage logits
**Resolution:** Two-line fix in `lfm2_trix.c` and `gguf_loader.c`

---

## The Symptom

The TriX LFM2 forward pass ran end-to-end without crashing. All sub-chip unit
tests passed (RMSNorm, RoPE, SiLU, matvec shape checks). The GGUF loader
correctly parsed all 148 tensors. Embeddings matched llama.cpp to the last
decimal. RMSNorm output matched to 6 decimal places.

But the logits were wrong. Not slightly wrong — structurally wrong:

```
llama.cpp:  -0.4547  13.7858  13.8545  -0.4003  ...
Our chip:   -0.7942   2.5935  -0.3974  -0.7973  ...
```

The divergence started at the very first matmul: `in_proj` in shortconv block 0.

```
llama.cpp in_proj[0]: -0.000834
Our chip  in_proj[0]: -0.295303
```

---

## The Hunt

### Phase 1: Verify the obvious (Hours 1-2)

**Hypothesis:** We're loading the wrong tensor, or reading from the wrong offset.

Built `dump_gguf_meta.c` to verify GGUF metadata. Confirmed tensor
`blk.0.shortconv.in_proj.weight` at offset 65421248, shape [1024, 3072], type Q4_0.

Built `compare_weights.c` to mmap the raw file and compare bytes at that offset
to our loader's pointer. **Result: byte-identical.** First 36 bytes match exactly.

Built `debug_inproj.c` to manually dequantize row 0 and compute the dot product
from raw file bytes. **Result: -0.295303** — same as our code. The bytes were
right, the dequant was apparently right, the math was apparently right.

**Conclusion:** The weight data is correct. The input is correct. Something else
is wrong.

### Phase 2: Statistical analysis (Hours 3-4)

**Hypothesis:** Maybe the rows are permuted — our output is llama.cpp's output
in a different order.

Built `cross_correlate.c`. Computed all 3072 dot products from our chip and
dumped llama.cpp's 3072 in_proj outputs. Ran a permutation-match analysis:
98.4% of our values were within tolerance 0.01 of some llama.cpp value.

This looked promising — maybe a row permutation? But then we ran the control
experiment: a **random** permutation of our values ALSO matched at 98.4%.
The values just had similar distributions. The correlation was spurious.

**Tighter tolerance analysis killed the hypothesis:**
```
tol=1e-2: 98.4% (random control = same)
tol=1e-5:  6.5%
tol=0:     0.0%
```

Not a permutation. We were computing entirely different dot products.

### Phase 3: Blame the repack (Hours 4-5)

**Hypothesis:** llama.cpp repacks Q4_0 weights into `q4_0_4x8` interleaved format
on Apple Silicon. The repacked format might reorder rows.

Studied `repack.cpp` lines 2138-2164 (repack) and 306-348 (gemv). The repack
interleaves groups of 4 rows, but the output indexing `s[x * 4 + j]` should
preserve sequential order.

Built llama.cpp with `GGML_CPU_REPACK=OFF`. **Result: identical output.**
in_proj[0] still = -0.000834. Repack was not the cause.

Built llama.cpp with `GGML_METAL=OFF` and `GGML_BLAS=OFF` too (pure CPU, no
acceleration). **Result: still identical.** Ruled out all backend differences.

### Phase 4: The definitive test (Hour 6)

At this point I had proven:
- Same bytes in the GGUF file
- Same input vector (operator_norm-0 matches to all decimals)
- Not a repack issue
- Not a Metal/GPU issue
- Not a row permutation

The only remaining possibility: **our dequantization code is computing
different values from the same bytes.** But how? I'd checked the scale
extraction, the nibble masking, the subtract-8... everything looked right.

I wrote `verify_inproj_weights.cpp` — a C++ test that runs inside llama.cpp's
eval callback, accesses the weight tensor via `t->src[0]`, reads the raw Q4_0
bytes from llama's own data pointer, and computes the dot product using our
dequant logic.

**This was the breakthrough.** The test printed:

```
Row    0: dot = -0.000982
```

From llama.cpp's tensor pointer, our dequant code gives -0.000982 (matching
llama.cpp's -0.000834 within Q8_0 noise). But our standalone code gives
-0.295303 from the SAME bytes.

The bytes were identical. The input was identical. The only difference was the
dequant code itself.

### Phase 5: The two-line diff (Hour 6, minute 45)

I put the two code paths side by side:

**verify_inproj_weights.cpp (correct):**
```c
dot += v0 * g_norm_output[b * 32 + j];       // element j
dot += v1 * g_norm_output[b * 32 + j + 16];  // element j + 16
```

**lfm2_trix.c (broken):**
```c
sum += v0 * xb[2 * j];      // element 2*j
sum += v1 * xb[2 * j + 1];  // element 2*j + 1
```

The Q4_0 format stores 32 values per block in 16 bytes. Each byte encodes two
4-bit values. The low nibble (bits 0-3) maps to elements 0-15. The high nibble
(bits 4-7) maps to elements 16-31. This is the **split** layout.

We were using the **interleaved** layout: low nibble -> even elements, high
nibble -> odd elements. This is wrong. It produces a valid-looking (bounded,
reasonable-magnitude) dot product from any input, but it's combining the wrong
weight values with the wrong input elements.

---

## Why the Bug Was Hard to Find

1. **The output looked reasonable.** It wasn't NaN, wasn't huge, wasn't zero.
   The logits were in a plausible range. The model even generated tokens
   (wrong ones, but it didn't crash).

2. **All sub-chip tests passed.** Our matvec test verified the shape and that
   the output was non-zero. But the test used synthetic weights where the
   interleaved vs split distinction doesn't matter (if the test weights have
   a specific pattern, both orderings might give similar results).

3. **The bytes matched.** Every diagnostic confirmed the raw data was correct.
   The pointer was right. The offset was right. The file contents were right.
   This led us away from the dequant code and toward increasingly exotic
   hypotheses (repack, Metal, row permutation, column-major storage).

4. **The statistical coincidence.** The 98.4% "permutation match" was a red
   herring that consumed an hour. It turned out to be a property of the value
   distribution, not evidence of a permutation.

5. **Confirmation bias.** The dequant code "looked correct" — the nibble
   masking, the scale multiplication, the subtract-8 all matched the Q4_0
   spec. The element indexing was the one thing that looked correct but wasn't,
   and it's easy to confuse the two valid-looking mappings.

---

## The Fix

```diff
-  sum += v0 * xb[2 * j];
-  sum += v1 * xb[2 * j + 1];
+  sum += v0 * xb[j];
+  sum += v1 * xb[j + 16];
```

Two lines. Applied in `src/lfm2_trix.c:110-111` and `src/gguf_loader.c:673-674`.

After the fix:
- Top-5 predicted tokens: identical to llama.cpp
- Logit values: within ~0.06 (Q4_0 noise from not using Q8_0 intermediate)
- All 32 tests pass

---

## Lessons

1. **When the data is correct but the output is wrong, the interpretation of
   the data is wrong.** We verified the bytes were right 5 different ways
   before questioning how we *read* those bytes.

2. **The definitive test crosses trust boundaries.** The bug was found by
   running our dequant code against llama.cpp's data pointer inside llama's
   own process. This eliminated all variables except the dequant logic itself.

3. **Statistical analysis can mislead.** The 98.4% match looked like evidence
   but was noise. Always run a random control.

4. **Unit tests that don't test against a reference implementation are
   necessary but not sufficient.** Our matvec test verified the computation
   ran and produced a non-trivial output. It didn't verify the output matched
   a known-good reference. The GGUF comparison test was the one that caught it.

5. **The LMM worked.** The Lincoln Manifold Method's structure — raw
   observation, node decomposition, reflection — kept the debug process
   systematic when it would have been easy to thrash randomly.

---

## Tools Built During the Hunt

| Tool | Purpose | Status |
|------|---------|--------|
| `dump_gguf_meta.c` | Dump all GGUF KV metadata + tensor shapes | Complete |
| `compare_weights.c` | Byte-level comparison of GGUF data | Complete |
| `cross_correlate.c` | Permutation analysis with random control | Complete |
| `debug_inproj.c` | Focused in_proj matvec with Q8_0 simulation | Complete |
| `debug_layers.c` | Per-layer activation dump | Complete |
| `debug_gguf.c` | Conv kernel dump + forward pass | Complete |
| `llama_layer_dump.cpp` | llama.cpp eval callback activation dump | Complete |
| `llama_logit_dump.cpp` | llama.cpp logit extraction | Complete |
| `verify_inproj_weights.cpp` | **The breakthrough** — dequant from llama's tensor pointer | Complete |
| `test_q4_verify.c` | Q4_0 dequant verification | Complete |

These tools remain useful for future debugging and for verifying NEON-optimized
kernels against the scalar reference.
