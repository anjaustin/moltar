# Reflections: ShortConv in_proj Matvec Bug

## Core Tension: Node 1 vs Node 13

The L2 norms nearly match (permutation signature), but searching specific rows 
didn't find matches (anti-permutation signature). How?

Resolution: We only searched 300 out of 3072 rows. If the mapping is a complex 
permutation (not identity + small offset), we could easily miss. The proper test 
is to compute ALL 3072 outputs and correlate them with llama.cpp's ALL 3072 
outputs. If the L2 match is real, there should be a 1-to-1 correspondence.

BUT — there's a simpler possibility. What if it's NOT a permutation at all? 
What if we're reading the WRONG TENSOR? The L2 being CLOSE but not IDENTICAL 
could be coincidence — many random 3072-vectors of similar-magnitude weights 
dotted with the same input would give similar L2 norms.

## The Delta (Node 10): The Sock That Could Be Underwear

We verified our bytes match the GGUF file at the expected offset. But we're 
reading from the GGUF mmap. llama.cpp ALSO reads from the GGUF, then REPACKS 
into a new buffer. We're both starting from the same raw data.

So either:
(a) We both dequantize identically but the MATMUL works differently, or
(b) Something else entirely is happening

For (a) — we've exhausted all the matmul interpretations (row-major, column-major, 
transposed). None match.

For (b) — what else could change the output?

Wait. Re-read the llama.cpp code:

```cpp
cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);
auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
```

What is `cur` at this point? It's the output of build_norm. Let me look at 
build_norm again:

```cpp
cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
cb(cur, "model.layers.{}.operator_norm", il);
```

The cb fires on cur AFTER the norm. Then:

```cpp
cur = hparams.is_recurrent(il) ? build_shortconv_block(cur, ...) : ...;
```

Inside build_shortconv_block, the FIRST thing is the reshape. So cur still 
has the normed value. OK so input is correct.

## Wait — What About the token_embd_norm?

Looking at GGUF metadata: there's a tensor "token_embd_norm.weight" that we 
use as our output_norm. But there's ALSO an actual embedding normalization 
step before layer 0.

Looking at lfm2.cpp line 9:
```cpp
ggml_tensor * cur = build_inp_embd(model.tok_embd);
cb(cur, "model.embed_tokens", -1);
```

What does build_inp_embd do? Does it normalize the embedding?

## NEW HYPOTHESIS: Embedding Normalization

What if llama.cpp applies token_embd_norm to the embedding BEFORE layer 0, 
and we don't? That would mean llama.cpp's input to layer 0 is DIFFERENT 
from ours — but we already showed it matches (embedding L2 matches, 
operator_norm matches).

Unless... the embedding norm happens INSIDE build_inp_embd and the cb() 
captures the ALREADY-NORMALIZED embedding. Let me check.

No — we verified embedding values match. The cb("model.embed_tokens") output 
matches our trix_embed_token output exactly. So embedding is fine.

## Back to Basics: What If We Search ALL 3072 Rows?

The cleanest experiment: compute all 3072 dot products ourselves, then find 
which of OUR outputs matches llama.cpp's first output (-0.000834). If we find 
it, that tells us exactly which row llama.cpp is mapping to output[0].

If we DON'T find it anywhere, then it's not a permutation — the weight data 
itself is different, which means our weight pointer is wrong.

This is THE experiment. Everything else is noise.

## What I Now Understand

I've been testing individual hypotheses (Q8_0, transpose, column-major) when 
I should have run the definitive experiment: compute ALL 3072 of our outputs 
and match them against ALL 3072 of llama.cpp's outputs. This tells us 
immediately whether it's permutation, wrong weights, or something else.

If it's a permutation: we find which row maps to which output, and derive 
the permutation pattern.

If no match: our weights are wrong, and we need to find the right ones.

The experiment takes seconds to compute. I should have done it first.
