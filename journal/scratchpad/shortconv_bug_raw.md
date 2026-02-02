# Raw Thoughts: ShortConv in_proj Matvec Bug

## Stream of Consciousness

We have a bug where our Q4_0 matvec produces completely different output from llama.cpp's
for the in_proj weight in layer 0's shortconv block. Same input (operator_norm matches to 
6 decimal places), same weight data (verified byte-for-byte from the GGUF file at the correct 
offset), but totally different dot product results.

Our output[0] = -0.295303. llama.cpp output[0] = -0.000834. These aren't close. They're 
not even in the same ballpark. The full L2 norms are similar (15.86 vs 15.79) which means 
the right values ARE being computed — just mapped to wrong positions.

Wait. L2 norms being similar is a HUGE clue. L2 norm is rotationally invariant. If we 
computed the exact same set of 3072 dot products but assigned them to different output 
indices, the L2 would be identical. Ours is 15.86 vs 15.79 — close but not identical, 
which fits with Q8_0 quantization noise in llama.cpp's path.

So the hypothesis is: we compute the right dot products but assign them to the wrong 
output indices. This is a PERMUTATION bug, not a computation bug.

What could cause a permutation? Row ordering in the weight matrix. If GGML stores the 
weight rows in a different order than we think, our row 0 would dot with the input to 
produce a valid dot product — just not the one that should go to output[0].

But I verified: the weight tensor shape is {ne[0]=1024, ne[1]=3072}. GGML stores data 
contiguous along ne[0]. Row i starts at offset i * (1024/32) * 18 bytes. This should 
be straightforward.

Unless... wait. What about the `ggml_reshape_3d` that happens BEFORE the matmul?

```cpp
cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);
auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
```

For single token: cur goes from {1024} to {1024, 1, 1}. The matmul W has shape 
{1024, 3072}. mul_mat(W, cur) where W is {1024, 3072} and cur is {1024, 1, 1}...

In GGML, mul_mat(A, B) requires A->ne[0] == B->ne[0]. Output ne[0] = A->ne[1].
So output = {3072, 1, 1}. This is correct.

What about the weight shape though? In GGUF, the tensor is stored as 
ne[0]=1024, ne[1]=3072. But does llama.cpp ever transpose or reshape the weight?

Actually — the KEY question I haven't asked: does the q4_0_4x8 REPACKING change the 
ROW ORDER of the output? The repack function interleaves 4 rows at a time:

```cpp
for (int b = 0; b < nrow; b += 4) {
    for (int x = 0; x < nblocks; x++) {
        for (int i = 0; i < 4; i++) {
            dst_tmp[i] = src[x + i * nblocks];
        }
        *dst++ = make_block_q4_0x4(dst_tmp, 8);
    }
    src += 4 * nblocks;
}
```

This gathers blocks from 4 consecutive rows and interleaves them. The gemv function 
then outputs 4 values per iteration: output[x*4+j] for j=0..3. So the output order 
should still be sequential: 0,1,2,3,4,5,...

But what if nrow != what we think? `nrow = ggml_nrows(t)`. For a tensor with 
ne[0]=1024, ne[1]=3072, nrows = ne[1]*ne[2]*ne[3] = 3072. And nblocks = ne[0]/32 = 32.

So the repack iterates: rows 0-3, then 4-7, etc. Block x from row i within the group.
The gemv outputs in the same order. So output row 0 IS still the dot product of 
original weight row 0 with the input.

Hmm. But wait — what if there's an issue with how the repacked tensor's mul_mat 
maps nrows? Let me think about this differently.

What if the problem is not in the MATVEC but in the INPUT?

No, we verified the input matches to 6 decimal places.

What if we're looking at the wrong weight tensor? What if blk.0.shortconv.in_proj.weight 
in the GGUF is NOT what llama.cpp loads for the shortconv.in_proj of layer 0?

Actually... there IS a potential issue. Our GGUF says the in_proj tensor name is 
"blk.0.shortconv.in_proj.weight". Let me check what name llama.cpp uses to look up 
this tensor.

What if llama.cpp looks for a DIFFERENT tensor name and loads a DIFFERENT weight?

## Questions Arising

1. What if the in_proj weight in llama.cpp is stored with dimensions SWAPPED relative 
   to what we read? i.e., llama.cpp might treat it as {3072, 1024} internally after 
   loading, even though GGUF says {1024, 3072}?

2. What if the repacking process changes the effective row-to-output mapping due to 
   the 3D tensor shape after reshape_3d?

3. L2 norms: 15.86 vs 15.79. These are CLOSE. Is the difference entirely from Q8_0 
   quantization? If so, we ARE computing the same SET of values, just permuted.

4. What would it look like if our row layout is TRANSPOSED? i.e., our "row 0" 
   (first 1024 elements) is actually "column 0" (first element from each of 1024 rows)?

5. Is there a token embedding norm step we're missing? No — we verified embedding 
   AND operator_norm match exactly.

6. The reshape_3d before matmul: {1024} -> {1024, 1, 1}. The matmul with {1024, 3072} 
   weight should produce {3072, 1, 1}. But what if the reshape changes the stride 
   and the matmul interprets the weight differently?

## First Instincts

My gut says this is a row ordering / transpose issue in how we interpret the Q4_0 
blocks. The L2 norm similarity is too strong to be coincidence. We're computing the 
right values in the wrong order.

The most suspicious thing: we tried both row-major and column-major interpretations 
and NEITHER matched. But we only tried those two extremes. What about a block-level 
permutation?

Actually, wait. What if the Q4_0 blocks in the GGUF file are stored in COLUMN-MAJOR 
order within each block row? i.e., the 32 blocks per row are NOT stored as 
[row0_blk0, row0_blk1, ..., row0_blk31, row1_blk0, ...] but rather as
[row0_blk0, row1_blk0, row2_blk0, ..., row31_blk0, row0_blk1, ...]?

No, that doesn't make sense for Q4_0 which is always contiguous per row.

Let me think about what GGML actually does differently. The standard (non-repacked) 
Q4_0 mul_mat path uses `vec_dot_q4_0_q8_0`. Let me look at how that function indexes 
the Q4_0 blocks. If it uses nb01 (the row stride) rather than assuming contiguous 
blocks, maybe there's a stride we're not accounting for.
