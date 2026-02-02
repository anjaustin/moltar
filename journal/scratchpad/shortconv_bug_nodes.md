# Nodes of Interest: ShortConv in_proj Matvec Bug

## Node 1: L2 Norm Near-Match
Our full in_proj L2 = 15.857, llama.cpp = 15.787. Only 0.4% different.
Why it matters: L2 is invariant to permutation. If values are just reordered, 
L2 is identical. The 0.4% gap fits Q8_0 quantization noise. This strongly 
suggests we compute the SAME set of dot products, just in a different order.

## Node 2: Values Totally Different, Not Just Noisy
Our first 8: [-0.295, -0.081, 0.236, -0.303, 0.347, -0.270, 0.367, 0.090]
Llama first 8: [-0.001, 0.125, 0.268, -0.535, 0.070, -0.038, 0.018, 0.400]
These aren't close. Not even the same sign in many cases. This rules out 
rounding or quantization as the sole cause.

## Node 3: Byte-Level Data Verified Correct
We read the exact same bytes from the GGUF file as verified by independent 
hex dump. Offset 65421248, first 20 bytes match. Our GGUF parser is reading 
the right data.

## Node 4: Input (operator_norm) Verified Identical
Both match to 6 decimal places. The divergence is purely in the matmul.

## Node 5: Our Matvec Is Internally Consistent
Manual dequant + dot product matches our trix_matvec_q4_0 to float epsilon.
So our Q4_0 interpretation (low nibble first, subtract 8, multiply by scale) 
is self-consistent. The question is whether it matches llama.cpp's interpretation.

## Node 6: Q8_0 Quantization Is Not the Cause
We implemented Q4_0 x Q8_0 integer dot product mimicking llama.cpp's approach.
Result: -0.297762 vs our F32 -0.295303. Very close to each other, both very 
far from llama.cpp's -0.000834.

## Node 7: The Reshape Before Matmul
llama.cpp does ggml_reshape_3d(cur, {1024, 1, 1}) before the matmul. For 
single-token, this shouldn't matter — same data, same memory layout.

## Node 8: The q4_0_4x8 Repacking
Confirmed: repacking does NOT change dequantized values. It's a pure layout 
transform for SIMD. The XOR mask converts unsigned to signed but the math 
is identical.

## Node 9: The "build_lora_mm" Call
It's just ggml_mul_mat(W, cur) with no LoRA. Plain matmul.

## Node 10: The Delta — What We Haven't Checked
We checked our dequant. We checked the raw bytes. We checked the offset.
We checked the input. We checked Q8_0 quantization. We checked repacking.
What we HAVEN'T checked: whether llama.cpp's in_proj tensor actually 
points to the SAME byte offset as what we read. The repacked tensor lives 
in a DIFFERENT buffer (CPU_REPACK). The data was COPIED and REPACKED from 
the original mmap into a new buffer. 

## Node 11: The Fundamental Experiment We Haven't Run
We haven't used llama.cpp to directly read back the (repacked) weight data 
and dequantize row 0 to verify it matches our row 0. We've been comparing 
OUTPUTS without verifying the WEIGHTS match after repacking.

## Node 12: What If GGML's ne[0]/ne[1] Convention Is Opposite to Ours?
GGUF stores ne[0]=1024, ne[1]=3072 for in_proj. What if GGML internally 
swaps these during loading? What if the loaded tensor has ne[0]=3072, ne[1]=1024?
Then "row i" in the matmul would be 3072 elements long, and there would be 
1024 output rows — but that contradicts the cb() output which has 3072 elements.

Actually no — the captured tensor has shape=[3072] with n_elem=3072. That 
confirms the output IS 3072 elements. So the matmul IS doing 
{ne0=1024, ne1=3072} @ {1024} = {3072}.

## Node 13: Searching for the Right Row Was Inconclusive
We searched rows 0-99, 1024-1123, 2048-2147 for a row whose dot product 
matches -0.000834. Best match was row 80 with diff=7.49e-3. That's not close 
enough. If it were a simple permutation, we should have found an exact match 
(within Q8_0 noise).

Tension with Node 1: How can the L2 match if no individual value matches?
Resolution possibility: the permutation might scatter our values across all 
3072 outputs, not just within our search range.
