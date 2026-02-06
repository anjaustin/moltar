# Batched GEMV Integration Guide for KleidiAI

## Summary

This document describes how to integrate the batched-8 GEMV optimization into llama.cpp's KleidiAI backend for ~1.8-2.2x GEMV speedup.

## Validated Results

**Standalone benchmark (2.24x speedup):**
```
Matrix: 1024 x 4608 (144 blocks/row), 1000 iterations

Performance:
  Reference:          3330.7 us (1.00x)
  Batched (precomp):  1486.9 us (2.24x)
  Batched (inline):   2431.2 us (1.37x)
```

**KleidiAI-style data layout (1.82x speedup):**
```
Output: 1024, Input: 4608 (144 blocks), 1000 iterations

Results:
  KleidiAI-style:     2193.6 us (1.00x)
  Batched (precomp):  1208.6 us (1.82x)
```

## How It Works

### The Problem

KleidiAI's Q4_0 GEMV kernel converts FP16 scales to FP32 in the inner loop:
- Each 32-element block has an FP16 scale
- `fcvtl` converts FP16→FP32 (1-2 cycles per block)
- For 4608-dim input, that's 144 conversions per row

More significantly, the per-block scale multiplication creates a serial dependency chain that limits instruction-level parallelism.

### The Solution

1. **Precompute FP32 scales at model load time**
   - Extract FP16 scales from packed weights
   - Convert to FP32 once and store separately
   - Memory overhead: +22% of weight tensor size

2. **Batch multiple blocks before float conversion**
   - Process 8 blocks worth of int32 dot products
   - Convert 8 values to float in two `vcvtq_f32_s32` instructions
   - Hides SCVTF latency through instruction parallelism

3. **Vectorized scale multiply**
   - Load 4 precomputed scales at a time
   - Single `vmlaq_f32` for multiply-accumulate

## Files Created

- `batched_gemv.h` - Header with kernel interface
- `batched_gemv.c` - Optimized kernel implementation
- `test_batched_gemv.c` - Validation and benchmark
- `kleidiai_compare.c` - Comparison with KleidiAI-style layout

## Integration Steps

### 1. Extend Buffer Allocation

In `kleidiai.cpp`, modify `ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size()`:

```cpp
static size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, 
    const struct ggml_tensor * tensor
) {
    // ... existing code ...
    
    const size_t packed = kernels->rhs_info.packed_size_ex(n, k, nr, kr, block_len);
    
    // NEW: Add space for precomputed FP32 scales
    const size_t num_blocks = (k + block_len - 1) / block_len;
    const size_t scale_size = n * num_blocks * sizeof(float);
    
    const size_t total = packed + scale_size;
    const size_t raw = ggml_nbytes(tensor);
    
    return total > raw ? total : raw;
}
```

### 2. Modify Weight Repacking

In `tensor_traits::repack()`, after packing weights, extract scales:

```cpp
int repack(struct ggml_tensor * tensor, const void * data, size_t data_size) {
    // ... existing packing code ...
    
    if (tensor->type == GGML_TYPE_Q4_0) {
        // Pack weights (existing)
        ctx.kernels_q4->rhs_info.pack_func_ex(...);
        
        // NEW: Extract and store FP32 scales
        const size_t num_blocks = k / QK4_0;
        float* scales_ptr = (float*)((uint8_t*)tensor->data + packed_size);
        
        const block_q4_0* src_blocks = (const block_q4_0*)data;
        for (size_t row = 0; row < n; row++) {
            for (size_t b = 0; b < num_blocks; b++) {
                uint16_t fp16_scale = src_blocks[row * num_blocks + b].d;
                scales_ptr[row * num_blocks + b] = GGML_FP16_TO_FP32(fp16_scale);
            }
        }
        
        // Store scales pointer in tensor metadata
        // (Need to extend tensor_traits or use tensor->extra)
    }
}
```

### 3. Add Batched GEMV Kernel

Create a new kernel in `kernels.cpp`:

```cpp
void kai_run_matmul_batched8_gemv(
    size_t m, size_t n, size_t k, size_t bl,
    const void* lhs_packed,     // Packed activations
    const void* rhs_packed,     // Packed weights
    const float* rhs_scales,    // Precomputed FP32 scales
    float* dst,
    size_t dst_stride_row,
    size_t dst_stride_col,
    float scalar_min,
    float scalar_max
) {
    // Implementation follows batched_gemv.c pattern
    // but works with KleidiAI's packed format
}
```

### 4. Modify compute_forward_q4_0

Replace GEMV kernel dispatch:

```cpp
bool compute_forward_q4_0(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    // ... existing setup ...
    
    bool is_gemv = src1->ne[1] == 1;
    
    if (is_gemv && use_batched_gemv) {
        // Get precomputed scales from tensor
        const float* rhs_scales = get_precomputed_scales(src0);
        
        // Compute combined scales (weight * activation)
        // This is cheap: only nb floats for one row
        float* combined = compute_combined_scales(lhs_scales, rhs_scales, nb);
        
        // Call batched kernel
        kai_run_matmul_batched8_gemv(..., rhs_scales, ...);
        
        return true;
    }
    
    // Fallback to original KleidiAI kernel
    kernel->run_kernel_ex(...);
}
```

### 5. Handle Activation Scales

The LHS (activation) scales are computed per-forward-pass. Options:
1. **Store with LHS packing** - Add FP32 scale extraction during LHS quantization
2. **Compute on-the-fly** - For GEMV (m=1), LHS packing is fast anyway

## Memory Layout After Integration

```
Tensor buffer layout:
+------------------+
| Packed weights   |  (KleidiAI qsi4c32p format)
| (n * packed_k)   |
+------------------+
| FP32 scales      |  (n * num_blocks * 4 bytes)
| NEW              |
+------------------+

For LFM2-350M (Q4_0):
- Original packed: ~209 MB
- FP32 scales:     ~46 MB  
- Total:           ~255 MB (+22%)
```

## Expected Performance Impact

For LFM2-350M at Q4_0:
- Current: ~57 tok/s, ~51% time in GEMM (~8.9 ms/token)
- With 1.8x GEMV speedup: ~4.9 ms/token for GEMM
- Projected: ~70-75 tok/s (1.25-1.35x overall improvement)

Note: The 2.24x GEMV speedup doesn't translate to 2.24x end-to-end because:
1. GEMV is only part of the forward pass
2. Other operations (attention, sampling) are not affected
3. Some overhead from combined scale computation

## Testing

Build and run validation:
```bash
cd research/int_matmul/fabric
make test NDK=/path/to/ndk

# On device:
taskset 30 /data/local/tmp/test_batched_gemv_android 4608 1024 1000
```

## Next Steps

1. A human developer should review and adapt this guide
2. Create proper tests that validate numerical accuracy
3. Consider GEMM optimization (batch size > 1) - may need different approach
4. Profile end-to-end with llama-bench to measure actual improvement

## References

- KleidiAI source: `ggml/src/ggml-cpu/kleidiai/`
- Benchmark code: `research/int_matmul/fabric/`
- Original analysis: See conversation history for batched_convert_test.c results
