/**
 * Neural Interposer Attention Custom Op
 *
 * ExecuTorch custom operation for attention with KV-cache handling
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace ni {

// Attention with explicit KV-cache (LFM2-style)
executorch::aten::Tensor& attention_step_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& x,              // Input [1, 1, hidden_dim]
    const executorch::aten::Tensor& k_cache,        // K-cache [1, n_kv_heads, max_seq_len, head_dim]
    const executorch::aten::Tensor& v_cache,        // V-cache [1, n_kv_heads, max_seq_len, head_dim]
    const executorch::aten::Tensor& input_pos,      // Position [1]
    const executorch::aten::Tensor& wq,             // Q projection weights [hidden_dim, n_heads * head_dim]
    const executorch::aten::Tensor& wk,             // K projection weights [hidden_dim, n_kv_heads * head_dim]
    const executorch::aten::Tensor& wv,             // V projection weights [hidden_dim, n_kv_heads * head_dim]
    const executorch::aten::Tensor& wo,             // Output projection weights [n_heads * head_dim, hidden_dim]
    const executorch::aten::Tensor& freqs_cos,      // RoPE cosines [max_seq_len, head_dim/2]
    const executorch::aten::Tensor& freqs_sin,      // RoPE sines [max_seq_len, head_dim/2]
    executorch::aten::Tensor& out,                  // Output [1, 1, hidden_dim]
    executorch::aten::Tensor& k_cache_out,          // Updated K-cache [1, n_kv_heads, max_seq_len, head_dim]
    executorch::aten::Tensor& v_cache_out           // Updated V-cache [1, n_kv_heads, max_seq_len, head_dim]
);

} // namespace ni