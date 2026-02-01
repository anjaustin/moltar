#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace ni {

// Vulkan-backed ShortConv3 step.
//
// Contract (v0):
// - All tensors are float32, contiguous.
// - Shapes:
//   - bx:    [D]
//   - c:     [D]
//   - state: [D, 2]  (flattened is also accepted if contiguous)
//   - w:     [D, 3]
//   - out:   [D]
// - v0 runtime updates next-state internally (not reflected in schema).
executorch::aten::Tensor& shortconv3_step_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& bx,
    const executorch::aten::Tensor& c,
    const executorch::aten::Tensor& state,
    const executorch::aten::Tensor& w,
    executorch::aten::Tensor& out);

} // namespace ni

