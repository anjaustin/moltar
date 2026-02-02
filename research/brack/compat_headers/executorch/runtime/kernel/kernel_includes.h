#ifndef EXECUTORCH_RUNTIME_KERNEL_KERNEL_INCLUDES_H_
#define EXECUTORCH_RUNTIME_KERNEL_KERNEL_INCLUDES_H_

#include <cstdint>
#include <cstddef>

namespace executorch {
namespace runtime {

class KernelRuntimeContext {
public:
    void* context_data = nullptr;
};

} // namespace runtime
} // namespace executorch

#endif
