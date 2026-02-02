cmake_minimum_required(VERSION 3.19)
project(neural_interposer_ops)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# ExecuTorch dependencies
set(EXECUTORCH_BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/../spaceghost/executorch/cmake-out-android-ni")

# Include directories
set(_common_include_directories
  ${EXECUTORCH_BUILD_DIR}/include
  ${EXECUTORCH_BUILD_DIR}/schema/include
  ${EXECUTORCH_BUILD_DIR}/extension/include
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/include
  ${EXECUTORCH_BUILD_DIR}/backends/vulkan/include
)

# Create shared library with Neural Interposer ops
add_library(neural_interposer_ops SHARED
  ../executorch_softchip_ops/ni_vulkan_shortconv3.cpp
  ../executorch_softchip_ops/ni_attention_op.cpp
  ../executorch_softchip_ops/ni_channel.cpp
  ../executorch_softchip_ops/ni_trix_context.cpp
)

target_include_directories(neural_interposer_ops PRIVATE ${_common_include_directories})

# Link dependencies
target_link_libraries(neural_interposer_ops
  ${EXECUTORCH_BUILD_DIR}/libexecutorch.a
  ${EXECUTORCH_BUILD_DIR}/libexecutorch_core.a
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_ops_lib.a
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_kernels.a
)
