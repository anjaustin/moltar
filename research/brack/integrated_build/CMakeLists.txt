cmake_minimum_required(VERSION 3.19)
project(integrated_executorch_runner)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# ExecuTorch paths
set(EXECUTORCH_BUILD_DIR "/Users/aaronjosserand-austin/000/Motorola/research/brack/../spaceghost/executorch/cmake-out-android-ni")

# Include directories
set(_common_include_directories
  ${EXECUTORCH_BUILD_DIR}/include
  ${EXECUTORCH_BUILD_DIR}/schema/include
  ${EXECUTORCH_BUILD_DIR}/extension/include
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/include
  ${EXECUTORCH_BUILD_DIR}/backends/vulkan/include
  /Users/aaronjosserand-austin/000/Motorola/research/brack/executorch_softchip_ops
)

add_executable(integrated_runner
  /Users/aaronjosserand-austin/000/Motorola/research/brack/integrated_runner_main.cpp
  /Users/aaronjosserand-austin/000/Motorola/research/brack/executorch_softchip_ops/ni_vulkan_shortconv3.cpp
  /Users/aaronjosserand-austin/000/Motorola/research/brack/executorch_softchip_ops/ni_attention_op.cpp
  /Users/aaronjosserand-austin/000/Motorola/research/brack/executorch_softchip_ops/ni_channel.cpp
  /Users/aaronjosserand-austin/000/Motorola/research/brack/executorch_softchip_ops/ni_trix_context.cpp
)

target_compile_options(integrated_runner PUBLIC -Wno-deprecated-declarations -fPIC)
target_compile_definitions(integrated_runner PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)
target_include_directories(integrated_runner PRIVATE $ {_common_include_directories})

# Link libraries
target_link_libraries(integrated_runner
  ${EXECUTORCH_BUILD_DIR}/libexecutorch.a
  ${EXECUTORCH_BUILD_DIR}/libexecutorch_core.a
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_ops_lib.a
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_kernels.a
  ${EXECUTORCH_BUILD_DIR}/extension/runner_util/libextension_runner_util.a
  ${EXECUTORCH_BUILD_DIR}/extension/evalue_util/libextension_evalue_util.a
  ${EXECUTORCH_BUILD_DIR}/extension/data_loader/libextension_data_loader.a
  ${EXECUTORCH_BUILD_DIR}/extension/flat_tensor/libextension_flat_tensor.a
  ${EXECUTORCH_BUILD_DIR}/backends/vulkan/libvulkan_backend.a
)
