cmake_minimum_required(VERSION 3.19)
project(final_integrated_runner)

set(CMAKE_CXX_STANDARD 17)

# ExecuTorch paths
set(EXECUTORCH_BUILD_DIR "/Users/aaronjosserand-austin/000/Motorola/research/brack/local_final_build/../spaceghost/executorch/cmake-out-android-ni")

# Include directories with compatibility headers first
set(_common_include_directories
  /Users/aaronjosserand-austin/000/Motorola/research/brack/local_final_build/../compat_headers
  ${EXECUTORCH_BUILD_DIR}/include
  ${EXECUTORCH_BUILD_DIR}/schema/include
  ${EXECUTORCH_BUILD_DIR}/extension/include
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/include
)

# Create integrated runner with Neural Interposer ops
add_executable(final_integrated_runner
  ../executorch_softchip_ops/ni_vulkan_shortconv3.cpp
  ../executorch_softchip_ops/ni_attention_op.cpp
  ../executorch_softchip_ops/ni_channel.cpp
  ../executorch_softchip_ops/ni_trix_context.cpp
  ../integrated_runner_main.cpp
)

target_compile_options(final_integrated_runner PUBLIC -Wno-deprecated-declarations -fPIC)
target_compile_definitions(final_integrated_runner PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)
target_include_directories(final_integrated_runner PRIVATE $ {_common_include_directories})

# Link libraries
target_link_libraries(final_integrated_runner
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
