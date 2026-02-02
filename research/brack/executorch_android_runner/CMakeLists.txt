cmake_minimum_required(VERSION 3.19)
project(executorch_android_runner)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Direct paths - no find_package needed
set(EXECUTORCH_BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/../../spaceghost/executorch/cmake-out-android-ni")

# Include directories
set(_common_include_directories
  ${EXECUTORCH_BUILD_DIR}/include
  ${EXECUTORCH_BUILD_DIR}/schema/include
  ${EXECUTORCH_BUILD_DIR}/extension/include
)

add_executable(
  executorch_runner_integrated
  ${CMAKE_CURRENT_LIST_DIR}/executorch_runner_main.cpp
)

target_compile_options(executorch_runner_integrated PUBLIC -Wno-deprecated-declarations -fPIC)
target_compile_definitions(executorch_runner_integrated PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)
target_include_directories(executorch_runner_integrated PRIVATE ${_common_include_directories})

#
# Soft-chip custom ops (Neural Interposer)
#
add_library(
  ni_softchip_ops
  ${CMAKE_CURRENT_LIST_DIR}/../executorch_softchip_ops/ni_vulkan_shortconv3.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../executorch_softchip_ops/ni_attention_op.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../executorch_softchip_ops/ni_channel.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../executorch_softchip_ops/ni_trix_context.cpp
)
target_include_directories(ni_softchip_ops PRIVATE ${_common_include_directories})
target_compile_options(ni_softchip_ops PUBLIC -Wno-deprecated-declarations -fPIC)
target_compile_definitions(ni_softchip_ops PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)

# Android system libs
find_library(_android_log log)
find_library(_android_vulkan vulkan)
if(_android_log)
  target_link_libraries(ni_softchip_ops PUBLIC ${_android_log})
endif()
if(_android_vulkan)
  target_link_libraries(ni_softchip_ops PUBLIC ${_android_vulkan})
endif()

# Direct library linking
target_link_libraries(
  executorch_runner_integrated
  ${EXECUTORCH_BUILD_DIR}/libexecutorch.a
  ${EXECUTORCH_BUILD_DIR}/libexecutorch_core.a
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_ops_lib.a
  ${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_kernels.a
  ${EXECUTORCH_BUILD_DIR}/extension/runner_util/libextension_runner_util.a
  ${EXECUTORCH_BUILD_DIR}/extension/evalue_util/libextension_evalue_util.a
  ${EXECUTORCH_BUILD_DIR}/extension/data_loader/libextension_data_loader.a
  ${EXECUTORCH_BUILD_DIR}/extension/flat_tensor/libextension_flat_tensor.a
  ${EXECUTORCH_BUILD_DIR}/backends/vulkan/libvulkan_backend.a
  ni_softchip_ops
)
