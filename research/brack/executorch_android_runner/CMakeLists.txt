cmake_minimum_required(VERSION 3.19)
project(executorch_android_runner)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Path to the ExecuTorch source tree (used for the runner source file).
if(NOT EXECUTORCH_ROOT)
  message(FATAL_ERROR "EXECUTORCH_ROOT must be set to the ExecuTorch source directory")
endif()

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Find installed ExecuTorch package (from the Vulkan-enabled Android install).
find_package(executorch CONFIG REQUIRED)
target_include_directories(executorch INTERFACE ${_common_include_directories})

# gflags is built as part of the ExecuTorch build tree; point gflags_DIR at it.
find_package(gflags REQUIRED)

if(TARGET gflags::gflags)
  set(_gflags_target gflags::gflags)
elseif(TARGET gflags)
  set(_gflags_target gflags)
else()
  message(FATAL_ERROR "gflags target not found after find_package(gflags)")
endif()

add_executable(
  executorch_runner
  ${EXECUTORCH_ROOT}/examples/portable/executor_runner/executor_runner.cpp
)

target_compile_options(executorch_runner PUBLIC -Wno-deprecated-declarations -fPIC)

target_link_libraries(
  executorch_runner
  executorch
  executorch_core
  extension_evalue_util
  extension_runner_util
  extension_data_loader
  extension_flat_tensor
  portable_ops_lib
  portable_kernels
  ${_gflags_target}
)

# If Vulkan backend target is available, link it in.
if(TARGET vulkan_backend)
  target_link_libraries(executorch_runner vulkan_backend)
endif()

