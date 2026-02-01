cmake_minimum_required(VERSION 3.22)
project(neural_interposer_demo C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(
  interposer_demo
  src/main.cpp
)

find_library(log-lib log)

# Android NDK provides libvulkan + headers via <vulkan/vulkan.h>.
find_library(vulkan-lib vulkan)

target_link_libraries(
  interposer_demo
  ${log-lib}
  ${vulkan-lib}
)

