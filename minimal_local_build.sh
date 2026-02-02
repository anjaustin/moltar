#!/bin/bash
echo "🔧 MINIMAL LOCAL BUILD APPROACH"

# Try to build with minimal dependencies
mkdir -p minimal_build
cd minimal_build

# Create minimal CMakeLists that avoids problematic headers
cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.19)
project(minimal_integrated_runner)

set(CMAKE_CXX_STANDARD 17)

# Minimal includes
include_directories(
  /Users/aaronjosserand-austin/000/Motorola/research/spaceghost/executorch/cmake-out-android-ni/include
)

# Simple executable without problematic ops
add_executable(minimal_runner
  ../executorch_softchip_ops/ni_attention_op.cpp
)

target_link_libraries(minimal_runner
  /Users/aaronjosserand-austin/000/Motorola/research/spaceghost/executorch/cmake-out-android-ni/libexecutorch.a
)
