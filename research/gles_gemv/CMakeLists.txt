cmake_minimum_required(VERSION 3.10)
project(gles_gemv)

set(CMAKE_CXX_STANDARD 17)

add_executable(gles_gemv gles_gemv.cpp)
target_link_libraries(gles_gemv EGL GLESv3 log)

add_executable(texture_llm texture_llm.cpp)
target_link_libraries(texture_llm EGL GLESv3 log)

add_executable(compute_llm compute_llm.cpp)
target_link_libraries(compute_llm EGL GLESv3 log)

add_executable(cfc_ternary_llm cfc_ternary_llm.cpp)
target_link_libraries(cfc_ternary_llm EGL GLESv3 log)

add_executable(spectral_cfc_llm spectral_cfc_llm.cpp)
target_link_libraries(spectral_cfc_llm EGL GLESv3 log)

# NEON-based CPU engine (no GPU dependencies)
add_executable(neon_ternary_llm neon_ternary_llm.cpp)
target_link_libraries(neon_ternary_llm log)
target_compile_options(neon_ternary_llm PRIVATE -O3 -march=armv8-a+simd)

# Dual-core pipeline engine
add_executable(dual_core_llm dual_core_llm.cpp)
target_link_libraries(dual_core_llm log)
target_compile_options(dual_core_llm PRIVATE -O3 -march=armv8-a+simd)

# Six-core engine (2x A78 + 4x A55)
add_executable(six_core_llm six_core_llm.cpp)
target_link_libraries(six_core_llm log)
target_compile_options(six_core_llm PRIVATE -O3 -march=armv8-a+simd)

# Experimental: Spectral-Rotational-CfC FFN
add_executable(src_ffn src_ffn.cpp)
target_link_libraries(src_ffn log)
target_compile_options(src_ffn PRIVATE -O3 -march=armv8-a+simd)

# Production SRC-FFN Engine
add_executable(src_ffn_engine src_ffn_engine.cpp)
target_link_libraries(src_ffn_engine log)
target_compile_options(src_ffn_engine PRIVATE -O3 -march=armv8-a+simd)
