cmake_minimum_required(VERSION 3.20)
project(trix_lfm2_chip C)

# ═══════════════════════════════════════════════════════════════════════════
# TriX LFM2 Chip — Build System
#
# Usage (host / macOS):
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# Usage (Android ARM64 cross-compile):
#   cmake -B build-android \
#     -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a \
#     -DANDROID_PLATFORM=android-28 \
#     -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-android -j
#
# NDK default: ~/Library/Android/sdk/ndk/28.2.13676358
# ═══════════════════════════════════════════════════════════════════════════

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ─── Compiler flags ───
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Aggressive optimization for the hot path (Q4_0 matvec)
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

    if(ANDROID)
        # ARM64 with dotprod + FP16 (Cortex-A55/A76 on Dimensity 930)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod+fp16")
        # Enable auto-vectorization reports (optional, for tuning)
        # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Rpass=loop-vectorize -Rpass-missed=loop-vectorize")
    elseif(APPLE)
        # Apple Silicon — already has NEON + dotprod
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=apple-m1")
    endif()
endif()

# Always enable math
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lm")

# ─── Sources ───
set(TRIX_LFM2_SOURCES
    src/lfm2_trix.c
    src/gguf_loader.c
    src/trix_spline.c
)

set(TRIX_LFM2_HEADERS
    include/lfm2_trix.h
    include/trix_spline.h
)

# ─── Static library ───
add_library(trix_lfm2 STATIC ${TRIX_LFM2_SOURCES})
target_include_directories(trix_lfm2 PUBLIC include)
target_link_libraries(trix_lfm2 PRIVATE m)

# ─── Test harness ───
add_executable(test_trix_lfm2 tests/test_trix_lfm2.c)
target_link_libraries(test_trix_lfm2 PRIVATE trix_lfm2 m)

# ─── Standalone runner (load GGUF + generate tokens) ───
add_executable(trix_lfm2_run tests/run_lfm2.c)
target_link_libraries(trix_lfm2_run PRIVATE trix_lfm2 m)

# ─── Spline activation test ───
add_executable(test_spline tests/test_spline.c)
target_link_libraries(test_spline PRIVATE trix_lfm2 m)

# ─── GGUF debug tools ───
add_executable(debug_gguf tests/debug_gguf.c)
target_link_libraries(debug_gguf PRIVATE trix_lfm2 m)

add_executable(debug_layers tests/debug_layers.c)
target_link_libraries(debug_layers PRIVATE trix_lfm2 m)

add_executable(debug_inproj tests/debug_inproj.c)
target_link_libraries(debug_inproj PRIVATE trix_lfm2 m)

add_executable(cross_correlate tests/cross_correlate.c)
target_link_libraries(cross_correlate PRIVATE trix_lfm2 m)

add_executable(compare_weights tests/compare_weights.c)
target_link_libraries(compare_weights PRIVATE trix_lfm2 m)

# ─── Install ───
install(TARGETS trix_lfm2 ARCHIVE DESTINATION lib)
install(FILES ${TRIX_LFM2_HEADERS} DESTINATION include)
install(TARGETS test_trix_lfm2 trix_lfm2_run RUNTIME DESTINATION bin)
