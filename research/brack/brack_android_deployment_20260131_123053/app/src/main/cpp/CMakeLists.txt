cmake_minimum_required(VERSION 3.18.1)

project("brack_jni")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the native library
add_library(
        brack_jni
        SHARED
        native-lib.cpp
)

# Link standard libraries
find_library(
        log-lib
        log
)

find_library(
        android-lib
        android
)

find_library(
        vulkan-lib
        vulkan
)

# Link libraries
target_link_libraries(
        brack_jni
        ${log-lib}
        ${android-lib}
        ${vulkan-lib}
)
