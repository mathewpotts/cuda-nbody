cmake_minimum_required(VERSION 3.18)

project(NBodyCUDA LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(nbody
    src/main.cu
)

# Enable optimization for Release builds
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Useful CUDA option for debugging kernel launches
set_target_properties(nbody PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)