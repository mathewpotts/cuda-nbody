cmake_minimum_required(VERSION 3.18)
project(NBodyCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(nbody
    src/main.cu
    src/nbody.cu
    src/shader.cpp
    src/simulation.cu
)

target_include_directories(nbody PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(nbody PRIVATE
    glfw
    glad::glad
    OpenGL::GL
)

target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(nbody PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)