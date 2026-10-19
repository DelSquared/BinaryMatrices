
cmake_minimum_required(VERSION 3.24)

project(BinaryMatricesCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_library(binary_matrices_cuda SHARED
    src/binary_matrix_cuda.cu
)

target_include_directories(binary_matrices_cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(binary_matrices_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/lib"
)

install(
    TARGETS binary_matrices_cuda
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(
    FILES include/binary_matrix_cuda.hpp
    DESTINATION include
)


