cmake_minimum_required(VERSION 3.18)
project(svpf_python LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Source files from parent src/ directory
set(SVPF_CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/../src/svpf_kernels.cu
    ${CMAKE_SOURCE_DIR}/../src/svpf_optimized_graph.cu
    ${CMAKE_SOURCE_DIR}/../src/svpf_opt_kernels.cu
)

# Create the Python module
pybind11_add_module(svpf 
    svpf_python.cpp
    ${SVPF_CUDA_SOURCES}
)

# Include directories
target_include_directories(svpf PRIVATE
    ${CMAKE_SOURCE_DIR}/../include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# CUDA flags
target_compile_options(svpf PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        -lineinfo
    >
)

# Link CUDA libraries
target_link_libraries(svpf PRIVATE
    CUDA::cudart
    CUDA::curand
)

# Set CUDA architecture (adjust for your GPU - RTX 5090 = sm_120)
set_target_properties(svpf PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;89;90;120"
    POSITION_INDEPENDENT_CODE ON
)

# Output to py-torch folder for easy import
set_target_properties(svpf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
