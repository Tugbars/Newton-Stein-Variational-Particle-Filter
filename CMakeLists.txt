cmake_minimum_required(VERSION 3.24)

# =============================================================================
# Compiler setup - Intel ICX on Linux, MSVC on Windows
# =============================================================================
if(NOT WIN32)
    # Linux: Use Intel ICX if available
    find_program(ICX_COMPILER icpx)
    if(ICX_COMPILER)
        set(CMAKE_C_COMPILER "icx")
        set(CMAKE_CXX_COMPILER "icpx")
        set(CMAKE_CUDA_HOST_COMPILER "icpx")
    endif()
endif()

# =============================================================================
# Project
# =============================================================================
project(SteinVariationalParticleFilter 
    VERSION 1.0.0
    LANGUAGES C CXX CUDA
    DESCRIPTION "SVPF for Stochastic Volatility Estimation"
)

# =============================================================================
# C++ and CUDA Standards
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# CUDA Configuration
# =============================================================================
find_package(CUDAToolkit REQUIRED)

# Detect GPU architecture (default to sm_120 for RTX 5080 Blackwell)
# Override with: cmake -DCUDA_ARCH=89 (for RTX 4090)
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "120" CACHE STRING "CUDA architecture (e.g., 89 for Ada, 120 for Blackwell)")
endif()

message(STATUS "CUDA Architecture: sm_${CUDA_ARCH}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# =============================================================================
# Compiler-specific flags
# =============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "Using Intel ICX compiler")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
elseif(MSVC)
    message(STATUS "Using MSVC compiler")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od /RTC1")
    # Enable OpenMP
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    # Disable annoying warnings
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    message(STATUS "Using ${CMAKE_CXX_COMPILER_ID} compiler")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

# =============================================================================
# Build Type
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# Output Directories
# =============================================================================
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =============================================================================
# Include Directories
# =============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# =============================================================================
# Subdirectories
# =============================================================================
add_subdirectory(src)
add_subdirectory(test)

# =============================================================================
# Python Bindings (requires pybind11)
# =============================================================================
# Use modern Python finder first
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development QUIET)

# Auto-detect pybind11 from Python installation
if(Python3_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE PYBIND11_RESULT
    )
    if(PYBIND11_RESULT EQUAL 0 AND PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    endif()
endif()

find_package(pybind11 QUIET)
if(pybind11_FOUND)
    add_subdirectory(py-torch)
    set(PYTHON_BINDINGS_STATUS "ENABLED")
else()
    message(WARNING "pybind11 not found - Python bindings disabled. Install with: pip install pybind11")
    set(PYTHON_BINDINGS_STATUS "DISABLED (pybind11 not found)")
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== SVPF Build Configuration ===")
message(STATUS "  C++ Compiler:    ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA Compiler:   ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Version:    ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Arch:       sm_${CUDA_ARCH}")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python Bindings: ${PYTHON_BINDINGS_STATUS}")
message(STATUS "================================")
message(STATUS "")