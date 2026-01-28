# =============================================================================
# SVPF Tools Library
# =============================================================================

set(SVPF_TOOLS_SOURCES
    svpf_finite_diff.cu
    svpf_gradient_diagnostic.cu
    svpf_test_framework.cu
)

# =============================================================================
# Static Library
# =============================================================================
add_library(svpf_tools STATIC ${SVPF_TOOLS_SOURCES})

target_include_directories(svpf_tools
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(svpf_tools
    PUBLIC
        svpf
        CUDA::cudart
        CUDA::curand
)

set_target_properties(svpf_tools PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Standalone Diagnostic (has its own main)
# =============================================================================
add_executable(bandwidth_diagnostic svpf_bandwidth_diagnostic.cu)

target_include_directories(bandwidth_diagnostic
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(bandwidth_diagnostic
    PRIVATE
        CUDA::cudart
        CUDA::curand
)

set_target_properties(bandwidth_diagnostic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)