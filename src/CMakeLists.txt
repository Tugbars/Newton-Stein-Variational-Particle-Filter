# =============================================================================
# SVPF Library
# =============================================================================

set(SVPF_SOURCES
    svpf_finite_diff.cu
    svpf_optimized_graph.cu
    svpf_opt_kernels.cu
    svpf_gradient_diagnostic.cu
)

# =============================================================================
# Static Library
# =============================================================================
add_library(svpf STATIC ${SVPF_SOURCES})

target_include_directories(svpf
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(svpf
    PUBLIC
        CUDA::cudart
        CUDA::curand
)

# CUDA properties
set_target_properties(svpf PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Installation
# =============================================================================
install(TARGETS svpf
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${CMAKE_SOURCE_DIR}/include/svpf.cuh
    DESTINATION include
)