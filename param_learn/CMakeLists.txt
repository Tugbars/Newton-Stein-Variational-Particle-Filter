# =============================================================================
# CPMMH Parameter Learning for SVPF
# =============================================================================
# 
# Correlated Pseudo-Marginal Metropolis-Hastings for learning SV parameters
# θ = (ρ, μ, σ_z) using SVPF as the inner likelihood estimator.
#
# This subdirectory builds:
#   - libcpmmh_svpf    : Static library for CPMMH
#   - cpmmh_test       : Test executable (simulate + recover parameters)
#
# Usage from parent:
#   add_subdirectory(param_learn)
#
# =============================================================================

# Library: CPMMH with SVPF replay
add_library(cpmmh_svpf STATIC
    cpmmh_svpf.cu
)

target_include_directories(cpmmh_svpf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cpmmh_svpf PUBLIC
    CUDA::curand
    CUDA::cublas
)

# Set CUDA properties
set_target_properties(cpmmh_svpf PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Test Executable
# =============================================================================

add_executable(cpmmh_test
    cpmmh_test.cu
)

target_link_libraries(cpmmh_test PRIVATE
    cpmmh_svpf
    CUDA::curand
)

set_target_properties(cpmmh_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# =============================================================================
# Optional: Link against main SVPF library for production kernels
# =============================================================================
# 
# To use the full SVPF kernels instead of simplified replay kernels:
#
# option(CPMMH_USE_SVPF_KERNELS "Use production SVPF kernels in replay" OFF)
# 
# if(CPMMH_USE_SVPF_KERNELS)
#     target_link_libraries(cpmmh_svpf PUBLIC svpf)
#     target_compile_definitions(cpmmh_svpf PUBLIC CPMMH_USE_SVPF_KERNELS)
# endif()

# =============================================================================
# Test target
# =============================================================================

add_custom_target(test_cpmmh
    COMMAND cpmmh_test 500 2000 12345
    DEPENDS cpmmh_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Running CPMMH parameter recovery test"
)

# =============================================================================
# Installation (optional)
# =============================================================================

install(TARGETS cpmmh_svpf
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    cpmmh_svpf.cuh
    svpf_replay_mode.cuh
    DESTINATION include/cpmmh
)
