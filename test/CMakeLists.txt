# =============================================================================
# SVPF Tests
# =============================================================================

# Main test executable
add_executable(svpf_test svpf_test.cu)

target_include_directories(svpf_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(svpf_test
    PRIVATE
        svpf
        CUDA::cudart
        CUDA::curand
)

set_target_properties(svpf_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# =============================================================================
# CTest Integration
# =============================================================================
enable_testing()

add_test(NAME BasicFunctionality COMMAND svpf_test)
set_tests_properties(BasicFunctionality PROPERTIES
    TIMEOUT 120
    LABELS "unit"
)

# =============================================================================
# Custom Targets
# =============================================================================

# Run tests with verbose output
add_custom_target(run_tests
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_test
    DEPENDS svpf_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running SVPF tests..."
)

# Benchmark target
add_custom_target(benchmark
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_test --benchmark
    DEPENDS svpf_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running SVPF benchmarks..."
)
