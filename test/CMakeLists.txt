# =============================================================================
# SVPF Tests
# =============================================================================

# Main test executable (unit tests + benchmarks)
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

# -----------------------------------------------------------------------------
# Scenario test (continuous SV DGP - SVPF-favorable)
# -----------------------------------------------------------------------------
add_executable(svpf_scenarios test_svpf_scenarios.cu)

target_include_directories(svpf_scenarios
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(svpf_scenarios
    PRIVATE
        svpf
        CUDA::cudart
        CUDA::curand
)

set_target_properties(svpf_scenarios PROPERTIES
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

add_test(NAME ScenarioTest COMMAND svpf_scenarios)
set_tests_properties(ScenarioTest PROPERTIES
    TIMEOUT 300
    LABELS "scenario"
)

# =============================================================================
# Custom Targets
# =============================================================================

# Run unit tests with verbose output
add_custom_target(run_tests
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_test
    DEPENDS svpf_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running SVPF unit tests..."
)

# Benchmark target
add_custom_target(benchmark
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_test --benchmark
    DEPENDS svpf_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running SVPF benchmarks..."
)

# Run scenario test (outputs CSV for comparison)
add_custom_target(scenarios
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_scenarios
    DEPENDS svpf_scenarios
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running SVPF scenario test (outputs svpf_results.csv)..."
)

# Run scenario test with custom parameters
add_custom_target(scenarios_custom
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_scenarios 
            --particles 1024 --stein 10 --nu 5.0
    DEPENDS svpf_scenarios
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running SVPF scenarios with custom params..."
)