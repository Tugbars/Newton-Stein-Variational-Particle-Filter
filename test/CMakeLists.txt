# =============================================================================
# SVPF Tests
# =============================================================================

# Main test executable (unit tests + benchmarks)
#add_executable(svpf_test svpf_test.cu)

#target_include_directories(svpf_test
#    PRIVATE
#        ${CMAKE_SOURCE_DIR}/include
#        ${CUDAToolkit_INCLUDE_DIRS}
#)

#target_link_libraries(svpf_test
#    PRIVATE
#        svpf
#        CUDA::cudart
#        CUDA::curand
#)

#set_target_properties(svpf_test PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON
#)

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

# -----------------------------------------------------------------------------
# HCRBPF DGP comparison test (fair comparison - HCRBPF-favorable)
# -----------------------------------------------------------------------------
add_executable(svpf_hcrbpf_dgp test_svpf_hcrbpf_dgp.cu)

target_include_directories(svpf_hcrbpf_dgp
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(svpf_hcrbpf_dgp
    PRIVATE
        svpf
        CUDA::cudart
        CUDA::curand
)

set_target_properties(svpf_hcrbpf_dgp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# -----------------------------------------------------------------------------
# Gradient diagnostic test (self-tuning SVPF - Step 0 verification)
# -----------------------------------------------------------------------------
add_executable(svpf_gradient_dgp test_svpf_gradient_dgp.cu)

target_include_directories(svpf_gradient_dgp
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(svpf_gradient_dgp
    PRIVATE
        svpf
        CUDA::cudart
        CUDA::curand
)

set_target_properties(svpf_gradient_dgp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# -----------------------------------------------------------------------------
# Market data gradient test (real crashes: 2008, 2020, etc.)
# -----------------------------------------------------------------------------
add_executable(svpf_gradient_market test_svpf_gradient_market.cu)

target_include_directories(svpf_gradient_market
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(svpf_gradient_market
    PRIVATE
        svpf
        CUDA::cudart
        CUDA::curand
)

set_target_properties(svpf_gradient_market PROPERTIES
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

add_test(NAME HCRBPFComparison COMMAND svpf_hcrbpf_dgp)
set_tests_properties(HCRBPFComparison PROPERTIES
    TIMEOUT 300
    LABELS "comparison"
)

add_test(NAME GradientDiagnostic COMMAND svpf_gradient_dgp)
set_tests_properties(GradientDiagnostic PROPERTIES
    TIMEOUT 300
    LABELS "gradient;self-tuning"
)

add_test(NAME GradientMarket COMMAND svpf_gradient_market)
set_tests_properties(GradientMarket PROPERTIES
    TIMEOUT 600
    LABELS "gradient;market;self-tuning"
)

# =============================================================================
# Custom Targets
# =============================================================================

# Run unit tests with verbose output
#add_custom_target(run_tests
#   COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_test
#    DEPENDS svpf_test
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#    COMMENT "Running SVPF unit tests..."
#)

# Benchmark target
#add_custom_target(benchmark
#    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_test --benchmark
#   DEPENDS svpf_test
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#    COMMENT "Running SVPF benchmarks..."
#)

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

# Run HCRBPF DGP comparison (fair comparison with HCRBPF)
add_custom_target(hcrbpf_compare
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_hcrbpf_dgp
    DEPENDS svpf_hcrbpf_dgp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running SVPF on HCRBPF DGP (fair comparison)..."
)

# Run HCRBPF comparison with custom parameters
add_custom_target(hcrbpf_compare_custom
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_hcrbpf_dgp
            --particles 1024 --stein 10 --ticks 5000
    DEPENDS svpf_hcrbpf_dgp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running SVPF on HCRBPF DGP with custom params..."
)

# -----------------------------------------------------------------------------
# Gradient Diagnostic Targets (Self-Tuning SVPF Development)
# -----------------------------------------------------------------------------

# Run gradient diagnostic test (Step 0: verify gradients point correct direction)
add_custom_target(gradient_test
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_gradient_dgp
    DEPENDS svpf_gradient_dgp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running gradient diagnostic test (Step 0: verify before learning)..."
)

# Run gradient test with CSV log output (for plotting)
add_custom_target(gradient_log
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_gradient_dgp --log svpf_gradient_diagnostic.csv
    DEPENDS svpf_gradient_dgp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running gradient diagnostic with CSV log output..."
)

# Run gradient test on real market data
add_custom_target(gradient_market
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_gradient_market
    DEPENDS svpf_gradient_market
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running gradient diagnostic on real market data (2008, 2020 crashes)..."
)

# Run gradient market test with time series output
add_custom_target(gradient_market_ts
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_gradient_market --timeseries gradient_timeseries.csv
    DEPENDS svpf_gradient_market
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running gradient diagnostic with time series output..."
)

# Run all comparison tests
add_custom_target(compare_all
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_scenarios
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_hcrbpf_dgp
    DEPENDS svpf_scenarios svpf_hcrbpf_dgp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all comparison tests (SVPF-favorable + HCRBPF-favorable DGPs)..."
)

# Run all tests including gradient diagnostic
add_custom_target(all_tests
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_scenarios
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_hcrbpf_dgp
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_gradient_dgp
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/svpf_gradient_market
    DEPENDS svpf_scenarios svpf_hcrbpf_dgp svpf_gradient_dgp svpf_gradient_market
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests (scenarios + HCRBPF + gradient DGP + gradient market)..."
)