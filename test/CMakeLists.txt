# =============================================================================
# SVPF Tests
# =============================================================================

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
        svpf_tools 
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
        svpf_tools 
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
        svpf_tools 
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
        svpf_tools 
        CUDA::cudart
        CUDA::curand
)

set_target_properties(svpf_gradient_market PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# -----------------------------------------------------------------------------
# σ (vol-of-vol) gradient test - always informative (unlike ν)
# -----------------------------------------------------------------------------
add_executable(svpf_gradient_sigma test_svpf_gradient_sigma.cu)

target_include_directories(svpf_gradient_sigma
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(svpf_gradient_sigma
    PRIVATE
        svpf
        svpf_tools 
        CUDA::cudart
        CUDA::curand
)

set_target_properties(svpf_gradient_sigma PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# =============================================================================
# CTest Integration
# =============================================================================
enable_testing()

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

add_test(NAME GradientSigma COMMAND svpf_gradient_sigma)
set_tests_properties(GradientSigma PROPERTIES
    TIMEOUT 600
    LABELS "gradient;sigma;self-tuning"
)

# -----------------------------------------------------------------------------
# Performance profiler
# -----------------------------------------------------------------------------
add_executable(profile_svpf profile_svpf.cu)

target_include_directories(profile_svpf
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(profile_svpf
    PRIVATE
        svpf
        svpf_tools 
        CUDA::cudart
        CUDA::curand
)

set_target_properties(profile_svpf PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# -----------------------------------------------------------------------------
# Test framework runner
# -----------------------------------------------------------------------------
add_executable(run_svpf_tests run_svpf_tests.cu)

target_include_directories(run_svpf_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(run_svpf_tests
    PRIVATE
        svpf
        svpf_tools   
        CUDA::cudart
        CUDA::curand
)

set_target_properties(run_svpf_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME TestFramework COMMAND run_svpf_tests --quick)
set_tests_properties(TestFramework PROPERTIES
    TIMEOUT 600
    LABELS "framework;validation"
)

# -----------------------------------------------------------------------------
# Multi-Scale SVPF A/B Test (REACTIVE + INERTIAL vs Mono)
# -----------------------------------------------------------------------------
add_executable(svpf_multiscale_test svpf_multiscale_test.cu)

target_include_directories(svpf_multiscale_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(svpf_multiscale_test
    PRIVATE
        svpf
        svpf_tools 
        CUDA::cudart
        CUDA::curand
)

set_target_properties(svpf_multiscale_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME MultiscaleABTest COMMAND svpf_multiscale_test)
set_tests_properties(MultiscaleABTest PROPERTIES
    TIMEOUT 1200
    LABELS "multiscale;comparison;ab-test"
)

# -----------------------------------------------------------------------------
# Student-t State Dynamics A/B Test (Gaussian vs Student-t AR(1))
# Tests robustness when DGP has fat-tailed innovations
# -----------------------------------------------------------------------------
add_executable(svpf_student_t_test svpf_student_t_test.cu)

target_include_directories(svpf_student_t_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(svpf_student_t_test
    PRIVATE
        svpf
        svpf_tools 
        CUDA::cudart
        CUDA::curand
)

set_target_properties(svpf_student_t_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME StudentTABTest COMMAND svpf_student_t_test)
set_tests_properties(StudentTABTest PROPERTIES
    TIMEOUT 600
    LABELS "student-t;state-dynamics;ab-test"
)