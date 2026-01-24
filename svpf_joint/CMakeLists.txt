# =============================================================================
# SVPF Joint State-Parameter Estimation Module
# =============================================================================
# 
# Joint SVPF: Parameters (μ, ρ, σ) are particles, not external constants.
# Eliminates "injection problem" during regime changes.
#
# Structure:
#   include/  - Header (svpf_joint.cuh)
#   src/      - CUDA kernels and host implementation
#   test/     - Test harness
#
# =============================================================================

message(STATUS "Configuring SVPF Joint module...")

# Add source library first
add_subdirectory(src)

# Add tests
add_subdirectory(test)

message(STATUS "SVPF Joint module configured")
