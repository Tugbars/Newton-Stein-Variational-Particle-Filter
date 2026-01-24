# =============================================================================
# SVPF Joint State-Parameter Tests
# =============================================================================

# Test executable
add_executable(test_svpf_joint test_svpf_joint.cu)

# Include directories
target_include_directories(test_svpf_joint 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Link against svpf_joint library
target_link_libraries(test_svpf_joint 
    PRIVATE 
        svpf_joint
        CUDA::cudart
        CUDA::curand
)

# CUDA settings
set_target_properties(test_svpf_joint PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compiler flags
target_compile_options(test_svpf_joint PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
    >
)

# Windows-specific
if(MSVC)
    target_compile_options(test_svpf_joint PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=/W3
        >
    )
endif()

# Add to CTest (optional)
enable_testing()
add_test(NAME SVPFJointTest COMMAND test_svpf_joint)
