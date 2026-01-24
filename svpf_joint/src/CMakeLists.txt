# =============================================================================
# SVPF Joint State-Parameter Library
# =============================================================================

set(SVPF_JOINT_SOURCES
    svpf_joint_kernels.cu
    svpf_joint.cu
)

# Create library
add_library(svpf_joint STATIC ${SVPF_JOINT_SOURCES})

# Include directories
target_include_directories(svpf_joint 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CUDAToolkit_INCLUDE_DIRS}
)

# CUDA settings
set_target_properties(svpf_joint PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Link CUDA libraries
target_link_libraries(svpf_joint 
    PUBLIC 
        CUDA::cudart
        CUDA::curand
)

# Compiler flags for CUDA
target_compile_options(svpf_joint PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler=-Wall
    >
)

# Windows-specific: suppress MSVC warnings in CUDA
if(MSVC)
    target_compile_options(svpf_joint PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=/W3
            -Xcompiler=/wd4244
            -Xcompiler=/wd4267
        >
    )
endif()
