# =============================================================================
# SVPF Python Bindings
# =============================================================================

# Create Python module
pybind11_add_module(pysvpf 
    svpf_python.cpp
)

# Include directories
target_include_directories(pysvpf PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link against the svpf static library (built in src/)
target_link_libraries(pysvpf PRIVATE
    svpf
    CUDA::cudart
    CUDA::curand
)

# Output to py-torch folder
set_target_properties(pysvpf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)