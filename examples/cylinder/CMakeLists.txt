if (ATLAS_USE_CUDA)
    # Build the CUDA entry point when the project is configured for the CUDA
    # backend. The .cu file reuses the same example logic as main.cpp.
    add_executable(cylinder main.cu)
    set_target_properties(cylinder PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
    )
else ()
    # Use the regular C++ entry point for the TBB / host-only backend.
    add_executable(cylinder main.cpp)
endif ()

target_link_libraries(cylinder PRIVATE atlas-core)
