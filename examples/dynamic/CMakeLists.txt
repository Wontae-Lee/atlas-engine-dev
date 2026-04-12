# Select the entry point for the active backend.
if (ATLAS_USE_CUDA)
    add_executable(dynamic main.cu)
    set_target_properties(dynamic PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED OFF
    )
else ()
    add_executable(dynamic main.cpp)
endif ()

# Link against the header-only core target used by all example variants.
target_link_libraries(dynamic PRIVATE atlas-core)
