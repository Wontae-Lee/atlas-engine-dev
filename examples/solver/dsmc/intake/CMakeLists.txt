if (ATLAS_USE_CUDA)
    add_executable(solver main.cu)
    set_target_properties(solver PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED OFF
    )
else ()
    add_executable(solver main.cpp)
endif ()

target_link_libraries(solver PRIVATE atlas-core)
