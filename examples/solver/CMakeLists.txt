if (ATLAS_USE_CUDA)
    add_executable(solver main.cu)
    set_target_properties(solver PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED OFF
    )
    target_link_libraries(solver PRIVATE atlas-core)
else ()
    add_executable(solver main.cpp)
    target_link_libraries(solver PRIVATE atlas-core)
endif ()
