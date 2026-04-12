if (ATLAS_USE_CUDA)
    add_executable(overview main.cu)
    set_target_properties(overview PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED OFF
    )
else ()
    add_executable(overview main.cpp)
endif ()

target_link_libraries(overview PRIVATE atlas-core)
