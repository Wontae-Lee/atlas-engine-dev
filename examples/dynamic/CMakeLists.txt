if (ATLAS_USE_CUDA)
    add_executable(dynamic main.cu)
    set_target_properties(dynamic PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED OFF
    )
    target_link_libraries(dynamic PRIVATE atlas-core)
else ()
    add_executable(dynamic main.cpp)
    target_link_libraries(dynamic PRIVATE atlas-core)
endif ()
