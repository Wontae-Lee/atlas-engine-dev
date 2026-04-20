if (ATLAS_USE_CUDA)
    add_executable(cylinder main.cu)
    set_target_properties(cylinder PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED OFF
    )
else ()
    add_executable(cylinder main.cpp)
endif ()

target_link_libraries(cylinder PRIVATE atlas-core)
