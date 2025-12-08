if (ATLAS_USE_CUDA)
    add_executable(cuda_smart_pointer main.cu)
    target_link_libraries(cuda_smart_pointer PRIVATE atlas-core)
endif ()