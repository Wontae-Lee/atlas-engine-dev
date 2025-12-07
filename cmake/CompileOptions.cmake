# =====================================================
# CUDA settings
# =====================================================
if (ATLAS_USE_CUDA)
    message(STATUS "CUDA support is enabled.")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    add_compile_definitions(
            THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
            ATLAS_TASKING_CUDA
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
    if (EXISTS "/usr/local/cuda-12.9/include")
        include_directories("/usr/local/cuda-12.9/include")
    endif ()
else ()
    message(STATUS "CUDA support is disabled.")
    add_compile_definitions(ATLAS_TASKING_TBB)
    add_compile_definitions(
            THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    )
endif ()
