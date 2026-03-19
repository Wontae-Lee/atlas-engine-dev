# =====================================================
# Top-level CMake configuration for atlas-engine-dev
# (minimal options: TASKING (CUDA/TBB) + VIZKIT + LOGGING)
# =====================================================
cmake_minimum_required(VERSION 3.20)

project(atlas-engine-dev LANGUAGES CXX)

include(CTest)

# =====================================================
# C++ standard configuration
# =====================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =====================================================
# User-configurable options
# =====================================================
option(ATLAS_USE_CUDA "Enable CUDA tasking backend" OFF )
option(ATLAS_USE_TBB "Enable TBB tasking backend" ON)
option(ATLAS_USE_VIZKIT "Build vizkit-based visualization modules" ON)
option(ATLAS_LOGGING "Enable logging functionality" ON)
option(ATLAS_TESTS "Build and enable tests" ON)
option(ATLAS_BENCHMARKS "Build and enable benchmarks" ON)

# =====================================================
# Enforce EXACTLY ONE tasking backend
# =====================================================
if (ATLAS_USE_CUDA AND ATLAS_USE_TBB)
    message(FATAL_ERROR "[ATLAS] Invalid configuration: ATLAS_USE_CUDA and ATLAS_USE_TBB cannot both be ON.")
endif ()

if ((NOT ATLAS_USE_CUDA) AND (NOT ATLAS_USE_TBB))
    message(FATAL_ERROR "[ATLAS] Invalid configuration: one of ATLAS_USE_CUDA or ATLAS_USE_TBB must be ON.")
endif ()

if (ATLAS_USE_CUDA AND ATLAS_TESTS)
    message(WARNING "[ATLAS] ATLAS_TESTS are not yet compatible with CUDA backend. Disabling tests.")
    set(ATLAS_TESTS OFF)
endif ()

if (ATLAS_USE_CUDA AND ATLAS_BENCHMARKS)
    message(WARNING "[ATLAS] ATLAS_BENCHMARKS are not yet compatible with CUDA backend. Disabling benchmarks.")
    set(ATLAS_BENCHMARKS OFF)
endif ()


# =====================================================
# Third-party (in-tree) libs
# =====================================================
add_subdirectory(external/tinyobj)

# =====================================================
# Dependencies (conditional)
# =====================================================
if (ATLAS_USE_TBB)
    find_package(TBB REQUIRED)
endif ()

# =====================================================
# atlas-core (header-only) main library
# =====================================================
message(STATUS "[ATLAS] Engine core: ENABLED")

set(ATLAS_CORE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(atlas-core INTERFACE)
add_library(atlas::core ALIAS atlas-core)

target_compile_features(atlas-core INTERFACE cxx_std_20)

target_include_directories(atlas-core
        INTERFACE
        "${ATLAS_CORE_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobj"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/lyra/include"
)

# =====================================================
# TASKING backend selection
# =====================================================
if (ATLAS_USE_CUDA)
    message(STATUS "[ATLAS] Tasking backend: CUDA")
    if (EXISTS "/usr/local/cuda-12.9/include")
        set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)
    endif ()
    find_package(TBB REQUIRED)
    target_link_libraries(atlas-core INTERFACE TBB::tbb)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Choose your default arch
    set(CMAKE_CUDA_ARCHITECTURES 86)

    # Prefer NOT setting nvcc path unless required by your environment.
    # set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)

    target_compile_definitions(atlas-core INTERFACE
            ATLAS_TASKING_CUDA
            THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
    )

    # nvcc flags (target-scoped is better, but CMake has quirks for INTERFACE CUDA flags)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

    # Optional cudart
    find_package(CUDAToolkit QUIET)
    if (TARGET CUDA::cudart)
        target_link_libraries(atlas-core INTERFACE CUDA::cudart)
    endif ()

    # Optional include dir
    if (EXISTS "/usr/local/cuda-12.9/include")
        target_include_directories(atlas-core SYSTEM INTERFACE "/usr/local/cuda-12.9/include")
    endif ()

else ()
    message(STATUS "[ATLAS] Tasking backend: TBB")

    target_compile_definitions(atlas-core INTERFACE
            ATLAS_TASKING_TBB
            THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    )

    target_link_libraries(atlas-core INTERFACE TBB::tbb)
endif ()

# tinyobj is used by core headers
target_link_libraries(atlas-core INTERFACE tinyobjloader)

# =====================================================
# atlas-logging (compiled implementation)
# =====================================================
if (ATLAS_LOGGING)
    message(STATUS "[ATLAS] Logging: ENABLED")

    add_library(atlas-logging STATIC
            "${CMAKE_CURRENT_SOURCE_DIR}/src/logging/logging.cpp"
    )
    add_library(atlas::logging ALIAS atlas-logging)

    target_compile_features(atlas-logging PUBLIC cxx_std_20)
    target_include_directories(atlas-logging PUBLIC "${ATLAS_CORE_INCLUDE_DIR}")

    target_compile_definitions(atlas-core INTERFACE ATLAS_ENABLE_LOGGING)
    target_compile_definitions(atlas-logging PUBLIC ATLAS_ENABLE_LOGGING)

    # Make atlas-core consumers automatically link logging impl
    target_link_libraries(atlas-core INTERFACE atlas::logging)

else ()
    message(STATUS "[ATLAS] Logging: DISABLED")
endif ()

# =====================================================
# Optional vizkit / OpenGL stack
# =====================================================
if (ATLAS_USE_VIZKIT)
    message(STATUS "[ATLAS] Vizkit visualization: ENABLED")

    target_compile_definitions(atlas-core INTERFACE ATLAS_ENABLE_VIZKIT)
    target_include_directories(atlas-core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(GLEW REQUIRED)

    target_link_libraries(atlas-core
            INTERFACE
            glfw
            GLEW::GLEW
            GL
            GLU
            glut
    )
else ()
    message(STATUS "[ATLAS] Vizkit visualization: DISABLED")
endif ()

# =====================================================
# Tests
# =====================================================
if (ATLAS_TESTS)
    add_subdirectory(external/googletest)
    add_subdirectory(tests)
endif ()

if (ATLAS_BENCHMARKS)
    add_subdirectory(external/googlebenchmark)
    add_subdirectory(benchmarks)
endif ()


# =====================================================
# Examples
# =====================================================
add_subdirectory(examples/solver)
add_subdirectory(examples/dynamic)

# =====================================================
# Summary
# =====================================================
message(STATUS "")
message(STATUS "========== ATLAS CONFIG SUMMARY ==========")
message(STATUS "C++ Standard       : ${CMAKE_CXX_STANDARD}")
message(STATUS "ATLAS_USE_CUDA     : ${ATLAS_USE_CUDA}")
message(STATUS "ATLAS_USE_TBB      : ${ATLAS_USE_TBB}")
message(STATUS "ATLAS_USE_VIZKIT   : ${ATLAS_USE_VIZKIT}")
message(STATUS "ATLAS_LOGGING      : ${ATLAS_LOGGING}")
message(STATUS "ATLAS_TESTS       : ${ATLAS_TESTS}")
message(STATUS "ATLAS_BENCHMARKS   : ${ATLAS_BENCHMARKS}")
if (ATLAS_USE_CUDA)
    message(STATUS "CUDA Archs         : ${CMAKE_CUDA_ARCHITECTURES}")
endif ()
message(STATUS "==========================================")
message(STATUS "")
