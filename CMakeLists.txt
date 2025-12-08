# =====================================================
# Top-level CMake configuration for atlas-core-dev
# =====================================================
cmake_minimum_required(VERSION 3.20)

# NOTE:
# - LANGUAGES declares which languages this project may use.
# - CUDA is enabled separately via enable_language(CUDA) below.
project(atlas-core-dev LANGUAGES CXX)

# =====================================================
# C++ standard configuration
# =====================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =====================================================
# User-configurable options (cache variables)
# =====================================================
# Visualization / vizkit-related code
option(ATLAS_USE_VIZKIT "Build vizkit-based visualization modules" ON)

# CUDA acceleration (enabled if a CUDA toolkit is available)
option(ATLAS_USE_CUDA "Enable CUDA acceleration" ON)

# Testing and benchmarking
option(ATLAS_ENABLE_TESTS "Enable GoogleTest-based tests" OFF)
option(ATLAS_ENABLE_BENCH "Enable Google Benchmark-based benchmarks" OFF)

# Logging
option(ATLAS_LOGGING "Enable logging functionality" ON)

# =====================================================
# Global compile-time definitions
# =====================================================
if (ATLAS_LOGGING)
    # Enable logging in the codebase (guarded by ATLAS_ENABLE_LOGGING)
    add_compile_definitions(ATLAS_ENABLE_LOGGING)
endif ()

# =====================================================
# Global include directories
# NOTE:
# - These apply to all targets.
# - For better isolation, prefer target_include_directories() on
#   individual targets where possible.
# =====================================================
include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobj"
)

# =====================================================
# Common compile options (warnings, sanitizers, etc.)
# Implemented in: cmake/CompileOptions.cmake
# =====================================================
include(cmake/CompileOptions.cmake)

# =====================================================
# CUDA configuration (optional)
# - Only enables the CUDA language.
# - Actual CUDA sources / compile definitions should be configured
#   per-target (e.g., under src/atlas or CUDA-specific libs).
# =====================================================
if (ATLAS_USE_CUDA)
    message(STATUS "CUDA support is enabled.")

    # Enable CUDA language for this project (after project())
    enable_language(CUDA)

    # CUDA standard and target architectures
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Example: compute capability 8.6 (e.g., RTX 30xx series)
    set(CMAKE_CUDA_ARCHITECTURES 86)

    # If a specific CUDA toolkit installation is required, configure it here
    if (EXISTS "/usr/local/cuda-12.9/include")
        include_directories("/usr/local/cuda-12.9/include")
    endif ()
else ()
    message(STATUS "CUDA support is disabled.")
endif ()

# =====================================================
# Third-party (in-tree) libraries
# - tinyobjloader (mesh loading)
# =====================================================
add_subdirectory(external/tinyobj)

# =====================================================
# atlas-core library
# - Core engine + math + optional CUDA kernels
# - May be extended with vizkit / GL in subdirectories
# =====================================================
message(STATUS "[ATLAS] Engine core: ENABLED")

# Public include root (installed / exported headers)
set(ATLAS_CORE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(ATLAS_CORE_PUBLIC_DIR "${ATLAS_CORE_INCLUDE_DIR}/atlas")

# Public header set (library interface)
file(GLOB_RECURSE atlas_core_headers
        CONFIGURE_DEPENDS
        "${ATLAS_CORE_PUBLIC_DIR}/*.h"
)

# Main core library target
add_library(atlas-core ${atlas_core_headers})
add_library(atlas::core ALIAS atlas-core)

# C++ standard + public include path for consumers
target_compile_features(atlas-core PUBLIC cxx_std_20)
target_include_directories(atlas-core PUBLIC "${ATLAS_CORE_INCLUDE_DIR}")

# =====================================================
# Core CPU-side dependencies (always required)
# =====================================================
find_package(TBB REQUIRED)
target_link_libraries(atlas-core
        PUBLIC
        TBB::tbb
        tinyobjloader
)
# =====================================================
# Optional vizkit / OpenGL stack
#   - Controlled by top-level option ATLAS_USE_VIZKIT
#   - Exposes ATLAS_ENABLE_VIZKIT to the code when enabled
# =====================================================
if (ATLAS_USE_VIZKIT)
    # Add a compile definition that code can use:
    # #ifdef ATLAS_ENABLE_VIZKIT ... #endif
    target_compile_definitions(atlas-core
            PUBLIC
            ATLAS_ENABLE_VIZKIT
    )

    # include vizkit headers
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

    # GL / windowing stack
    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(GLEW REQUIRED)

    target_link_libraries(atlas-core
            PUBLIC
            glfw
            GLEW::GLEW
            GL
            GLU
            glut
    )
endif ()

# =====================================================
# Optional CUDA runtime linkage
# - Controlled by ATLAS_USE_CUDA
# - Assumes that CUDA::cudart is provided by CUDAToolkit
# =====================================================
if (ATLAS_USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if (TARGET CUDA::cudart)
        target_link_libraries(atlas-core PUBLIC CUDA::cudart)
    endif ()
endif ()


# =====================================================
# Tests (optional)
# - All test targets / infrastructure live under tests/
# =====================================================
if (ATLAS_ENABLE_TESTS)
    add_subdirectory(tests)
endif ()

# =====================================================
# Examples (optional)
# - Example executables live under examples/
# - Individual example CMakeLists can check ATLAS_USE_VIZKIT / ATLAS_USE_CUDA.
# =====================================================
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    message(STATUS "[ATLAS] Adding example modules.")
    add_subdirectory(examples)
endif ()

# =====================================================
# Benchmarks (optional)
# - Google Benchmark-based micro-benchmarks under benchmarks/
# =====================================================
if (ATLAS_ENABLE_BENCH)
    add_subdirectory(benchmarks)
endif ()

# =====================================================
# Build configuration summary
# =====================================================
message(STATUS "")
message(STATUS "========== ATLAS CONFIG SUMMARY ==========")
message(STATUS "C++ Standard          : ${CMAKE_CXX_STANDARD}")
message(STATUS "ATLAS_USE_CUDA        : ${ATLAS_USE_CUDA}")
message(STATUS "ATLAS_ENABLE_TESTS    : ${ATLAS_ENABLE_TESTS}")
message(STATUS "ATLAS_ENABLE_BENCH    : ${ATLAS_ENABLE_BENCH}")
if (ATLAS_USE_CUDA)
    message(STATUS "CUDA Archs            : ${CMAKE_CUDA_ARCHITECTURES}")
endif ()
message(STATUS "==========================================")
message(STATUS "")
