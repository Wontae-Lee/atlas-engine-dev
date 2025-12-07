# =====================================================
# Top-level CMake configuration for atlas-engine-dev
# =====================================================
cmake_minimum_required(VERSION 3.20)

# NOTE:
# - LANGUAGES only declares which languages this project may use.
# - CUDA is enabled later conditionally via enable_language(CUDA).
project(atlas-engine-dev LANGUAGES CXX)

# =====================================================
# C++ standard configuration
# =====================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =====================================================
# User options (cache variables)
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
    # Enable logging in the codebase (checked via ATLAS_ENABLE_LOGGING)
    add_compile_definitions(ATLAS_ENABLE_LOGGING)
endif ()

# =====================================================
# Resource directory
#   - Exposed via RESOURCES_DIR macro to the code
# =====================================================
set(RESOURCES_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        CACHE PATH "Path to runtime resources"
)
add_compile_definitions(RESOURCES_DIR="${RESOURCES_DIR}")

# =====================================================
# Common include directories
#   NOTE:
#   - These are global; you may prefer target_include_directories()
#     inside each library/executable for better isolation.
# =====================================================
include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobj"
)

# =====================================================
# Common compile options (warnings, sanitizers, etc.)
#   - Implemented in cmake/CompileOptions.cmake
# =====================================================
include(cmake/CompileOptions.cmake)

# =====================================================
# CUDA configuration (optional)
#   - Only enables CUDA language; actual usage is handled in subdirs.
#   - Do NOT put global THRUST / ATLAS_TASKING_* macros here anymore;
#     they should be applied per-target in src/atlas or other CUDA libs.
# =====================================================
if (ATLAS_USE_CUDA)
    message(STATUS "CUDA support is enabled.")

    # Enable CUDA language in this project (after project())
    enable_language(CUDA)

    # CUDA standard and architectures; adjust as needed.
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Example: compute capability 8.6 (RTX 30xx)
    set(CMAKE_CUDA_ARCHITECTURES 86)

    # If a specific CUDA toolkit install is required, configure it here
    if (EXISTS "/usr/local/cuda-12.9/include")
        include_directories("/usr/local/cuda-12.9/include")
    endif ()
else ()
    message(STATUS "CUDA support is disabled.")
endif ()

# =====================================================
# Third-party (in-tree) libraries
#   - tinyobjloader (for mesh loading)
# =====================================================
add_subdirectory(external/tinyobj)

# =====================================================
# atlas core / engine (src/atlas)
#   - Contains main math, system, CUDA kernels, etc.
#   - Controlled by ATLAS_USE_ENGINE option.
# =====================================================
add_subdirectory(src/atlas)

# =====================================================
# Tests (optional)
#   - All test targets / infrastructure live under src/tests
# =====================================================
if (ATLAS_ENABLE_TESTS)
    add_subdirectory(src/tests)
endif ()

# =====================================================
# Examples (optional)
#   - Multiple example modules live under src/examples
#   - Individual example CMakeLists can check ATLAS_USE_VIZKIT / ATLAS_USE_CUDA
#     and add their own executables conditionally.
# =====================================================
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/examples")
    message(STATUS "[ATLAS] Adding example modules.")
    add_subdirectory(src/examples)
endif ()

# =====================================================
# Benchmarks (optional)
#   - Google Benchmark-based micro-benchmarks under src/benchmarks
# =====================================================
if (ATLAS_ENABLE_BENCH)
    add_subdirectory(src/benchmarks)
endif ()

# =====================================================
# Build configuration summary
# =====================================================
message(STATUS "")
message(STATUS "========== ATLAS CONFIG SUMMARY ==========")
message(STATUS "C++ Standard          : ${CMAKE_CXX_STANDARD}")
message(STATUS "ATLAS_USE_ENGINE      : ${ATLAS_USE_ENGINE}")
message(STATUS "ATLAS_USE_VIZKIT      : ${ATLAS_USE_VIZKIT}")
message(STATUS "ATLAS_USE_CUDA        : ${ATLAS_USE_CUDA}")
message(STATUS "ATLAS_ENABLE_TESTS    : ${ATLAS_ENABLE_TESTS}")
message(STATUS "ATLAS_ENABLE_BENCH    : ${ATLAS_ENABLE_BENCH}")
if (ATLAS_USE_CUDA)
    message(STATUS "CUDA Archs            : ${CMAKE_CUDA_ARCHITECTURES}")
endif ()
message(STATUS "Resources Dir         : ${RESOURCES_DIR}")
message(STATUS "==========================================")
message(STATUS "")
