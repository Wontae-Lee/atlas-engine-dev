# =====================================================
# Top-level CMake configuration for atlas-core-dev
# =====================================================
cmake_minimum_required(VERSION 3.20)

# LANGUAGES: languages directly enabled by project()
# CUDA is optionally enabled later via enable_language(CUDA)
project(atlas-engine-dev LANGUAGES CXX)

# =====================================================
# C++ standard configuration
# =====================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =====================================================
# User-configurable options (cache variables)
# =====================================================
# ATLAS_USE_VIZKIT:
#   Enables all vizkit-related visualization code under src/vizkit.
#   If enabled:
#     - ATLAS_ENABLE_VIZKIT compile definition becomes active.
#     - OpenGL/GLFW/GLEW/GLUT dependencies must be available.
option(ATLAS_USE_VIZKIT "Build vizkit-based visualization modules" ON)

# ATLAS_USE_CUDA:
#   Enables CUDA language support and optional CUDA kernels.
option(ATLAS_USE_CUDA "Enable CUDA acceleration" ON)

# Testing and benchmarking options
option(ATLAS_ENABLE_TESTS "Enable GoogleTest-based tests" OFF)
option(ATLAS_ENABLE_BENCH "Enable Google Benchmark-based benchmarks" OFF)

# Logging
option(ATLAS_LOGGING "Enable logging functionality" ON)

# =====================================================
# Global compile-time definitions
# =====================================================
if (ATLAS_LOGGING)
    # Enables ATLAS_ENABLE_LOGGING inside the C++ source code.
    add_compile_definitions(ATLAS_ENABLE_LOGGING)
endif ()

# =====================================================
# Global include directories
# These are applied to all targets; prefer target_include_directories()
# on specific targets wherever practical.
# =====================================================
include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobj"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/lyra/include"
)

# =====================================================
# Common compile options (clang-tidy, warnings, sanitizers, etc.)
# =====================================================
include(cmake/CompileOptions.cmake)

# =====================================================
# CUDA configuration (optional)
# - Only enables the CUDA language here.
# - Individual CUDA source files or libraries should configure their own
#   target properties under src/.
# =====================================================
if (ATLAS_USE_CUDA)
    message(STATUS "CUDA support is enabled.")
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Common default arch (sm_86 = NVIDIA RTX 30xx)
    set(CMAKE_CUDA_ARCHITECTURES 86)

    # Optional explicit CUDA include path
    if (EXISTS "/usr/local/cuda-12.9/include")
        include_directories("/usr/local/cuda-12.9/include")
    endif ()
else ()
    message(STATUS "CUDA support is disabled.")
endif ()

# =====================================================
# Third-party (in-tree) libs
# =====================================================
add_subdirectory(external/tinyobj)

# =====================================================
# atlas-core main library
# =====================================================
message(STATUS "[ATLAS] Engine core: ENABLED")

set(ATLAS_CORE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(ATLAS_CORE_PUBLIC_DIR  "${ATLAS_CORE_INCLUDE_DIR}/atlas")

# Collect all public headers
file(GLOB_RECURSE atlas_core_headers
        CONFIGURE_DEPENDS
        "${ATLAS_CORE_PUBLIC_DIR}/*.h"
)

add_library(atlas-core ${atlas_core_headers})
add_library(atlas::core ALIAS atlas-core)

# Public include path for users of atlas::core
target_compile_features(atlas-core PUBLIC cxx_std_20)
target_include_directories(atlas-core PUBLIC "${ATLAS_CORE_INCLUDE_DIR}")

# =====================================================
# Core CPU-side dependencies
# (TBB is always required)
# =====================================================
find_package(TBB REQUIRED)
target_link_libraries(atlas-core
        PUBLIC
        TBB::tbb
        tinyobjloader
)

# =====================================================
# Optional vizkit / OpenGL stack
#
# - Enabled by ATLAS_USE_VIZKIT.
# - Adds compile definition ATLAS_ENABLE_VIZKIT so C++ code can wrap
#   vizkit-specific features using:
#       #ifdef ATLAS_ENABLE_VIZKIT
#           ...
#       #endif
#
# - Includes src/ for vizkit headers (e.g., src/vizkit/*).
# - Links OpenGL windowing stack (GLFW, GLEW, GL, glut).
# =====================================================
if (ATLAS_USE_VIZKIT)
    message(STATUS "[ATLAS] Vizkit visualization: ENABLED")

    # C++ code can branch on this flag:
    #   #ifdef ATLAS_ENABLE_VIZKIT
    target_compile_definitions(atlas-core
            PUBLIC
            ATLAS_ENABLE_VIZKIT
    )

    # Add src/ so that atlas/vizkit/* headers become visible
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

    # GL + Windowing stack
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
else ()
    message(STATUS "[ATLAS] Vizkit visualization: DISABLED")
endif ()

# =====================================================
# Optional CUDA runtime linkage
# (If CUDA::cudart is found from CUDAToolkit)
# =====================================================
if (ATLAS_USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if (TARGET CUDA::cudart)
        target_link_libraries(atlas-core PUBLIC CUDA::cudart)
    endif ()
endif ()

# =====================================================
# Optional test modules
# =====================================================
if (ATLAS_ENABLE_TESTS)
    add_subdirectory(tests)
endif ()

# =====================================================
# Optional example executables
# =====================================================
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    message(STATUS "[ATLAS] Adding example modules.")
    add_subdirectory(examples)
endif ()

# =====================================================
# Optional benchmarks
# =====================================================
if (ATLAS_ENABLE_BENCH)
    add_subdirectory(benchmarks)
endif ()

# =====================================================
# Summary
# =====================================================
message(STATUS "")
message(STATUS "========== ATLAS CONFIG SUMMARY ==========")
message(STATUS "C++ Standard          : ${CMAKE_CXX_STANDARD}")
message(STATUS "ATLAS_USE_CUDA        : ${ATLAS_USE_CUDA}")
message(STATUS "ATLAS_ENABLE_TESTS    : ${ATLAS_ENABLE_TESTS}")
message(STATUS "ATLAS_ENABLE_BENCH    : ${ATLAS_ENABLE_BENCH}")
message(STATUS "ATLAS_USE_VIZKIT      : ${ATLAS_USE_VIZKIT}")

if (ATLAS_USE_CUDA)
    message(STATUS "CUDA Archs            : ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

message(STATUS "==========================================")
message(STATUS "")
