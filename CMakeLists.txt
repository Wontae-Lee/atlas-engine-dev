# Atlas build entry point. Detailed target configuration lives in cmake/.
# Native TBB requires no CUDA toolkit; CUDA and TBB-through-nvcc use nvcc.
# TBB is required for both backends. A GPU is needed only to run CUDA code.

# Project and language defaults -------------------------------------------------

cmake_minimum_required(VERSION 3.20)

project(
        atlas-engine-dev
        VERSION 0.1.0
        LANGUAGES C CXX
)

# Keep host and device code on C++20 without compiler-specific extensions.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Backend and feature options --------------------------------------------------

set(ATLAS_DEVICE_SYSTEM "TBB" CACHE STRING "Parallel backend: TBB (CPU) or CUDA (GPU)")
set_property(CACHE ATLAS_DEVICE_SYSTEM PROPERTY STRINGS TBB CUDA)

# Native TBB uses the host compiler. The optional nvcc path also checks
# host/device constructs that a native compiler would accept without validation.
set(ATLAS_HOST_COMPILER "native" CACHE STRING "TBB toolchain: native (CMAKE_CXX_COMPILER, e.g. g++) or nvcc")
set_property(CACHE ATLAS_HOST_COMPILER PROPERTY STRINGS native nvcc)

option(ATLAS_LOGGING      "Enable logging functionality" ON)
option(ATLAS_GOOGLE_TEST  "Build GoogleTest-based unit tests" ON)
option(ATLAS_PYTHON       "Build the nanobind-based Python bindings" ON)
option(ATLAS_INTERACTIVE  "Build Atlas interactive execution and control layer" ON)

include(CMakeDependentOption)
cmake_dependent_option(
        ATLAS_INTERACTIVE_RENDERING
        "Build Atlas interactive OpenGL rendering"
        ON
        "ATLAS_INTERACTIVE"
        OFF
)

option(ATLAS_BENCHMARKS   "Build and enable benchmark targets" ON)
option(ATLAS_EXAMPLES     "Build the example programs" ON)

# Configuration validation and derived flags -----------------------------------

include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (NOT ATLAS_DEVICE_SYSTEM MATCHES "^(TBB|CUDA)$")
    message(FATAL_ERROR
            "[ATLAS] Invalid ATLAS_DEVICE_SYSTEM='${ATLAS_DEVICE_SYSTEM}'. Expected TBB or CUDA.")
endif ()

if (NOT ATLAS_HOST_COMPILER MATCHES "^(native|nvcc)$")
    message(FATAL_ERROR
            "[ATLAS] Invalid ATLAS_HOST_COMPILER='${ATLAS_HOST_COMPILER}'. Expected native or nvcc.")
endif ()

# nvcc selects the compiler; Thrust selects CUDA buffers and algorithms.
# A TBB build through nvcc still uses the TBB backend.
if (ATLAS_DEVICE_SYSTEM STREQUAL "CUDA" OR ATLAS_HOST_COMPILER STREQUAL "nvcc")
    set(ATLAS_USE_NVCC ON)
else ()
    set(ATLAS_USE_NVCC OFF)
endif ()

if (ATLAS_DEVICE_SYSTEM STREQUAL "CUDA")
    set(ATLAS_USE_THRUST ON)
else ()
    set(ATLAS_USE_THRUST OFF)
endif ()

# Tests build only when CTest is enabled and GoogleTest is requested.
set(ATLAS_ENABLE_TESTS OFF)
if (BUILD_TESTING AND ATLAS_GOOGLE_TEST)
    set(ATLAS_ENABLE_TESTS ON)
endif ()

# Toolchain, dependencies, and Core --------------------------------------------

# Python extensions need position-independent static libraries. Set this before
# creating targets so they inherit it.
if (ATLAS_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif ()

if (ATLAS_USE_NVCC)
    include(CudaToolchain)
else ()
    message(STATUS "[ATLAS] Toolchain: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER}); no CUDA toolkit or Thrust")
endif ()

include(Dependencies)
include(AtlasLibraries)

# Optional targets ------------------------------------------------------------

# Tests define the shared GoogleTest helper before Interactive registers tests.
# Interactive targets must exist before their examples are added.
if (ATLAS_ENABLE_TESTS)
    add_subdirectory(tests)
elseif (ATLAS_GOOGLE_TEST)
    message(STATUS "[ATLAS] Tests requested but BUILD_TESTING=OFF. Skipping test targets.")
endif ()

if (ATLAS_PYTHON)
    add_subdirectory(bindings/python/atlas)
endif ()

if (ATLAS_INTERACTIVE)
    add_subdirectory(src/interactive)
endif ()

if (ATLAS_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()

if (ATLAS_EXAMPLES)
    add_subdirectory(examples)
endif ()

# Configuration summary -------------------------------------------------------
message(STATUS "")
message(STATUS "========== ATLAS CONFIG SUMMARY ==========")
message(STATUS "C++ Standard        : ${CMAKE_CXX_STANDARD}")
message(STATUS "C Compiler          : ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler        : ${CMAKE_CXX_COMPILER}")
message(STATUS "ATLAS_DEVICE_SYSTEM : ${ATLAS_DEVICE_SYSTEM}")
message(STATUS "ATLAS_HOST_COMPILER : ${ATLAS_HOST_COMPILER}")
message(STATUS "ATLAS_USE_NVCC      : ${ATLAS_USE_NVCC}")
message(STATUS "ATLAS_LOGGING       : ${ATLAS_LOGGING}")
message(STATUS "ATLAS_GOOGLE_TEST   : ${ATLAS_GOOGLE_TEST}")
message(STATUS "ATLAS_PYTHON        : ${ATLAS_PYTHON}")
message(STATUS "ATLAS_BENCHMARKS    : ${ATLAS_BENCHMARKS}")
message(STATUS "ATLAS_EXAMPLES      : ${ATLAS_EXAMPLES}")
message(STATUS "ATLAS_INTERACTIVE   : ${ATLAS_INTERACTIVE}")
message(STATUS "ATLAS_INTERACTIVE_RENDERING : ${ATLAS_INTERACTIVE_RENDERING}")
if (ATLAS_USE_NVCC)
    message(STATUS "CUDA Compiler       : ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Architectures  : ${CMAKE_CUDA_ARCHITECTURES}")
endif ()
message(STATUS "==========================================")
message(STATUS "")
