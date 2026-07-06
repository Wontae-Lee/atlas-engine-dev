# Top-level CMake for atlas-engine-dev.
#
# Single toolchain: nvcc compiles every Atlas TU. The execution backend is the
# Thrust device system, chosen by ATLAS_DEVICE_SYSTEM:
#   TBB  -> CPU build (THRUST_DEVICE_SYSTEM=TBB, no GPU code generated)
#   CUDA -> GPU build (THRUST_DEVICE_SYSTEM=CUDA)
# The Thrust host system is always TBB; a GPU is needed only to run the CUDA
# variant, never to build it.
#
# This file wires options and orchestrates the build; the details live in
# cmake/ modules and in the tests/, benchmarks/, and src/python subdirectories.

cmake_minimum_required(VERSION 3.20)

project(atlas-engine-dev LANGUAGES CXX)

include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# C++20, no compiler-specific extensions (consistent host/device behavior).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options.
#   ATLAS_DEVICE_SYSTEM : Thrust device system / backend, TBB (CPU) or CUDA (GPU)
#   ATLAS_LOGGING       : compiled logging library + feature macros
#   ATLAS_GOOGLE_TEST   : GoogleTest-based unit tests
#   ATLAS_PYTHON        : nanobind Python bindings (src/python/atlas)
#   ATLAS_BENCHMARKS    : benchmark targets (backend permitting)
set(ATLAS_DEVICE_SYSTEM "TBB" CACHE STRING "Thrust device system: TBB (CPU) or CUDA (GPU)")
set_property(CACHE ATLAS_DEVICE_SYSTEM PROPERTY STRINGS TBB CUDA)

option(ATLAS_LOGGING      "Enable logging functionality" ON)
option(ATLAS_GOOGLE_TEST  "Build GoogleTest-based unit tests" ON)
option(ATLAS_PYTHON       "Build the nanobind-based Python bindings" OFF)
option(ATLAS_BENCHMARKS   "Build and enable benchmark targets" OFF)

if (NOT ATLAS_DEVICE_SYSTEM MATCHES "^(TBB|CUDA)$")
    message(FATAL_ERROR
            "[ATLAS] Invalid ATLAS_DEVICE_SYSTEM='${ATLAS_DEVICE_SYSTEM}'. Expected TBB or CUDA.")
endif ()

# Tests build only when CTest is enabled and GoogleTest is requested.
set(ATLAS_ENABLE_TESTS OFF)
if (BUILD_TESTING AND ATLAS_GOOGLE_TEST)
    set(ATLAS_ENABLE_TESTS ON)
endif ()

# The nanobind Python bindings are a shared object; every static dependency
# linked into it (atlas, protobuf, absl, ...) must be position-independent.
# Set this before the dependency/library targets are defined so they inherit it.
if (ATLAS_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif ()

# Toolchain, dependencies, and library targets (cmake/ modules).
include(CudaToolchain)
include(Dependencies)
include(AtlasLibraries)

# Tests, benchmarks, and Python bindings own their own configuration.
if (ATLAS_ENABLE_TESTS)
    add_subdirectory(tests)
elseif (ATLAS_GOOGLE_TEST)
    message(STATUS "[ATLAS] Tests requested but BUILD_TESTING=OFF. Skipping test targets.")
endif ()

add_subdirectory(benchmarks)

if (ATLAS_PYTHON)
    add_subdirectory(src/python/atlas)
endif ()

# Configuration summary.
message(STATUS "")
message(STATUS "========== ATLAS CONFIG SUMMARY ==========")
message(STATUS "C++ Standard        : ${CMAKE_CXX_STANDARD}")
message(STATUS "ATLAS_DEVICE_SYSTEM : ${ATLAS_DEVICE_SYSTEM}")
message(STATUS "ATLAS_LOGGING       : ${ATLAS_LOGGING}")
message(STATUS "ATLAS_GOOGLE_TEST   : ${ATLAS_GOOGLE_TEST}")
message(STATUS "ATLAS_PYTHON        : ${ATLAS_PYTHON}")
message(STATUS "ATLAS_BENCHMARKS    : ${ATLAS_BENCHMARKS}")
message(STATUS "CUDA Architectures  : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "==========================================")
message(STATUS "")
