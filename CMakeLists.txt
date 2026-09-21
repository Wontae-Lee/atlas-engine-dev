# Top-level CMake for atlas-engine-dev.
#
# ATLAS_DEVICE_SYSTEM selects the execution backend:
#   TBB  -> CPU build with the native C/C++ compilers (e.g. GCC/G++), no CUDA toolkit
#   CUDA -> GPU build with nvcc and Thrust
# ATLAS_HOST_COMPILER=nvcc also supports compiling the TBB backend with nvcc.
# TBB is required in every configuration; a GPU is needed only to run CUDA code.
#
# This file wires options and orchestrates the build; the details live in
# cmake/ modules and in the tests/, benchmarks/, and bindings/python subdirectories.

cmake_minimum_required(VERSION 3.20)

project(atlas-engine-dev LANGUAGES C CXX)

include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# C++20, no compiler-specific extensions (consistent host/device behavior).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options.
#   ATLAS_DEVICE_SYSTEM : parallel backend, TBB (CPU) or CUDA (GPU)
#   ATLAS_HOST_COMPILER : which compiler builds a TBB configuration
#   ATLAS_LOGGING       : compiled logging library + feature macros
#   ATLAS_GOOGLE_TEST   : GoogleTest-based unit tests
#   ATLAS_PYTHON        : nanobind Python bindings (bindings/python/atlas)
#   ATLAS_BENCHMARKS    : benchmark targets
set(ATLAS_DEVICE_SYSTEM "TBB" CACHE STRING "Parallel backend: TBB (CPU) or CUDA (GPU)")
set_property(CACHE ATLAS_DEVICE_SYSTEM PROPERTY STRINGS TBB CUDA)

# A TBB configuration needs neither nvcc nor Thrust, so by default it is built by the host
# compiler alone and the CUDA toolkit is not even looked for. Setting this to `nvcc` keeps
# the old behavior: a CPU build compiled by nvcc. That is worth having in CI, because nvcc
# rejects constructs the host compiler accepts -- notably an extended __host__ __device__
# lambda inside a private member function -- and a TBB-only build would stop catching them.
set(ATLAS_HOST_COMPILER "native" CACHE STRING "TBB toolchain: native (CMAKE_CXX_COMPILER, e.g. g++) or nvcc")
set_property(CACHE ATLAS_HOST_COMPILER PROPERTY STRINGS native nvcc)

option(ATLAS_LOGGING      "Enable logging functionality" ON)
option(ATLAS_GOOGLE_TEST  "Build GoogleTest-based unit tests" ON)
option(ATLAS_PYTHON       "Build the nanobind-based Python bindings" OFF)
option(ATLAS_BENCHMARKS   "Build and enable benchmark targets" OFF)
option(ATLAS_EXAMPLES     "Build the example programs" OFF)

if (NOT ATLAS_DEVICE_SYSTEM MATCHES "^(TBB|CUDA)$")
    message(FATAL_ERROR
            "[ATLAS] Invalid ATLAS_DEVICE_SYSTEM='${ATLAS_DEVICE_SYSTEM}'. Expected TBB or CUDA.")
endif ()

if (NOT ATLAS_HOST_COMPILER MATCHES "^(native|nvcc)$")
    message(FATAL_ERROR
            "[ATLAS] Invalid ATLAS_HOST_COMPILER='${ATLAS_HOST_COMPILER}'. Expected native or nvcc.")
endif ()

# Two independent switches fall out of the pair above.
#
#   ATLAS_USE_NVCC    : the .cu sources are compiled by nvcc, so the CUDA toolkit is needed
#                       to build (not necessarily to run).
#   ATLAS_USE_THRUST  : the buffers and parallel algorithms are Thrust's. Only the CUDA
#                       device system defines ATLAS_BACKEND_CUDA; a TBB build defines
#                       ATLAS_BACKEND_TBB instead.
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

# The nanobind Python bindings are a shared object; every static dependency
# linked into it (atlas, protobuf, absl, ...) must be position-independent.
# Set this before the dependency/library targets are defined so they inherit it.
if (ATLAS_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif ()

# Toolchain, dependencies, and library targets (cmake/ modules).
if (ATLAS_USE_NVCC)
    include(CudaToolchain)
else ()
    message(STATUS "[ATLAS] Toolchain: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER}); no CUDA toolkit or Thrust")
endif ()

include(Dependencies)
include(AtlasLibraries)

# Tests, benchmarks, and Python bindings own their own configuration.
if (ATLAS_ENABLE_TESTS)
    add_subdirectory(tests)
elseif (ATLAS_GOOGLE_TEST)
    message(STATUS "[ATLAS] Tests requested but BUILD_TESTING=OFF. Skipping test targets.")
endif ()

add_subdirectory(benchmarks)
add_subdirectory(examples)

if (ATLAS_PYTHON)
    add_subdirectory(bindings/python/atlas)
endif ()

# Configuration summary.
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
if (ATLAS_USE_NVCC)
    message(STATUS "CUDA Compiler       : ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Architectures  : ${CMAKE_CUDA_ARCHITECTURES}")
endif ()
message(STATUS "==========================================")
message(STATUS "")
