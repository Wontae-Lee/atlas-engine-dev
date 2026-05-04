# ============================================================
# Top-level CMake configuration for atlas-engine-dev
#
# This file is the single entry point for configuring the build
# of atlas-engine-dev. It is responsible for:
#
#   1. defining global language and project settings
#   2. exposing user-selectable build options
#   3. validating mutually exclusive or unsupported options
#   4. configuring the selected tasking backend
#   5. wiring optional subsystems such as logging and visualization
#   6. creating test and benchmark targets when requested
#   7. printing a final configuration summary for diagnostics
#
# The overall design goal is to keep the top-level build readable,
# predictable, and easy to diagnose when configuration issues occur.
# ============================================================

cmake_minimum_required(VERSION 3.20)

project(atlas-engine-dev LANGUAGES CXX)

include(CTest)

# ============================================================
# Global language configuration
#
# The project is built in C++20 mode and explicitly disables
# compiler-specific language extensions. This ensures that:
#
#   - all targets see a consistent language standard
#   - non-portable GNU/MSVC extensions are avoided by default
#   - template and constexpr-heavy code behaves consistently
#     across supported toolchains
#
# These settings are applied globally because the project treats
# C++20 as a baseline requirement rather than an optional feature.
# ============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# User-configurable build options
#
# The project supports two mutually exclusive tasking backends:
#
#   - ATLAS_USE_CUDA:
#       Enables the CUDA-based execution path.
#
#   - ATLAS_USE_TBB:
#       Enables the CPU/TBB-based execution path.
#
# Exactly one of these backends must be enabled.
#
# Additional options control optional subsystems:
#
#   - ATLAS_USE_VIZKIT:
#       Enables visualization components built on top of the
#       Vizkit/OpenGL dependency stack.
#
#   - ATLAS_LOGGING:
#       Enables the compiled logging implementation and exposes
#       logging-related feature macros to downstream consumers.
#
#   - ATLAS_GOOGLE_TEST:
#       Enables GoogleTest-based C++ unit tests.
#
#   - ATLAS_CUDA_TEST:
#       Enables CUDA-specific test targets compiled from .cu files.
#
#   - ATLAS_BENCHMARKS:
#       Enables benchmark targets when supported by the selected
#       backend configuration.
# ============================================================
option(ATLAS_USE_CUDA     "Enable the CUDA tasking backend" OFF)
option(ATLAS_USE_TBB      "Enable the TBB tasking backend" ON)
option(ATLAS_USE_VIZKIT   "Build Vizkit-based visualization modules" ON)
option(ATLAS_LOGGING      "Enable logging functionality" ON)
option(ATLAS_GOOGLE_TEST  "Build GoogleTest-based C++ tests" ON)
option(ATLAS_CUDA_TEST    "Build CUDA-specific tests" OFF)
option(ATLAS_BENCHMARKS   "Build and enable benchmark targets" ON)

# ============================================================
# NVIDIA utility discovery
#
# nvidia-smi is used only as a best-effort helper for querying
# the local GPU environment. It is not a hard build dependency.
#
# When available, it allows the configuration step to:
#
#   - query the installed NVIDIA driver version
#   - detect the CUDA runtime version reported by the driver
#   - detect the GPU compute capability
#
# These values are used for diagnostics and for selecting a more
# appropriate default CUDA architecture when the user has not
# provided one explicitly.
# ============================================================
find_program(ATLAS_NVIDIA_SMI_EXECUTABLE
        NAMES nvidia-smi
        HINTS /usr/bin /bin /usr/local/nvidia/bin
)

# ============================================================
# atlas_query_nvidia_smi(out_var, query_name)
#
# Execute a structured nvidia-smi query and return the first
# reported value through the parent scope.
#
# Parameters:
#   out_var
#       Name of the output variable that will receive the result.
#
#   query_name
#       Field name passed to --query-gpu=<query_name>.
#
# Behavior:
#   - returns an empty string if nvidia-smi is unavailable
#   - returns an empty string if the query fails
#   - returns the first GPU's value when multiple GPUs are present
#
# This helper intentionally hides command invocation details so
# that the CUDA configuration path remains easier to read.
# ============================================================
function(atlas_query_nvidia_smi out_var query_name)
    if (NOT ATLAS_NVIDIA_SMI_EXECUTABLE)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif ()

    execute_process(
            COMMAND "${ATLAS_NVIDIA_SMI_EXECUTABLE}" "--query-gpu=${query_name}" "--format=csv,noheader"
            OUTPUT_VARIABLE atlas_query_output
            ERROR_VARIABLE atlas_query_error
            RESULT_VARIABLE atlas_query_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT atlas_query_result EQUAL 0 OR atlas_query_output STREQUAL "")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif ()

    string(REPLACE "\n" ";" atlas_query_lines "${atlas_query_output}")
    list(GET atlas_query_lines 0 atlas_query_first_line)
    string(STRIP "${atlas_query_first_line}" atlas_query_first_line)

    set(${out_var} "${atlas_query_first_line}" PARENT_SCOPE)
endfunction()

# ============================================================
# atlas_detect_cuda_driver_runtime_version(out_var)
#
# Extract the CUDA runtime version reported by the NVIDIA driver
# from the human-readable nvidia-smi output.
#
# Parameters:
#   out_var
#       Name of the output variable that will receive the detected
#       CUDA runtime version, such as "12.4".
#
# Behavior:
#   - returns an empty string if nvidia-smi is unavailable
#   - returns an empty string if the command fails
#   - returns an empty string if the expected text pattern cannot
#     be extracted from the output
#
# This value is useful for warning about mismatches between the
# installed CUDA toolkit and the maximum runtime/PTX support level
# exposed by the current NVIDIA driver.
# ============================================================
function(atlas_detect_cuda_driver_runtime_version out_var)
    if (NOT ATLAS_NVIDIA_SMI_EXECUTABLE)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif ()

    execute_process(
            COMMAND "${ATLAS_NVIDIA_SMI_EXECUTABLE}"
            OUTPUT_VARIABLE atlas_smi_output
            ERROR_VARIABLE atlas_smi_error
            RESULT_VARIABLE atlas_smi_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT atlas_smi_result EQUAL 0)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif ()

    string(REGEX MATCH "CUDA Version: ([0-9]+\\.[0-9]+)" atlas_cuda_runtime_match "${atlas_smi_output}")
    set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

# ============================================================
# atlas_detect_cuda_architectures(out_var)
#
# Detect the installed GPU compute capability and convert it into
# a value suitable for CMAKE_CUDA_ARCHITECTURES.
#
# Example:
#   compute capability "8.9" -> "89-real"
#
# Parameters:
#   out_var
#       Name of the output variable that will receive the detected
#       architecture string.
#
# Behavior:
#   - returns an empty string when detection is unavailable
#   - emits a real architecture only, which favors native SASS
#     generation and avoids depending on PTX JIT at runtime
#
# This is intentionally conservative because native code generation
# is usually more predictable on deployed systems than PTX fallback.
# ============================================================
function(atlas_detect_cuda_architectures out_var)
    atlas_query_nvidia_smi(atlas_detected_compute_cap "compute_cap")

    if (atlas_detected_compute_cap STREQUAL "")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif ()

    string(REPLACE "." "" atlas_detected_compute_cap "${atlas_detected_compute_cap}")
    set(${out_var} "${atlas_detected_compute_cap}-real" PARENT_SCOPE)
endfunction()

# ============================================================
# Configuration validation
#
# This section validates the build option matrix before any major
# targets or dependencies are configured.
#
# Validation is performed early so that unsupported combinations
# fail with explicit messages instead of producing confusing build
# errors later in the configuration or compilation phases.
#
# Rules enforced here:
#
#   1. exactly one tasking backend must be selected
#   2. benchmarks are disabled for the CUDA backend
#   3. GoogleTest-based tests are disabled for the CUDA backend
#
# The benchmark and GoogleTest cases are downgraded to warnings
# and auto-corrected because the user's broader configuration can
# still proceed successfully without those optional features.
# ============================================================
if (ATLAS_USE_CUDA AND ATLAS_USE_TBB)
    message(FATAL_ERROR
            "[ATLAS] Invalid configuration: "
            "ATLAS_USE_CUDA and ATLAS_USE_TBB cannot both be ON."
    )
endif ()

if ((NOT ATLAS_USE_CUDA) AND (NOT ATLAS_USE_TBB))
    message(FATAL_ERROR
            "[ATLAS] Invalid configuration: "
            "one of ATLAS_USE_CUDA or ATLAS_USE_TBB must be ON."
    )
endif ()

if (ATLAS_USE_CUDA AND ATLAS_BENCHMARKS)
    message(WARNING
            "[ATLAS] ATLAS_BENCHMARKS are not yet compatible with the CUDA backend. "
            "Disabling benchmarks."
    )
    set(ATLAS_BENCHMARKS OFF)
endif ()

if (ATLAS_USE_CUDA AND ATLAS_GOOGLE_TEST)
    message(WARNING
            "[ATLAS] GoogleTest-based tests are disabled for the CUDA backend. "
            "Disabling ATLAS_GOOGLE_TEST."
    )
    set(ATLAS_GOOGLE_TEST OFF)
endif ()

# ============================================================
# Unified test enable flag
#
# The build may expose multiple test families, but actual test
# target creation is gated behind a single derived flag.
#
# Tests are enabled only when:
#
#   - BUILD_TESTING is ON
#   - and at least one concrete test family has been requested
#
# This keeps downstream conditionals simpler and makes it easier
# to reason about whether the project is expected to produce any
# test targets at all.
# ============================================================
set(ATLAS_ENABLE_TESTS OFF)
if (BUILD_TESTING AND (ATLAS_GOOGLE_TEST OR ATLAS_CUDA_TEST))
    set(ATLAS_ENABLE_TESTS ON)
endif ()

# ============================================================
# Third-party dependencies bundled with the source tree
#
# These dependencies are built as part of the current project
# tree and do not rely on a separate system installation.
# ============================================================
add_subdirectory(external/tinyobj)

# ============================================================
# Protobuf runtime and code generation
#
# Atlas uses a vendored protobuf source tree under
# `external/protobuf`. Both the protobuf C++ runtime and the
# `protoc` compiler are built from that tree so no separate
# prebuilt archive is required.
#
# Keep protobuf's own CMake output under the active Atlas build
# tree, not under the vendored source checkout. Atlas-generated
# files from our .proto schemas are kept in a separate generated/
# directory in the same build tree.
# ============================================================
unset(WITH_PROTOC CACHE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_CONFORMANCE OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "" FORCE)
set(protobuf_BUILD_LIBPROTOC ON CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_USE_UNITY_BUILD OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

set(ATLAS_PROTOBUF_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/protobuf")
set(ATLAS_PROTOBUF_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf")
add_subdirectory("${ATLAS_PROTOBUF_SOURCE_DIR}" "${ATLAS_PROTOBUF_BINARY_DIR}" EXCLUDE_FROM_ALL)

set(ATLAS_PROTO_SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/serialization/proto")
set(ATLAS_PROTO_SCHEMA "${ATLAS_PROTO_SCHEMA_DIR}/atlas_snapshot.proto")
set(ATLAS_PROTO_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ATLAS_PROTO_SOURCES
        "${ATLAS_PROTO_OUTPUT_DIR}/atlas_snapshot.pb.cc"
        "${ATLAS_PROTO_OUTPUT_DIR}/atlas_snapshot.pb.h")

file(MAKE_DIRECTORY "${ATLAS_PROTO_OUTPUT_DIR}")

add_custom_command(
        OUTPUT ${ATLAS_PROTO_SOURCES}
        COMMAND protobuf::protoc
        ARGS
        "--proto_path=${ATLAS_PROTO_SCHEMA_DIR}"
        "--cpp_out=${ATLAS_PROTO_OUTPUT_DIR}"
        "atlas_snapshot.proto"
        DEPENDS "${ATLAS_PROTO_SCHEMA}" protobuf::protoc
        WORKING_DIRECTORY "${ATLAS_PROTO_SCHEMA_DIR}"
        VERBATIM
)

add_library(atlas-serialization STATIC
        ${ATLAS_PROTO_OUTPUT_DIR}/atlas_snapshot.pb.cc
        src/serialization/protobuf_snapshot.cpp
)
add_library(atlas::serialization ALIAS atlas-serialization)

target_include_directories(atlas-serialization
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        PRIVATE
        "${ATLAS_PROTO_OUTPUT_DIR}"
)

target_link_libraries(atlas-serialization
        PUBLIC
        protobuf::libprotobuf
)

# ============================================================
# External package dependencies
#
# TBB is treated as a required package for all configurations.
#
# Reason:
#   - the TBB backend links it directly
#   - the CUDA backend still uses TBB as the selected host-side
#     backend for Thrust
#
# As a result, the build does not currently support a mode in
# which TBB is entirely absent.
# ============================================================
find_package(TBB REQUIRED)

# ============================================================
# atlas-core
#
# atlas-core is the primary interface target exposed by the
# project. It is header-only and acts as the central propagation
# point for:
#
#   - public include directories
#   - compile definitions
#   - transitive dependency links
#
# Consumers are expected to link against atlas::core rather than
# assembling includes and dependencies manually.
# ============================================================
message(STATUS "[ATLAS] Engine core: ENABLED")

set(ATLAS_CORE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(atlas-core INTERFACE)
add_library(atlas::core ALIAS atlas-core)

target_include_directories(atlas-core
        INTERFACE
        "${ATLAS_CORE_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobj"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/lyra/include"
)

# tinyobj is referenced by public atlas-core headers. Because it
# appears in the public header surface, it must be propagated as a
# transitive dependency to downstream users of atlas-core.
target_link_libraries(atlas-core INTERFACE tinyobjloader)
target_link_libraries(atlas-core INTERFACE atlas::serialization)

# ============================================================
# Tasking backend selection
#
# Exactly one execution backend is configured in this section.
#
# CUDA path responsibilities:
#   - locate nvcc when possible
#   - inspect the local NVIDIA environment for diagnostics
#   - enable the CUDA language in CMake
#   - choose a reasonable default CUDA architecture
#   - define backend-selection macros for atlas-core
#   - link host/runtime dependencies required by Thrust and CUDA
#
# TBB path responsibilities:
#   - define backend-selection macros for atlas-core
#   - link the required TBB dependency
#
# The goal is to make backend selection transparent to downstream
# targets that simply consume atlas::core.
# ============================================================
if (ATLAS_USE_CUDA)
    message(STATUS "[ATLAS] Tasking backend: CUDA")

    # --------------------------------------------------------
    # CUDA compiler discovery
    #
    # If the CUDA compiler has not already been selected by the
    # toolchain or by the CUDACXX environment variable, try to
    # discover nvcc from common installation paths.
    #
    # This improves usability on developer systems where CUDA is
    # installed conventionally but not explicitly exported into
    # the current CMake environment.
    # --------------------------------------------------------
    if (NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
        find_program(ATLAS_NVCC_EXECUTABLE
                NAMES nvcc
                HINTS /usr/bin /usr/local/cuda/bin
        )

        if (ATLAS_NVCC_EXECUTABLE)
            set(CMAKE_CUDA_COMPILER "${ATLAS_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
            message(STATUS "[ATLAS] Using nvcc: ${CMAKE_CUDA_COMPILER}")
        endif ()
    endif ()

    # --------------------------------------------------------
    # NVIDIA environment diagnostics
    #
    # Query driver version, driver-reported CUDA runtime version,
    # and native GPU architecture information before enabling the
    # CUDA language.
    #
    # These values are used to:
    #   - improve configuration diagnostics
    #   - auto-select a better architecture default
    #   - warn about toolkit/driver version mismatches
    # --------------------------------------------------------
    atlas_query_nvidia_smi(ATLAS_CUDA_DRIVER_VERSION "driver_version")
    atlas_detect_cuda_driver_runtime_version(ATLAS_CUDA_DRIVER_RUNTIME_VERSION)
    atlas_detect_cuda_architectures(ATLAS_DETECTED_CUDA_ARCHITECTURES)

    if (ATLAS_CUDA_DRIVER_VERSION)
        message(STATUS "[ATLAS] NVIDIA driver version: ${ATLAS_CUDA_DRIVER_VERSION}")
    endif ()

    if (ATLAS_CUDA_DRIVER_RUNTIME_VERSION)
        message(STATUS "[ATLAS] NVIDIA driver CUDA runtime: ${ATLAS_CUDA_DRIVER_RUNTIME_VERSION}")
    endif ()

    # Enable CUDA as a first-class language after the compiler has
    # been selected and environment diagnostics have been gathered.
    enable_language(CUDA)

    # Apply the CUDA language standard globally so CUDA-enabled targets do not
    # need to restate the same target properties or per-source standard flags
    # in each example directory.
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # --------------------------------------------------------
    # CUDA architecture selection
    #
    # If the user did not provide a meaningful architecture list,
    # choose one automatically.
    #
    # Selection policy:
    #   1. use the installed GPU architecture when detected
    #   2. otherwise fall back to 89-real
    #
    # The use of a real architecture prefers native SASS output and
    # reduces the likelihood of PTX JIT issues on systems where the
    # driver runtime may lag behind the toolkit.
    #
    # The special "52" fallback check is included because some CUDA
    # toolchain setups may inject that legacy default implicitly.
    # --------------------------------------------------------
    if ((NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            OR CMAKE_CUDA_ARCHITECTURES STREQUAL ""
            OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
        if (ATLAS_DETECTED_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "${ATLAS_DETECTED_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
        else ()
            set(CMAKE_CUDA_ARCHITECTURES "89-real" CACHE STRING "CUDA architectures" FORCE)
        endif ()
    endif ()

    if (CMAKE_CUDA_COMPILER_VERSION)
        message(STATUS "[ATLAS] CUDA toolkit version: ${CMAKE_CUDA_COMPILER_VERSION}")
    endif ()

    # --------------------------------------------------------
    # Toolkit vs. driver compatibility warning
    #
    # When the installed CUDA toolkit is newer than the CUDA runtime
    # level reported by the driver, native compilation may still
    # succeed for the selected real architecture, but PTX-based JIT
    # fallback may fail at runtime on certain deployment paths.
    #
    # This warning is intentionally non-fatal because many practical
    # setups still function correctly when native code is emitted.
    # --------------------------------------------------------
    if (CMAKE_CUDA_COMPILER_VERSION
            AND ATLAS_CUDA_DRIVER_RUNTIME_VERSION
            AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER ATLAS_CUDA_DRIVER_RUNTIME_VERSION)
        message(WARNING
                "[ATLAS] CUDA toolkit ${CMAKE_CUDA_COMPILER_VERSION} is newer than "
                "driver CUDA runtime ${ATLAS_CUDA_DRIVER_RUNTIME_VERSION}. "
                "Native SASS builds may work, but PTX JIT can fail at runtime."
        )
    endif ()

    # Export compile definitions so public headers can switch
    # behavior according to the selected tasking implementation.
    target_compile_definitions(atlas-core
            INTERFACE
            ATLAS_TASKING_CUDA
            THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
    )
    target_compile_definitions(atlas-serialization
            PRIVATE
            ATLAS_TASKING_CUDA
            THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
    )

    # TBB remains required on the CUDA path because Thrust is
    # configured to use TBB for host-side parallel execution.
    target_link_libraries(atlas-core INTERFACE $<LINK_ONLY:TBB::tbb>)
    target_link_libraries(atlas-serialization PRIVATE TBB::tbb)

    # Append CUDA-specific flags only when they are not already
    # present. This avoids duplicated flags during repeated CMake
    # reconfiguration passes.
    foreach (atlas_cuda_flag IN ITEMS --expt-relaxed-constexpr --extended-lambda)
        if (NOT CMAKE_CUDA_FLAGS MATCHES "(^| )${atlas_cuda_flag}($| )")
            string(APPEND CMAKE_CUDA_FLAGS " ${atlas_cuda_flag}")
        endif ()
    endforeach ()

    # CUDAToolkit is required to expose imported runtime targets
    # such as CUDA::cudart for downstream linkage.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(atlas-core INTERFACE CUDA::cudart)
    target_link_libraries(atlas-serialization PRIVATE CUDA::cudart)

else ()
    message(STATUS "[ATLAS] Tasking backend: TBB")

    # Export compile definitions so public headers can select the
    # CPU/TBB execution path and configure Thrust accordingly.
    target_compile_definitions(atlas-core
            INTERFACE
            ATLAS_TASKING_TBB
            THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    )
    target_compile_definitions(atlas-serialization
            PRIVATE
            ATLAS_TASKING_TBB
            THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    )

    target_link_libraries(atlas-core INTERFACE TBB::tbb)
    target_link_libraries(atlas-serialization PRIVATE TBB::tbb)
endif ()

# ============================================================
# atlas-logging
#
# Logging support is provided as a compiled static library rather
# than a header-only feature. When enabled:
#
#   - atlas-logging is built from its implementation source
#   - atlas-core exports ATLAS_ENABLE_LOGGING to consumers
#   - atlas-core also propagates the atlas::logging dependency
#
# This design allows downstream users to gain logging support by
# linking atlas::core alone, without having to remember an extra
# logging target manually.
# ============================================================
if (ATLAS_LOGGING)
    message(STATUS "[ATLAS] Logging: ENABLED")

    add_library(atlas-logging STATIC
            "${CMAKE_CURRENT_SOURCE_DIR}/src/logging/logging.cpp"
    )
    add_library(atlas::logging ALIAS atlas-logging)

    target_include_directories(atlas-logging
            PUBLIC
            "${ATLAS_CORE_INCLUDE_DIR}"
    )

    target_compile_definitions(atlas-core INTERFACE ATLAS_ENABLE_LOGGING)
    target_compile_definitions(atlas-logging PUBLIC ATLAS_ENABLE_LOGGING)

    # Propagate the compiled logging implementation through the
    # main interface target so that atlas-core consumers link it
    # automatically when logging is enabled.
    target_link_libraries(atlas-core INTERFACE atlas::logging)

else ()
    message(STATUS "[ATLAS] Logging: DISABLED")
endif ()

# ============================================================
# Optional Vizkit / OpenGL visualization stack
#
# Visualization support is modeled as an optional transitive
# feature of atlas-core. When enabled, atlas-core exports:
#
#   - the ATLAS_ENABLE_VIZKIT feature macro
#   - additional include paths for visualization-related headers
#   - all OpenGL/windowing dependencies required by consumers
#
# This keeps user-facing linkage simple at the expense of a larger
# transitive dependency surface for visualization-enabled builds.
# ============================================================
if (ATLAS_USE_VIZKIT)
    message(STATUS "[ATLAS] Vizkit visualization: ENABLED")

    target_compile_definitions(atlas-core INTERFACE ATLAS_ENABLE_VIZKIT)
    target_include_directories(atlas-core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)

    set(ATLAS_VIZKIT_OPENGL_TARGET OpenGL::GL)
    if (TARGET OpenGL::OpenGL)
        set(ATLAS_VIZKIT_OPENGL_TARGET OpenGL::OpenGL)
    endif ()

    target_link_libraries(atlas-core
            INTERFACE
            glfw
            GLEW::GLEW
            ${ATLAS_VIZKIT_OPENGL_TARGET}
    )
else ()
    message(STATUS "[ATLAS] Vizkit visualization: DISABLED")
endif ()

# ============================================================
# Tests
#
# Test configuration is isolated under a single top-level guard
# controlled by ATLAS_ENABLE_TESTS.
#
# Shared structure:
#   - testkit is an interface target containing common headers and
#     transitive dependencies needed by multiple test families
#
# Supported test families:
#   - GoogleTest-based C++ tests (*.cpp)
#   - CUDA-specific tests (*.cu)
#
# No test targets are generated when BUILD_TESTING is OFF, even if
# individual test options were requested by the user.
# ============================================================
if (ATLAS_ENABLE_TESTS)

    file(GLOB_RECURSE TESTKIT_PUBLIC_HEADERS CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/testkit/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/testkit/internal/*.h"
    )

    add_library(testkit INTERFACE)
    target_sources(testkit INTERFACE ${TESTKIT_PUBLIC_HEADERS})
    target_include_directories(testkit INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

    # CUDA runtime is linked into testkit only when the imported
    # cudart target exists. This keeps the interface portable across
    # non-CUDA and CUDA-enabled configurations.
    if (TARGET CUDA::cudart)
        target_link_libraries(testkit INTERFACE CUDA::cudart)
    endif ()

    # ========================================================
    # GoogleTest-based unit tests
    #
    # This test family discovers all C++ test sources under tests/.
    # When sources exist, two kinds of executables are created:
    #
    #   1. atlas_tests
    #      An aggregate executable containing all .cpp tests.
    #
    #   2. atlas_tests_<directory>
    #      One executable per test directory for finer-grained runs.
    #
    # This dual structure is useful when developers want both:
    #   - a single all-in-one unit test binary
    #   - smaller scoped binaries for focused iteration
    # ========================================================
    if (ATLAS_GOOGLE_TEST)
        add_subdirectory(external/googletest)
        include(GoogleTest)

        file(GLOB_RECURSE ATLAS_TEST_HEADERS CONFIGURE_DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h"
        )

        file(GLOB_RECURSE ATLAS_TEST_SOURCES CONFIGURE_DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        )

        if (ATLAS_TEST_SOURCES STREQUAL "")
            message(WARNING "[ATLAS][TEST] No GoogleTest source files found.")
        else ()

            # ------------------------------------------------
            # atlas_configure_gtest_target(target_name)
            #
            # Apply the shared configuration required by every
            # GoogleTest-based executable:
            #
            #   - link atlas::core and testkit
            #   - enable TESTKIT_USE_GTEST
            #   - link the GoogleTest main libraries
            #   - link pthreads on Linux when required
            #   - register tests through gtest_discover_tests
            #
            # This helper prevents configuration duplication
            # across aggregate and per-directory test targets.
            # ------------------------------------------------
            function(atlas_configure_gtest_target target_name)
                target_link_libraries(${target_name}
                        PRIVATE
                        atlas::core
                        testkit
                )

                target_compile_definitions(${target_name}
                        PRIVATE
                        TESTKIT_USE_GTEST
                )

                if (TARGET GTest::gtest AND TARGET GTest::gtest_main)
                    target_link_libraries(${target_name}
                            PRIVATE
                            GTest::gtest
                            GTest::gtest_main
                    )
                elseif (TARGET gtest AND TARGET gtest_main)
                    target_link_libraries(${target_name}
                            PRIVATE
                            gtest
                            gtest_main
                    )
                else ()
                    find_package(GTest REQUIRED)
                    target_link_libraries(${target_name}
                            PRIVATE
                            GTest::gtest
                            GTest::gtest_main
                    )
                endif ()

                if (UNIX AND NOT APPLE)
                    find_package(Threads REQUIRED)
                    target_link_libraries(${target_name} PRIVATE Threads::Threads)
                endif ()

                gtest_discover_tests(${target_name}
                        TEST_PREFIX "${target_name}."
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                )
            endfunction()

            # Aggregate executable containing every discovered
            # GoogleTest-compatible .cpp test source.
            add_executable(atlas_tests
                    ${ATLAS_TEST_HEADERS}
                    ${ATLAS_TEST_SOURCES}
            )
            atlas_configure_gtest_target(atlas_tests)

            # Build one additional executable per directory so that
            # developers can run narrower test groups more easily.
            set(ATLAS_TEST_DIRECTORIES)
            foreach (test_source IN LISTS ATLAS_TEST_SOURCES)
                get_filename_component(test_directory "${test_source}" DIRECTORY)
                list(APPEND ATLAS_TEST_DIRECTORIES "${test_directory}")
            endforeach ()

            list(REMOVE_DUPLICATES ATLAS_TEST_DIRECTORIES)
            list(SORT ATLAS_TEST_DIRECTORIES)

            foreach (test_directory IN LISTS ATLAS_TEST_DIRECTORIES)
                file(GLOB ATLAS_DIRECTORY_TEST_SOURCES CONFIGURE_DEPENDS
                        "${test_directory}/*.cpp"
                )

                if (ATLAS_DIRECTORY_TEST_SOURCES STREQUAL "")
                    continue()
                endif ()

                file(RELATIVE_PATH ATLAS_DIRECTORY_TEST_NAME
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
                        "${test_directory}"
                )

                # Convert the relative directory path into a valid
                # and stable CMake target identifier.
                string(REPLACE "/" "_" ATLAS_DIRECTORY_TEST_NAME "${ATLAS_DIRECTORY_TEST_NAME}")
                string(MAKE_C_IDENTIFIER "${ATLAS_DIRECTORY_TEST_NAME}" ATLAS_DIRECTORY_TEST_NAME)

                set(ATLAS_DIRECTORY_TEST_TARGET "atlas_tests_${ATLAS_DIRECTORY_TEST_NAME}")

                add_executable(${ATLAS_DIRECTORY_TEST_TARGET}
                        ${ATLAS_TEST_HEADERS}
                        ${ATLAS_DIRECTORY_TEST_SOURCES}
                )

                atlas_configure_gtest_target(${ATLAS_DIRECTORY_TEST_TARGET})
            endforeach ()
        endif ()
    endif ()

    # ========================================================
    # CUDA-specific tests
    #
    # All discovered CUDA test sources are linked into a single
    # executable driven by tests/cuda/main.cu.
    #
    # Some CUDA test sources are explicitly excluded because they
    # are currently unsupported, unstable, redundant, or otherwise
    # not intended to participate in the unified CUDA test binary.
    #
    # This section therefore performs:
    #   - recursive discovery of .cu test sources
    #   - explicit removal of the main entry source from the list
    #   - explicit removal of excluded CUDA test sources
    #   - creation of the atlas_all_cuda_test executable
    #   - CTest registration of the final test binary
    # ========================================================
    if (ATLAS_CUDA_TEST)
        file(GLOB_RECURSE ATLAS_CUDA_TEST_SOURCES CONFIGURE_DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
        )

        if (ATLAS_CUDA_TEST_SOURCES STREQUAL "")
            message(WARNING "[ATLAS][TEST] No CUDA test source files found.")
        else ()
            set(ATLAS_CUDA_TEST_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/main.cu")
            set(ATLAS_CUDA_EXCLUDED_SOURCES
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/remove/remove_tests.cu"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/solver/dsmc/dsmc_disjoint_pair_solver_tests.cu"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/solver/dsmc/dsmc_ntc_solver_tests.cu"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/transform/transform_reduce_tests.cu"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/transform/transform_tests.cu"
            )

            # Remove the unified CUDA test entry point from the list
            # of auxiliary sources because it is added explicitly as
            # the executable's first compilation unit.
            list(REMOVE_ITEM ATLAS_CUDA_TEST_SOURCES "${ATLAS_CUDA_TEST_MAIN}")

            # Remove sources that are intentionally excluded from the
            # aggregate CUDA test executable.
            list(REMOVE_ITEM ATLAS_CUDA_TEST_SOURCES ${ATLAS_CUDA_EXCLUDED_SOURCES})

            add_executable(atlas_all_cuda_test
                    ${ATLAS_CUDA_TEST_MAIN}
                    ${ATLAS_CUDA_TEST_SOURCES}
            )

            # The CUDA target itself is configured with CUDA 17, while
            # non-main CUDA sources additionally receive a C++20 flag.
            # This mixed setup reflects the current project/toolchain
            # expectations and should be revisited if the CUDA dialect
            # policy is unified later.
            set_target_properties(atlas_all_cuda_test PROPERTIES
                    CUDA_STANDARD 17
                    CUDA_STANDARD_REQUIRED ON
            )

            set_source_files_properties(${ATLAS_CUDA_TEST_SOURCES} PROPERTIES
                    COMPILE_OPTIONS "--std=c++20"
            )

            target_link_libraries(atlas_all_cuda_test
                    PRIVATE
                    atlas::core
                    testkit
            )

            # Register the executable as a CTest test and also expose
            # a convenience custom target for building the binary.
            add_test(NAME atlas_all_cuda_test COMMAND atlas_all_cuda_test)

            add_custom_target(atlas_cuda_test_all
                    DEPENDS atlas_all_cuda_test
            )
        endif ()
    endif ()

elseif (ATLAS_GOOGLE_TEST OR ATLAS_CUDA_TEST)
    # This message helps explain why explicit test options did not
    # produce targets when CTest support itself is disabled.
    message(STATUS "[ATLAS] Tests requested but BUILD_TESTING=OFF. Skipping test targets.")
endif ()

# ============================================================
# Benchmarks
#
# Benchmark targets are added only when explicitly enabled and not
# disabled earlier by compatibility validation.
#
# The benchmark framework itself is vendored through the source tree
# and benchmark definitions are expected to live under benchmarks/.
# ============================================================
if (ATLAS_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Google Benchmark self tests" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable Google Benchmark GoogleTest tests" FORCE)
    add_subdirectory(external/googlebenchmark)
    add_subdirectory(benchmarks)
endif ()

# ============================================================
# Example applications
#
# These subdirectories provide reference or demonstration programs
# for major atlas-engine-dev usage scenarios.
#
# Keeping them in the default build helps ensure that public APIs
# remain exercised by realistic consumer-style targets.
# ============================================================
add_subdirectory(examples)

# ============================================================
# Configuration summary
#
# Print the final resolved configuration so that developers can
# quickly verify:
#
#   - selected backend
#   - enabled optional features
#   - enabled test/benchmark modes
#   - resolved CUDA architecture when relevant
#
# This summary is particularly useful when options were modified
# automatically due to compatibility rules or environment detection.
# ============================================================
message(STATUS "")
message(STATUS "========== ATLAS CONFIG SUMMARY ==========")
message(STATUS "C++ Standard        : ${CMAKE_CXX_STANDARD}")
message(STATUS "ATLAS_USE_CUDA      : ${ATLAS_USE_CUDA}")
message(STATUS "ATLAS_USE_TBB       : ${ATLAS_USE_TBB}")
message(STATUS "ATLAS_USE_VIZKIT    : ${ATLAS_USE_VIZKIT}")
message(STATUS "ATLAS_LOGGING       : ${ATLAS_LOGGING}")
message(STATUS "ATLAS_GOOGLE_TEST   : ${ATLAS_GOOGLE_TEST}")
message(STATUS "ATLAS_CUDA_TEST     : ${ATLAS_CUDA_TEST}")
message(STATUS "ATLAS_BENCHMARKS    : ${ATLAS_BENCHMARKS}")
if (ATLAS_USE_CUDA)
    message(STATUS "CUDA Architectures  : ${CMAKE_CUDA_ARCHITECTURES}")
endif ()
message(STATUS "==========================================")
message(STATUS "")
