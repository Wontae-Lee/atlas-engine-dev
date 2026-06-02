# ============================================================
# Benchmark configuration for atlas-engine-dev
#
# This file owns all benchmark-specific build wiring. It is entered
# unconditionally from the top-level CMakeLists.txt, but it creates
# targets only when ATLAS_BENCHMARKS is enabled.
#
# Responsibilities:
#
#   1. exit early when benchmark targets are disabled
#   2. reject benchmark builds for unsupported backend combinations
#   3. configure Google Benchmark without its self-test targets
#   4. configure external reference-code builds used for comparison
#   5. expose smoke-run targets for copied cylinder benchmark inputs
#   6. include Atlas-native benchmark targets under benchmarks/atlas
#
# The external reference projects are intentionally configured through
# ExternalProject so their build settings, languages, and dependencies
# remain isolated from the Atlas target graph.
# ============================================================

# ============================================================
# Benchmark option gate
#
# ATLAS_BENCHMARKS defaults to OFF in the top-level project. When it is
# disabled, no benchmark dependencies, reference-code projects, or smoke
# targets are added to the build graph.
# ============================================================
if (NOT ATLAS_BENCHMARKS)
    return()
endif ()

# ============================================================
# Backend compatibility
#
# Benchmark support is currently TBB-oriented. CUDA builds can still
# configure the main project, but benchmark targets are disabled because
# the vendored reference-code and Atlas benchmark paths are not yet
# validated against the CUDA backend.
#
# The option is forced back into the cache and parent scope so the final
# top-level configuration summary reports the resolved value.
# ============================================================
if (ATLAS_USE_CUDA)
    message(WARNING
            "[ATLAS] ATLAS_BENCHMARKS are not yet compatible with the CUDA backend. "
            "Disabling benchmarks."
    )
    set(ATLAS_BENCHMARKS OFF CACHE BOOL "Build and enable benchmark targets" FORCE)
    set(ATLAS_BENCHMARKS OFF PARENT_SCOPE)
    return()
endif ()

# ============================================================
# Google Benchmark dependency
#
# Atlas provides its own benchmark targets, so Google Benchmark's
# upstream test targets are disabled before adding the vendored project.
#
# The target existence check keeps this file safe if another directory
# has already provided benchmark::benchmark_main.
# ============================================================
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Google Benchmark self tests" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable Google Benchmark GoogleTest tests" FORCE)

include(ExternalProject)

if (NOT TARGET benchmark::benchmark_main)
    add_subdirectory(
            "${PROJECT_SOURCE_DIR}/external/googlebenchmark"
            "${PROJECT_BINARY_DIR}/external/googlebenchmark"
    )
endif ()

# ============================================================
# Shared external-project arguments
#
# The external reference-code builds use their own build directories
# under the corresponding submodule. Passing through CMAKE_BUILD_TYPE
# keeps single-config generators aligned with the selected Atlas preset
# without imposing Atlas-specific target properties on those projects.
# ============================================================
set(ATLAS_EXTERNAL_BENCHMARK_CMAKE_ARGS "")
if (CMAKE_BUILD_TYPE)
    list(APPEND ATLAS_EXTERNAL_BENCHMARK_CMAKE_ARGS
            "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    )
endif ()

# ============================================================
# Benchmark input locations
#
# Copied reference input cases live under benchmarks/atlas so they are
# grouped with Atlas-owned benchmark assets instead of the upstream
# submodules. The smoke inputs are shortened variants used only to
# verify that the external solvers can start and complete the cylinder
# setup in this repository.
# ============================================================
set(ATLAS_CYLINDER_BENCHMARK_DIR
        "${PROJECT_SOURCE_DIR}/benchmarks/atlas/cylinder"
)

set(ATLAS_PICLAS_SOURCE_DIR
        "${CMAKE_CURRENT_LIST_DIR}/piclas"
)
set(ATLAS_PICLAS_BINARY_DIR
        "${CMAKE_CURRENT_LIST_DIR}/piclas/build"
)
set(ATLAS_PICLAS_EXECUTABLE
        "${ATLAS_PICLAS_BINARY_DIR}/bin/piclas"
)

set(ATLAS_SPARTA_SOURCE_DIR
        "${CMAKE_CURRENT_LIST_DIR}/sparta/cmake"
)
set(ATLAS_SPARTA_BINARY_DIR
        "${CMAKE_CURRENT_LIST_DIR}/sparta/build"
)
set(ATLAS_SPARTA_INSTALL_DIR
        "${CMAKE_CURRENT_LIST_DIR}/sparta/install"
)
set(ATLAS_SPARTA_EXECUTABLE
        "${ATLAS_SPARTA_BINARY_DIR}/src/spa_atlas"
)

# ============================================================
# External reference-code builds
#
# PICLas is configured from its submodule root. SPARTA's documented
# CMake entry point is the submodule's cmake/ directory, so its source
# directory is set there.
#
# SPARTA_MACHINE is fixed to "atlas" to make the generated executable
# path predictable for smoke targets. MPI is not disabled; SPARTA and
# PICLas can both use MPI when the system dependencies are available.
# ============================================================
ExternalProject_Add(atlas_benchmark_piclas_external
        SOURCE_DIR "${ATLAS_PICLAS_SOURCE_DIR}"
        BINARY_DIR "${ATLAS_PICLAS_BINARY_DIR}"
        CMAKE_ARGS
        ${ATLAS_EXTERNAL_BENCHMARK_CMAKE_ARGS}
        INSTALL_COMMAND ""
)

ExternalProject_Add(atlas_benchmark_sparta_external
        SOURCE_DIR "${ATLAS_SPARTA_SOURCE_DIR}"
        BINARY_DIR "${ATLAS_SPARTA_BINARY_DIR}"
        CMAKE_ARGS
        ${ATLAS_EXTERNAL_BENCHMARK_CMAKE_ARGS}
        "-DCMAKE_INSTALL_PREFIX=${ATLAS_SPARTA_INSTALL_DIR}"
        "-DSPARTA_MACHINE=atlas"
        "-DSPARTA_ENABLE_TESTING=OFF"
        INSTALL_COMMAND ""
)

# ============================================================
# Aggregate external benchmark build target
#
# This target is attached to ALL so enabling ATLAS_BENCHMARKS causes the
# two external reference projects to be configured and built as part of
# the normal Atlas build. Smoke simulations remain opt-in custom targets
# so ordinary builds do not run simulations as a side effect.
# ============================================================
add_custom_target(atlas_external_benchmarks ALL
        DEPENDS
        atlas_benchmark_piclas_external
        atlas_benchmark_sparta_external
)

# ============================================================
# Cylinder smoke simulations
#
# The smoke targets run one-step/shortened cylinder cases copied from
# the SPARTA and PICLas submodules. They are useful for verifying that
# the external solvers can execute the reference cylinder setup after
# their binaries are built.
#
# Full benchmark inputs are kept next to the smoke variants and can be
# executed manually with the same external binaries when longer runtime
# comparisons are needed.
# ============================================================
add_custom_target(atlas_benchmark_cylinder_sparta_smoke
        COMMAND "${CMAKE_COMMAND}" -E chdir
        "${ATLAS_CYLINDER_BENCHMARK_DIR}/sparta"
        "${ATLAS_SPARTA_EXECUTABLE}"
        -in in.cylinder.smoke
        DEPENDS atlas_benchmark_sparta_external
        COMMENT "Running SPARTA cylinder smoke simulation"
)

add_custom_target(atlas_benchmark_cylinder_piclas_smoke
        COMMAND "${CMAKE_COMMAND}" -E chdir
        "${ATLAS_CYLINDER_BENCHMARK_DIR}/piclas"
        "${ATLAS_PICLAS_EXECUTABLE}"
        parameter.smoke.ini DSMC.ini
        DEPENDS atlas_benchmark_piclas_external
        COMMENT "Running PICLas cylinder smoke simulation"
)

add_custom_target(atlas_benchmark_cylinder_smoke
        DEPENDS
        atlas_benchmark_cylinder_sparta_smoke
        atlas_benchmark_cylinder_piclas_smoke
)

# ============================================================
# Atlas-native benchmark targets
#
# Atlas-owned benchmark source files are discovered under
# benchmarks/atlas. Those targets depend on the external reference
# builds so native benchmark runs can assume the comparison binaries are
# available when ATLAS_BENCHMARKS is enabled.
# ============================================================
add_subdirectory(atlas)

if (TARGET atlas_benchmarks)
    add_dependencies(atlas_benchmarks
            atlas_benchmark_piclas_external
            atlas_benchmark_sparta_external
    )
endif ()
