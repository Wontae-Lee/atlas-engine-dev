# ============================================================
# Benchmark configuration for atlas-engine-dev
#
# Entered from the top-level CMakeLists.txt when ATLAS_BENCHMARKS is enabled.
#
# The four external reference projects (piclas, sparta, splishsplash, dumux)
# were wired through ExternalProject_Add against directories that are not
# registered in .gitmodules. They were removed with the engine restructuring and
# will be written again.
# ============================================================
# Google Benchmark backs the main.cpp cases. Atlas owns its benchmark targets, so
# the upstream self-test targets stay off. The target check keeps this file safe
# if another directory already provided benchmark::benchmark.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Google Benchmark self tests" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable Google Benchmark GoogleTest tests" FORCE)

if (NOT TARGET benchmark::benchmark)
    add_subdirectory(
            "${PROJECT_SOURCE_DIR}/external/googlebenchmark"
            "${PROJECT_BINARY_DIR}/external/googlebenchmark"
    )
endif ()

add_subdirectory(atlas)
