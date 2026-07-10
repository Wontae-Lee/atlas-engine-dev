# ============================================================
# Benchmark configuration for atlas-engine-dev
#
# Entered unconditionally from the top-level CMakeLists.txt; creates targets
# only when ATLAS_BENCHMARKS is enabled.
#
# The benchmarks were removed with the engine restructuring: every case drove
# the old MaterialProperties / SphSolver / measurer API, and the four external
# reference projects (piclas, sparta, splishsplash, dumux) were wired through
# ExternalProject_Add against directories that are not registered in
# .gitmodules. Both will be written again against the current System.
# ============================================================
if (NOT ATLAS_BENCHMARKS)
    return()
endif ()

message(STATUS "[ATLAS] Benchmarks: enabled, but no benchmark targets are defined yet.")
