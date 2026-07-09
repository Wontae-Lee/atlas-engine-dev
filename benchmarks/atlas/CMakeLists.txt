# ============================================================
# Atlas-native benchmark cases
#
# These reference programs were previously shipped under examples/.
# They now live here as benchmark cases so the top-level repository
# stays focused on the engine core. Each case builds into a small
# executable that links atlas::core and drives a representative
# simulation setup.
#
# Every case has a single main.cu entry point: Atlas is compiled
# exclusively with nvcc, so no per-backend source selection is
# needed. This directory is entered only when ATLAS_BENCHMARKS is
# enabled (see benchmarks/CMakeLists.txt).
# ============================================================

# ------------------------------------------------------------
# atlas_add_benchmark_case(target_name relative_dir)
#
# Create a single Google Benchmark executable from the main.cu
# entry point found under relative_dir. Each case registers its
# cases with BENCHMARK(...) and provides main() via BENCHMARK_MAIN(),
# so it links benchmark::benchmark (the library, without its own main).
# ------------------------------------------------------------
function(atlas_add_benchmark_case target_name relative_dir)
    add_executable(${target_name} "${relative_dir}/main.cu")
    target_link_libraries(${target_name} PRIVATE atlas::core benchmark::benchmark)

    if (TARGET atlas)
        target_link_libraries(${target_name} PRIVATE atlas)
    endif ()
endfunction()

# The four cases (cylinder/continum, cylinder/rarefied_gas, inflow, waterfall)
# were removed with the engine restructuring: every one of them drove the old
# MaterialProperties / SphSolver / measurer API. Add new cases here with
# atlas_add_benchmark_case once they are written against the current System.

# Aggregate convenience target so the whole set can be built at once.
add_custom_target(atlas_benchmarks)
