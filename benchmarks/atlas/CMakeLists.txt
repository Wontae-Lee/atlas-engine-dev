# ============================================================
# Atlas-native benchmark cases
#
# Each case directory holds a `main.cpp` — the case wrapped in Google Benchmark.
# (Standalone, framework-free simulations live under examples/cpp/ instead.) It
# follows the engine's toolchain, compiled as CUDA when nvcc builds the engine, so
# the .cpp suffix names the entry point, not the compiler. The assets directory is
# passed in as ATLAS_BENCHMARK_ASSETS_DIR so a case can load a mesh without
# depending on the working directory it is run from.
#
# Only a smoke case is wired for now — the real benchmark cases are to be
# (re)written on Google Benchmark. Add one with atlas_add_benchmark_case(<name> <dir>).
# ============================================================

set(ATLAS_BENCHMARK_ASSETS_DIR "${PROJECT_SOURCE_DIR}/assets")

# ------------------------------------------------------------
# atlas_add_benchmark_case(case_name relative_dir)
#
# Adds `atlas_benchmark_<case_name>_gbench` from relative_dir/main.cpp and appends
# it to the aggregate atlas_benchmarks target.
# ------------------------------------------------------------
function(atlas_add_benchmark_case case_name relative_dir)
    set(gbench_source "${CMAKE_CURRENT_SOURCE_DIR}/${relative_dir}/main.cpp")

    if (EXISTS "${gbench_source}")
        set(target "atlas_benchmark_${case_name}_gbench")

        # Under nvcc the Atlas headers pull in device-dispatch code only nvcc can compile.
        if (ATLAS_USE_NVCC)
            set_source_files_properties("${gbench_source}" PROPERTIES LANGUAGE CUDA)
        endif ()

        add_executable(${target} "${gbench_source}")
        atlas_configure_benchmark_target(${target})

        # The case registers with BENCHMARK(...) and supplies BENCHMARK_MAIN(),
        # so it links the library without Google Benchmark's own main.
        target_link_libraries(${target} PRIVATE benchmark::benchmark)

        add_dependencies(atlas_benchmarks ${target})
    endif ()
endfunction()

# ------------------------------------------------------------
# Shared wiring for every benchmark executable.
# ------------------------------------------------------------
function(atlas_configure_benchmark_target target)
    target_link_libraries(${target} PRIVATE atlas::core)

    if (TARGET atlas)
        target_link_libraries(${target} PRIVATE atlas)
    endif ()

    target_compile_definitions(${target}
            PRIVATE
            ATLAS_BENCHMARK_ASSETS_DIR="${ATLAS_BENCHMARK_ASSETS_DIR}"
    )
endfunction()

# Aggregate convenience target so the whole set can be built at once.
add_custom_target(atlas_benchmarks)

atlas_add_benchmark_case(smoke smoke)
