# ============================================================
# Atlas-native benchmark cases
#
# Each case directory holds up to two entry points:
#
#   main.cu  — a standalone simulation driven straight through the Atlas API.
#              It links no benchmark framework and prints its own timings.
#   main.cpp — the same case wrapped in Google Benchmark.
#
# Both are compiled as CUDA (Atlas is compiled exclusively with nvcc), and each
# is optional: a case may ship one, the other, or both. The assets directory is
# passed in as ATLAS_BENCHMARK_ASSETS_DIR so a case can load a mesh without
# depending on the working directory it is run from.
# ============================================================

set(ATLAS_BENCHMARK_ASSETS_DIR "${PROJECT_SOURCE_DIR}/assets")

# ------------------------------------------------------------
# atlas_add_benchmark_case(case_name relative_dir)
#
# Adds `atlas_benchmark_<case_name>` from relative_dir/main.cu when that file
# exists, and `atlas_benchmark_<case_name>_gbench` from relative_dir/main.cpp
# when that one does. Both are appended to the aggregate atlas_benchmarks target.
# ------------------------------------------------------------
function(atlas_add_benchmark_case case_name relative_dir)
    set(standalone_source "${CMAKE_CURRENT_SOURCE_DIR}/${relative_dir}/main.cu")
    set(gbench_source "${CMAKE_CURRENT_SOURCE_DIR}/${relative_dir}/main.cpp")

    if (EXISTS "${standalone_source}")
        set(target "atlas_benchmark_${case_name}")

        add_executable(${target} "${standalone_source}")
        atlas_configure_benchmark_target(${target})

        add_dependencies(atlas_benchmarks ${target})
    endif ()

    if (EXISTS "${gbench_source}")
        set(target "atlas_benchmark_${case_name}_gbench")

        # Atlas headers pull in device-dispatch code that only nvcc can compile.
        set_source_files_properties("${gbench_source}" PROPERTIES LANGUAGE CUDA)

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

atlas_add_benchmark_case(cylinder cylinder)
