# Benchmark cases use the engine toolchain and resolve assets independently of
# the working directory. Standalone simulations live in examples/cpp/.

set(ATLAS_BENCHMARK_ASSETS_DIR "${PROJECT_SOURCE_DIR}/assets")

# Shared engine linkage and asset location for every benchmark.
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

# Register a case and attach it to the aggregate build target.
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

# Case registration -----------------------------------------------------------
add_custom_target(atlas_benchmarks)

atlas_add_benchmark_case(smoke smoke)
