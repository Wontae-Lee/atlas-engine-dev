# ============================================================
# Atlas C++ examples
#
# Each case directory holds a `main.cu` — a standalone simulation driven straight
# through the Atlas API. It follows the engine's toolchain (compiled as CUDA when
# nvcc builds the engine, as plain C++ otherwise), so the .cu suffix names the
# entry point, not the compiler.
# ============================================================

# ------------------------------------------------------------
# atlas_add_example(case_name relative_dir)
#
# Adds `atlas_example_<case_name>` from relative_dir/main.cu, links the engine,
# and appends it to the aggregate atlas_examples target.
# ------------------------------------------------------------
function(atlas_add_example case_name relative_dir)
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/${relative_dir}/main.cu")

    if (NOT EXISTS "${source}")
        message(WARNING "[ATLAS][EXAMPLE] No main.cu for case '${case_name}' at ${source}")
        return()
    endif ()

    set(target "atlas_example_${case_name}")

    # Outside an nvcc build the .cu holds no CUDA-only syntax, so compile it as C++.
    if (NOT ATLAS_USE_NVCC)
        set_source_files_properties("${source}" PROPERTIES LANGUAGE CXX)
    endif ()

    add_executable(${target} "${source}")
    target_link_libraries(${target} PRIVATE atlas::core)

    if (TARGET atlas)
        target_link_libraries(${target} PRIVATE atlas)
    endif ()

    add_dependencies(atlas_examples ${target})
endfunction()

# Aggregate convenience target so the whole set can be built at once.
add_custom_target(atlas_examples)

atlas_add_example(cylinder cylinder)
