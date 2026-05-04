file(GLOB_RECURSE ATLAS_BENCHMARK_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if (ATLAS_BENCHMARK_SOURCES STREQUAL "")
    message(WARNING "[ATLAS][BENCHMARK] No benchmark source files found.")
else ()
    add_executable(atlas_benchmarks
            ${ATLAS_BENCHMARK_SOURCES}
    )

    target_link_libraries(atlas_benchmarks
            PRIVATE
            atlas::core
            benchmark::benchmark_main
    )

    add_custom_target(atlas_benchmark_all
            DEPENDS atlas_benchmarks
    )
endif ()
