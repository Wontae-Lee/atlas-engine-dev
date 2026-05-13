file(GLOB_RECURSE ATLAS_BENCHMARK_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if (ATLAS_BENCHMARK_SOURCES STREQUAL "")
    message(WARNING "[ATLAS][BENCHMARK] No benchmark source files found.")
else ()
    set(ATLAS_BENCHMARK_RUN_TARGETS "")
    set(ATLAS_BENCHMARK_DIRECTORIES "")

    foreach (ATLAS_BENCHMARK_SOURCE IN LISTS ATLAS_BENCHMARK_SOURCES)
        get_filename_component(ATLAS_BENCHMARK_DIRECTORY
                "${ATLAS_BENCHMARK_SOURCE}"
                DIRECTORY
        )

        list(APPEND ATLAS_BENCHMARK_DIRECTORIES
                "${ATLAS_BENCHMARK_DIRECTORY}"
        )
    endforeach ()

    list(REMOVE_DUPLICATES ATLAS_BENCHMARK_DIRECTORIES)

    foreach (ATLAS_BENCHMARK_DIRECTORY IN LISTS ATLAS_BENCHMARK_DIRECTORIES)
        file(GLOB ATLAS_DIRECTORY_BENCHMARK_SOURCES CONFIGURE_DEPENDS
                "${ATLAS_BENCHMARK_DIRECTORY}/*.cpp"
        )

        if (ATLAS_DIRECTORY_BENCHMARK_SOURCES STREQUAL "")
            continue()
        endif ()

        file(RELATIVE_PATH ATLAS_BENCHMARK_RELATIVE_DIRECTORY
                "${CMAKE_CURRENT_SOURCE_DIR}"
                "${ATLAS_BENCHMARK_DIRECTORY}"
        )

        if (ATLAS_BENCHMARK_RELATIVE_DIRECTORY STREQUAL ".")
            set(ATLAS_BENCHMARK_TARGET atlas_benchmark_root)
        else ()
            string(REPLACE "/" "_" ATLAS_BENCHMARK_DIRECTORY_NAME
                    "${ATLAS_BENCHMARK_RELATIVE_DIRECTORY}"
            )
            set(ATLAS_BENCHMARK_TARGET
                    "atlas_benchmark_${ATLAS_BENCHMARK_DIRECTORY_NAME}"
            )
        endif ()

        add_executable(${ATLAS_BENCHMARK_TARGET}
                ${ATLAS_DIRECTORY_BENCHMARK_SOURCES}
        )

        target_link_libraries(${ATLAS_BENCHMARK_TARGET}
                PRIVATE
                atlas::core
                benchmark::benchmark_main
        )

        set(ATLAS_BENCHMARK_RUN_TARGET
                "run_${ATLAS_BENCHMARK_TARGET}"
        )

        add_custom_target(${ATLAS_BENCHMARK_RUN_TARGET}
                COMMAND $<TARGET_FILE:${ATLAS_BENCHMARK_TARGET}>
                DEPENDS ${ATLAS_BENCHMARK_TARGET}
                COMMENT "Running ${ATLAS_BENCHMARK_TARGET}"
                VERBATIM
                USES_TERMINAL
        )

        list(APPEND ATLAS_BENCHMARK_RUN_TARGETS
                ${ATLAS_BENCHMARK_RUN_TARGET}
        )
    endforeach ()

    add_custom_target(atlas_benchmarks
            DEPENDS ${ATLAS_BENCHMARK_RUN_TARGETS}
    )
endif ()
