if (ATLAS_GOOGLE_TEST)
    file(GLOB_RECURSE ATLAS_TEST_HEADERS
            CONFIGURE_DEPENDS
            "*.h"
    )

    file(GLOB_RECURSE ATLAS_TEST_SOURCES
            CONFIGURE_DEPENDS
            "*.cpp"
    )

    if (ATLAS_TEST_SOURCES STREQUAL "")
        message(WARNING "[ATLAS][TEST] No GoogleTest source files found.")
        return()
    endif ()

    function(atlas_configure_gtest_target target_name)
        target_link_libraries(${target_name}
                PRIVATE
                atlas::core
        )

        if (TARGET GTest::gtest AND TARGET GTest::gtest_main)
            target_link_libraries(${target_name}
                    PRIVATE
                    GTest::gtest
                    GTest::gtest_main
            )
        elseif (TARGET gtest AND TARGET gtest_main)
            target_link_libraries(${target_name}
                    PRIVATE
                    gtest
                    gtest_main
            )
        else ()
            find_package(GTest REQUIRED)
            target_link_libraries(${target_name}
                    PRIVATE
                    GTest::gtest
                    GTest::gtest_main
            )
        endif ()

        if (UNIX AND NOT APPLE)
            find_package(Threads REQUIRED)
            target_link_libraries(${target_name} PRIVATE Threads::Threads)
        endif ()

        gtest_discover_tests(${target_name}
                TEST_PREFIX "${target_name}."
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endfunction()

    include(GoogleTest)

    add_executable(atlas_tests
            ${ATLAS_TEST_HEADERS}
            ${ATLAS_TEST_SOURCES}
    )

    atlas_configure_gtest_target(atlas_tests)

    set(ATLAS_TEST_DIRECTORIES)
    foreach (test_source IN LISTS ATLAS_TEST_SOURCES)
        get_filename_component(test_directory "${test_source}" DIRECTORY)
        list(APPEND ATLAS_TEST_DIRECTORIES "${test_directory}")
    endforeach ()
    list(REMOVE_DUPLICATES ATLAS_TEST_DIRECTORIES)
    list(SORT ATLAS_TEST_DIRECTORIES)

    foreach (test_directory IN LISTS ATLAS_TEST_DIRECTORIES)
        file(GLOB ATLAS_DIRECTORY_TEST_SOURCES
                CONFIGURE_DEPENDS
                "${test_directory}/*.cpp"
        )

        if (ATLAS_DIRECTORY_TEST_SOURCES STREQUAL "")
            continue()
        endif ()

        file(RELATIVE_PATH ATLAS_DIRECTORY_TEST_NAME
                "${CMAKE_CURRENT_SOURCE_DIR}"
                "${test_directory}"
        )
        string(REPLACE "/" "_" ATLAS_DIRECTORY_TEST_NAME "${ATLAS_DIRECTORY_TEST_NAME}")
        string(MAKE_C_IDENTIFIER "${ATLAS_DIRECTORY_TEST_NAME}" ATLAS_DIRECTORY_TEST_NAME)

        set(ATLAS_DIRECTORY_TEST_TARGET "atlas_tests_${ATLAS_DIRECTORY_TEST_NAME}")

        add_executable(${ATLAS_DIRECTORY_TEST_TARGET}
                ${ATLAS_TEST_HEADERS}
                ${ATLAS_DIRECTORY_TEST_SOURCES}
        )

        atlas_configure_gtest_target(${ATLAS_DIRECTORY_TEST_TARGET})
    endforeach ()
endif ()

if (ATLAS_CUDA_TEST)
    message(WARNING "[ATLAS][TEST] Cuda Tests are not supported yet.")
endif ()
