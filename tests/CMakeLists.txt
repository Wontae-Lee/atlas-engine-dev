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

    add_executable(atlas_tests
            ${ATLAS_TEST_HEADERS}
            ${ATLAS_TEST_SOURCES}
    )

    target_link_libraries(atlas_tests
            PRIVATE
            atlas::core
    )

    if (TARGET GTest::gtest AND TARGET GTest::gtest_main)
        target_link_libraries(atlas_tests
                PRIVATE
                GTest::gtest
                GTest::gtest_main
        )
    elseif (TARGET gtest AND TARGET gtest_main)
        target_link_libraries(atlas_tests
                PRIVATE
                gtest
                gtest_main
        )
    else ()
        find_package(GTest REQUIRED)
        target_link_libraries(atlas_tests
                PRIVATE
                GTest::gtest
                GTest::gtest_main
        )
    endif ()

    if (UNIX AND NOT APPLE)
        find_package(Threads REQUIRED)
        target_link_libraries(atlas_tests PRIVATE Threads::Threads)
    endif ()

    include(GoogleTest)
    gtest_discover_tests(atlas_tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif ()

if (ATLAS_CUDA_TEST)
    message(WARNING "[ATLAS][TEST] Cuda Tests are not supported yet.")
endif ()
