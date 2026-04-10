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
    file(GLOB_RECURSE ATLAS_CUDA_TEST_HEADERS
            CONFIGURE_DEPENDS
            "*.cuh"
    )

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_tests.cu")
        add_library(atlas_cuda_test_lib INTERFACE)

        target_sources(atlas_cuda_test_lib
                INTERFACE
                ${ATLAS_CUDA_TEST_HEADERS}
        )

        target_include_directories(atlas_cuda_test_lib
                INTERFACE
                "${CMAKE_CURRENT_SOURCE_DIR}"
        )

        target_link_libraries(atlas_cuda_test_lib
                INTERFACE
                atlas::core
        )

        add_executable(atlas_cuda_tests
                cuda/cuda_tests.cu
        )

        set_target_properties(atlas_cuda_tests PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED OFF
        )

        target_link_libraries(atlas_cuda_tests
                PRIVATE
                atlas_cuda_test_lib
        )

        add_test(NAME atlas_cuda_tests COMMAND atlas_cuda_tests)
    else ()
        message(WARNING "[ATLAS][TEST] CUDA test source not found: cuda/cuda_tests.cu")
    endif ()
endif ()
