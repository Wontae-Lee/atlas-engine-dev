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
    set(ATLAS_CUDA_TEST_SOURCES
            cuda/cuda_tests.cu
            buffer/device_buffer_tests.cu
            buffer/host_buffer_tests.cu
            unit/unit_tests.cu
            collider/collider_surface_interaction_tests.cu
            collider/collider_tests.cu
            measure/measure_tests.cu
            sampling/sampling_tests.cu
            source/source_tests.cu
            source/spawn_operator_tests/surface_spawn_operator_tests.cu
            source/spawn_operator_tests/volume_spawn_operator_tests.cu
            system/system_tests.cu
            transform/transform_reduce_tests.cu
    )

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_tests.cu")
        add_executable(atlas_cuda_tests
                ${ATLAS_CUDA_TEST_SOURCES}
        )

        set_target_properties(atlas_cuda_tests PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED OFF
        )

        target_include_directories(atlas_cuda_tests
                PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}"
        )

        target_link_libraries(atlas_cuda_tests
                PRIVATE
                atlas::core
        )

        add_test(NAME atlas_cuda_tests COMMAND atlas_cuda_tests)
    else ()
        message(WARNING "[ATLAS][TEST] CUDA test source not found: cuda/cuda_tests.cu")
    endif ()
endif ()
