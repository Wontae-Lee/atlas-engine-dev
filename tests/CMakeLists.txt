file(GLOB_RECURSE ATLAS_TEST_HEADERS
        CONFIGURE_DEPENDS
        "*.h"
)

file(GLOB_RECURSE ATLAS_TEST_SOURCES
        CONFIGURE_DEPENDS
        "*.cpp"
)

if (ATLAS_USE_CUDA)
    list(FILTER ATLAS_TEST_SOURCES EXCLUDE REGEX ".*/sampling/sampling_tests\\.cpp$")
    list(FILTER ATLAS_TEST_SOURCES EXCLUDE REGEX ".*/shuffle/shuffle_operator_tests\\.cpp$")
    list(FILTER ATLAS_TEST_SOURCES EXCLUDE REGEX ".*/sink/sink_tests\\.cpp$")
    list(FILTER ATLAS_TEST_SOURCES EXCLUDE REGEX ".*/source/source_tests\\.cpp$")
    list(FILTER ATLAS_TEST_SOURCES EXCLUDE REGEX ".*/source/spawn_operator_tests/surface_spawn_operator_tests\\.cpp$")
    list(FILTER ATLAS_TEST_SOURCES EXCLUDE REGEX ".*/source/spawn_operator_tests/volume_spawn_operator_tests\\.cpp$")
endif ()

if (ATLAS_TEST_SOURCES STREQUAL "")
    message(WARNING "[ATLAS][TEST] No test source files found.")
    return()
endif ()

add_executable(atlas_tests
        ${ATLAS_TEST_HEADERS}
        ${ATLAS_TEST_SOURCES}
)

if (ATLAS_USE_CUDA)
    set_source_files_properties(${ATLAS_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
    set_target_properties(atlas_tests PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED OFF
    )
endif ()

# atlas core
target_link_libraries(atlas_tests
        PRIVATE
        atlas::core
)

# GoogleTest 연동
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

# pthread (Linux)
if (UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(atlas_tests PRIVATE Threads::Threads)
endif ()

# CTest
include(GoogleTest)
gtest_discover_tests(atlas_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
