
# =====================================================
# Tests (optional)
# =====================================================
if (ATLAS_ENABLE_TESTS)
    message(STATUS "[ATLAS] Tests: ENABLED")
    enable_testing()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}../../external/googletest/CMakeLists.txt")
        add_subdirectory(external/googletest)
    else ()
        find_package(GTest REQUIRED)
    endif ()

    file(GLOB_RECURSE atlas_test_sources
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/object_tests/*.cpp"
    )
    add_executable(atlas_tests ${atlas_test_sources})
    target_link_libraries(atlas_tests PRIVATE atlas::core)

    if (TARGET GTest::gtest AND TARGET GTest::gtest_main)
        target_link_libraries(atlas_tests PRIVATE GTest::gtest GTest::gtest_main)
    elseif (TARGET gtest AND TARGET gtest_main)
        target_link_libraries(atlas_tests PRIVATE gtest gtest_main)
    else ()
        find_package(GTest REQUIRED)
        target_link_libraries(atlas_tests PRIVATE GTest::gtest GTest::gtest_main)
    endif ()

    if (UNIX AND NOT APPLE)
        find_package(Threads REQUIRED)
        target_link_libraries(atlas_tests PRIVATE Threads::Threads)
    endif ()

    include(GoogleTest)
    gtest_discover_tests(atlas_tests WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif ()
