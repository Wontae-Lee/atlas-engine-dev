# Interactive tests reuse the Core test helper configured by tests/.

include(GoogleTest)

add_executable(atlas_tests_interactive interactive_tests.cpp)
atlas_configure_gtest_target(atlas_tests_interactive)
target_link_libraries(atlas_tests_interactive PRIVATE atlas::interactive)

# Rendering geometry tests are available only with the rendering library.
if (TARGET atlas::interactive-rendering)
    add_executable(atlas_tests_interactive_rendering rendering_tests.cpp)
    atlas_configure_gtest_target(atlas_tests_interactive_rendering)
    target_link_libraries(atlas_tests_interactive_rendering PRIVATE atlas::interactive-rendering)
endif ()
