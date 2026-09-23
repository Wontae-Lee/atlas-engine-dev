set(ATLAS_CPP_EXAMPLE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/main.cu")

# The shared .cu entry point is ordinary C++ when nvcc is not the selected compiler.
if (NOT ATLAS_USE_NVCC)
    set_source_files_properties("${ATLAS_CPP_EXAMPLE_SOURCE}" PROPERTIES LANGUAGE CXX)
endif ()

add_executable(atlas_example "${ATLAS_CPP_EXAMPLE_SOURCE}")
target_link_libraries(atlas_example PRIVATE atlas::core)

# Shared builds expose the aggregate target in addition to atlas::core.
if (TARGET atlas)
    target_link_libraries(atlas_example PRIVATE atlas)
endif ()

add_custom_target(atlas_examples DEPENDS atlas_example)
