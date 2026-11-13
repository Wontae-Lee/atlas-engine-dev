set(example_target atlas_example_cpp_01_cylinder)
set(example_source "${CMAKE_CURRENT_SOURCE_DIR}/main.cu")

if (NOT ATLAS_USE_NVCC)
    set_source_files_properties("${example_source}" PROPERTIES LANGUAGE CXX)
endif ()

add_executable(${example_target} "${example_source}")
target_link_libraries(${example_target} PRIVATE atlas::core)
if (TARGET atlas)
    target_link_libraries(${example_target} PRIVATE atlas)
endif ()

add_dependencies(atlas_examples_cpp ${example_target})
