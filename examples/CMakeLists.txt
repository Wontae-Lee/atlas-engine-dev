add_subdirectory(cpp)

add_custom_target(atlas_examples DEPENDS atlas_examples_cpp)

if (ATLAS_INTERACTIVE AND ATLAS_INTERACTIVE_RENDERING)
    add_subdirectory(interactive)
    add_dependencies(atlas_examples atlas_examples_interactive)
endif ()
