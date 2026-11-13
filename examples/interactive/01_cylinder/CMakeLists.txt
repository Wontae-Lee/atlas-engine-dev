set(ATLAS_INTERACTIVE_EXAMPLE_NAME 01_cylinder)
set(ATLAS_INTERACTIVE_EXAMPLE_TARGET atlas_example_interactive_${ATLAS_INTERACTIVE_EXAMPLE_NAME})

add_executable(${ATLAS_INTERACTIVE_EXAMPLE_TARGET} main.cpp)
target_link_libraries(${ATLAS_INTERACTIVE_EXAMPLE_TARGET} PRIVATE atlas::interactive-rendering)
configure_file(config.json config.json COPYONLY)
add_dependencies(atlas_examples_interactive ${ATLAS_INTERACTIVE_EXAMPLE_TARGET})

install(TARGETS ${ATLAS_INTERACTIVE_EXAMPLE_TARGET}
        RUNTIME DESTINATION bin COMPONENT interactive)
install(FILES config.json
        DESTINATION examples/interactive/${ATLAS_INTERACTIVE_EXAMPLE_NAME}
        COMPONENT interactive)
