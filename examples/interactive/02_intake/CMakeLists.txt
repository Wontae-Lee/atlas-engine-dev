set(example_target atlas_example_interactive_02_intake)
add_executable(${example_target} main.cpp)
target_link_libraries(${example_target} PRIVATE atlas::interactive-rendering)
foreach (case RANGE 1 2)
    configure_file(config_${case}.json config_${case}.json COPYONLY)
endforeach ()

set(intake_assets support.obj support.mtl)
foreach (wing RANGE 1 12)
    list(APPEND intake_assets wing${wing}.obj wing${wing}.mtl)
endforeach ()
foreach (asset IN LISTS intake_assets)
    configure_file("${PROJECT_SOURCE_DIR}/assets/intake/${asset}"
            "assets/${asset}" COPYONLY)
    install(FILES "${PROJECT_SOURCE_DIR}/assets/intake/${asset}"
            DESTINATION examples/interactive/02_intake/assets COMPONENT interactive)
endforeach ()

add_dependencies(atlas_examples_interactive ${example_target})
install(TARGETS ${example_target} RUNTIME DESTINATION bin COMPONENT interactive)
install(FILES config_1.json config_2.json DESTINATION examples/interactive/02_intake COMPONENT interactive)
