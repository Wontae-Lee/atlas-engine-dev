add_executable(atlas-interactive-example main.cpp)
target_link_libraries(atlas-interactive-example PRIVATE atlas::interactive-rendering)

# Keep source-tree case lookup available without embedding machine-specific paths in code.
target_compile_definitions(atlas-interactive-example
        PRIVATE
        ATLAS_INTERACTIVE_EXAMPLE_CASES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cases"
)

# Copy cases beside the build-tree executable for direct local invocation.
file(COPY cases DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

install(
        TARGETS atlas-interactive-example
        RUNTIME DESTINATION bin
        COMPONENT interactive
)

install(
        DIRECTORY cases
        DESTINATION examples/interactive
        COMPONENT interactive
)
