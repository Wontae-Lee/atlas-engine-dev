add_executable(atlas-interactive-example native_window.cpp)
target_link_libraries(atlas-interactive-example PRIVATE atlas::interactive-rendering)

install(
        TARGETS atlas-interactive-example
        RUNTIME DESTINATION bin
        COMPONENT interactive
)
