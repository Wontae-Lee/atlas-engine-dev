# Execution and protocol library -----------------------------------------------

# Execution sources have no graphics dependency.
set(ATLAS_INTERACTIVE_EXECUTION_SOURCES
        application/interactive_application.cpp
        config/system_factory.cpp
        session/session.cpp
        server/server.cpp
        statistics/simulation_statistics.cpp
        output/csv_writer.cpp
        transport/json_codec.cpp
        transport/json_lines_transport.cpp
)

find_package(nlohmann_json CONFIG REQUIRED)

add_library(atlas-interactive STATIC ${ATLAS_INTERACTIVE_EXECUTION_SOURCES})
add_library(atlas::interactive ALIAS atlas-interactive)

target_include_directories(atlas-interactive
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(atlas-interactive PUBLIC atlas nlohmann_json::nlohmann_json)

# JSONL application ------------------------------------------------------------

add_executable(atlas-interactive-app main.cpp)
target_link_libraries(atlas-interactive-app PRIVATE atlas::interactive)
set_target_properties(atlas-interactive-app PROPERTIES OUTPUT_NAME atlas-interactive)

install(
        TARGETS atlas-interactive-app
        RUNTIME DESTINATION bin
        COMPONENT interactive
)

# Headless builds register execution tests and stop before graphics discovery.
if (NOT ATLAS_INTERACTIVE_RENDERING)
    if (ATLAS_ENABLE_TESTS)
        add_subdirectory("${CMAKE_SOURCE_DIR}/tests/interactive" "${CMAKE_BINARY_DIR}/tests/interactive")
    endif ()
    return()
endif ()

# Optional rendering library ---------------------------------------------------

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

set(ATLAS_INTERACTIVE_RENDERING_SOURCES
        application/render_manager.cpp
        rendering/camera.cpp
        rendering/renderer.cpp
        rendering/state/raw_state_provider.cpp
        rendering/layer/particle_layer.cpp
        rendering/layer/geometry_layer.cpp
        rendering/opengl/buffer.cpp
        rendering/opengl/shader.cpp
        rendering/opengl/framebuffer.cpp
        rendering/target/window_target.cpp
        rendering/target/offscreen_target.cpp
        rendering/stream/frame_stream.cpp
)

if (ATLAS_DEVICE_SYSTEM STREQUAL "CUDA")
    # Exactly one backend bridge implements the common StateBridge interface.
    list(APPEND ATLAS_INTERACTIVE_RENDERING_SOURCES rendering/backend/state_bridge.cu)
else ()
    list(APPEND ATLAS_INTERACTIVE_RENDERING_SOURCES rendering/backend/state_bridge.cpp)
endif ()

if (TARGET OpenGL::OpenGL)
    # Prefer the vendor-neutral GLVND target and retain compatibility with legacy FindOpenGL.
    set(ATLAS_INTERACTIVE_OPENGL_TARGET OpenGL::OpenGL)
else ()
    set(ATLAS_INTERACTIVE_OPENGL_TARGET OpenGL::GL)
endif ()

add_library(atlas-interactive-rendering STATIC ${ATLAS_INTERACTIVE_RENDERING_SOURCES})
add_library(atlas::interactive-rendering ALIAS atlas-interactive-rendering)

target_include_directories(atlas-interactive-rendering
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_definitions(atlas-interactive-rendering PRIVATE GLFW_INCLUDE_NONE)

target_link_libraries(atlas-interactive-rendering
        PUBLIC
        atlas::interactive
        glm::glm
        PRIVATE
        ${ATLAS_INTERACTIVE_OPENGL_TARGET}
        GLEW::GLEW
        glfw
)

if (ATLAS_DEVICE_SYSTEM STREQUAL "CUDA")
    set_target_properties(atlas-interactive-rendering PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(atlas-interactive-rendering PRIVATE CUDA::cudart)
endif ()

# Connect rendering to the application before registering Interactive tests.
target_link_libraries(atlas-interactive-app PRIVATE atlas::interactive-rendering)
target_compile_definitions(atlas-interactive-app PRIVATE ATLAS_INTERACTIVE_RENDERING_ENABLED=1)

if (ATLAS_ENABLE_TESTS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/tests/interactive" "${CMAKE_BINARY_DIR}/tests/interactive")
endif ()
