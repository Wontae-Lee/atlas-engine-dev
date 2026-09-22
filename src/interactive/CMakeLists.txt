find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

set(ATLAS_INTERACTIVE_SOURCES
        session/session.cpp
        rendering/camera.cpp
        rendering/renderer.cpp
        rendering/state/raw_state_provider.cpp
        rendering/state/interpolation_state_provider.cpp
        rendering/layer/particle_layer.cpp
        rendering/layer/geometry_layer.cpp
        rendering/opengl/buffer.cpp
        rendering/opengl/shader.cpp
        rendering/opengl/framebuffer.cpp
        rendering/target/window_target.cpp
        rendering/target/offscreen_target.cpp
        transport/server.cpp
        transport/frame_stream.cpp
)

if (ATLAS_DEVICE_SYSTEM STREQUAL "CUDA")
    list(APPEND ATLAS_INTERACTIVE_SOURCES rendering/backend/state_bridge.cu)
else ()
    list(APPEND ATLAS_INTERACTIVE_SOURCES rendering/backend/state_bridge.cpp)
endif ()

if (TARGET OpenGL::OpenGL)
    set(ATLAS_INTERACTIVE_OPENGL_TARGET OpenGL::OpenGL)
else ()
    set(ATLAS_INTERACTIVE_OPENGL_TARGET OpenGL::GL)
endif ()

add_library(atlas-interactive STATIC ${ATLAS_INTERACTIVE_SOURCES})
add_library(atlas::interactive ALIAS atlas-interactive)

target_include_directories(atlas-interactive
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_definitions(atlas-interactive PRIVATE GLFW_INCLUDE_NONE)

target_link_libraries(atlas-interactive
        PUBLIC
        atlas
        glm::glm
        PRIVATE
        ${ATLAS_INTERACTIVE_OPENGL_TARGET}
        GLEW::GLEW
        glfw
)

if (ATLAS_DEVICE_SYSTEM STREQUAL "CUDA")
    set_target_properties(atlas-interactive PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(atlas-interactive PRIVATE CUDA::cudart)
endif ()

add_executable(atlas-interactive-app main.cpp)
target_link_libraries(atlas-interactive-app PRIVATE atlas::interactive)
set_target_properties(atlas-interactive-app PROPERTIES OUTPUT_NAME atlas-interactive)

install(
        TARGETS atlas-interactive-app
        RUNTIME DESTINATION bin
        COMPONENT interactive
)
