/**
 * @file
 * @brief Implements point rendering for live particles.
 */

#include "rendering/layer/particle_layer.h"

#include "rendering/camera.h"
#include "rendering/state/render_state.h"

#include <atlas/math/vector/float3.h>

#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>

#include <stdexcept>
#include <type_traits>

namespace atlas::interactive {

namespace {

/// Minimal world-space point vertex shader.
constexpr const char* vertex_shader = R"(
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 view_projection;
void main() {
    gl_Position = view_projection * vec4(position, 1.0);
}
)";

/// Uniform-color fragment shader for particles.
constexpr const char* fragment_shader = R"(
#version 330 core
out vec4 fragment_color;
uniform vec4 color;
void main() {
    fragment_color = color;
}
)";

}

ParticleLayer::ParticleLayer(const float point_size) noexcept
    : _point_size(point_size) {}

ParticleLayer::~ParticleLayer() = default;

void
ParticleLayer::initialize() {
    if (_point_size <= 0.0f) throw std::invalid_argument("Particle point size must be positive.");
    // The direct vertex upload relies on Float3 matching a tightly packed vec3.
    static_assert(std::is_trivially_copyable_v<Float3>);
    static_assert(sizeof(Float3) == 3 * sizeof(float));

    _shader = opengl::Shader(vertex_shader, fragment_shader);
    _view_projection_location = _shader.uniform_location("view_projection");
    _color_location = _shader.uniform_location("color");
    glGenVertexArrays(1, &_vao);
}

void
ParticleLayer::render(const RenderState& state,
                      const Camera& camera,
                      const float aspect_ratio) {
    if (state.particle_count == 0) return;

    const glm::mat4 matrix = camera.view_projection(aspect_ratio);
    _shader.use();
    glUniformMatrix4fv(_view_projection_location, 1, GL_FALSE, glm::value_ptr(matrix));
    glUniform4f(_color_location, 0.20f, 0.72f, 1.0f, 1.0f);

    glPointSize(_point_size);
    glBindVertexArray(_vao);
    state.position.bind();
    // RenderState owns the VBO; the layer only describes how to interpret it.
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Float3), nullptr);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(state.particle_count));
    glBindVertexArray(0);
    opengl::Buffer::unbind();
}

void
ParticleLayer::shutdown() {
    if (_vao != 0) glDeleteVertexArrays(1, &_vao);
    _vao = 0;
    _view_projection_location = -1;
    _color_location = -1;
    _shader = opengl::Shader();
}

}
