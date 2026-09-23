/**
 * @file
 * @brief Implements geometry tessellation and line rendering.
 */

#include "rendering/layer/geometry_layer.h"

#include "rendering/camera.h"
#include "rendering/state/render_state.h"

#include <atlas/math/math.h>

#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

namespace atlas::interactive {

namespace {

constexpr int segments = 32; ///< Fixed wireframe resolution for curved primitives.
constexpr float pi = 3.14159265358979323846f; ///< Single-precision circle constant.

constexpr const char* vertex_shader = R"(
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 view_projection;
void main() {
    gl_Position = view_projection * vec4(position, 1.0);
}
)";

constexpr const char* fragment_shader = R"(
#version 330 core
out vec4 fragment_color;
uniform vec4 color;
void main() {
    fragment_color = color;
}
)";

/// Converts transport-friendly vector storage to Atlas math storage.
Float3
point(const SimulationConfig::Vec3& value) {
    return Float3(value[0], value[1], value[2]);
}

/// Appends one line segment as two GL_LINES vertices.
void
line(std::vector<Float3>& vertices, const Float3& a, const Float3& b) {
    vertices.push_back(a);
    vertices.push_back(b);
}

/// Transforms and appends a local-space line segment.
void
world_line(std::vector<Float3>& vertices,
           const Sync& sync,
           const Float3& a,
           const Float3& b) {
    line(vertices, sync.sync_to_world(a), sync.sync_to_world(b));
}

/// Builds a stable tangent basis around a possibly degenerate normal.
std::pair<Float3, Float3>
basis(const Float3& normal) {
    const Float3 n = atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));
    const Float3 reference = std::abs(n.z) < 0.9f
        ? Float3(0.0f, 0.0f, 1.0f)
        : Float3(0.0f, 1.0f, 0.0f);
    const Float3 u = atlas::normalized_or(atlas::cross(reference, n),
                                          Float3(1.0f, 0.0f, 0.0f));
    return { u, atlas::cross(n, u) };
}

/// Appends a segmented local-space circle transformed into world space.
void
circle(std::vector<Float3>& vertices,
       const Sync& sync,
       const Float3& center,
       const Float3& normal,
       const float radius) {
    const auto [u, v] = basis(normal);
    for (int index = 0; index < segments; ++index) {
        const float a = 2.0f * pi * static_cast<float>(index) / segments;
        const float b = 2.0f * pi * static_cast<float>(index + 1) / segments;
        world_line(vertices,
                   sync,
                   center + radius * (std::cos(a) * u + std::sin(a) * v),
                   center + radius * (std::cos(b) * u + std::sin(b) * v));
    }
}

/// Appends the twelve edges of an axis-aligned local-space box.
void
box(std::vector<Float3>& vertices,
    const Sync& sync,
    const Float3& lower,
    const Float3& upper) {
    const std::array<Float3, 8> corners = {
        Float3(lower.x, lower.y, lower.z), Float3(upper.x, lower.y, lower.z),
        Float3(upper.x, upper.y, lower.z), Float3(lower.x, upper.y, lower.z),
        Float3(lower.x, lower.y, upper.z), Float3(upper.x, lower.y, upper.z),
        Float3(upper.x, upper.y, upper.z), Float3(lower.x, upper.y, upper.z)
    };
    constexpr std::array<std::array<int, 2>, 12> edges = {
        std::array { 0, 1 }, std::array { 1, 2 }, std::array { 2, 3 }, std::array { 3, 0 },
        std::array { 4, 5 }, std::array { 5, 6 }, std::array { 6, 7 }, std::array { 7, 4 },
        std::array { 0, 4 }, std::array { 1, 5 }, std::array { 2, 6 }, std::array { 3, 7 }
    };
    for (const auto& edge : edges) world_line(vertices, sync, corners[edge[0]], corners[edge[1]]);
}

/// Appends wireframe vertices for the concrete configured geometry kind.
void
append_geometry_vertices(std::vector<Float3>& vertices,
                         const GeometryRenderView& view,
                         const Float3& domain_lower,
                         const Float3& domain_upper) {
    if (view.geometry == nullptr) return;
    const auto& geometry = *view.geometry;
    const Sync& sync = view.sync;
    switch (geometry.kind) {
    case SimulationConfig::GeometryKind::box:
        box(vertices, sync, point(geometry.lower_corner), point(geometry.upper_corner));
        break;
    case SimulationConfig::GeometryKind::circle:
        circle(vertices, sync, point(geometry.center), point(geometry.normal), geometry.radius);
        break;
    case SimulationConfig::GeometryKind::cylinder: {
        const Float3 center = point(geometry.center);
        const float half = 0.5f * geometry.height;
        circle(vertices, sync, center + Float3(0.0f, 0.0f, half),
               Float3(0.0f, 0.0f, 1.0f), geometry.radius);
        circle(vertices, sync, center - Float3(0.0f, 0.0f, half),
               Float3(0.0f, 0.0f, 1.0f), geometry.radius);
        for (int index = 0; index < segments; index += 4) {
            const float angle = 2.0f * pi * static_cast<float>(index) / segments;
            const Float3 radial(geometry.radius * std::cos(angle),
                                geometry.radius * std::sin(angle), 0.0f);
            world_line(vertices, sync, center + radial - Float3(0.0f, 0.0f, half),
                       center + radial + Float3(0.0f, 0.0f, half));
        }
        break;
    }
    case SimulationConfig::GeometryKind::plane: {
        const Float3 normal = atlas::normalized_or(point(geometry.normal), Float3(0.0f, 0.0f, 1.0f));
        const auto [u, v] = basis(normal);
        // Infinite planes use the domain span to choose a useful display extent.
        const float extent = std::max({ domain_upper.x - domain_lower.x,
                                        domain_upper.y - domain_lower.y,
                                        domain_upper.z - domain_lower.z, 1.0f });
        const Float3 center = geometry.offset * normal;
        const Float3 a = center - extent * u - extent * v;
        const Float3 b = center + extent * u - extent * v;
        const Float3 c = center + extent * u + extent * v;
        const Float3 d = center - extent * u + extent * v;
        world_line(vertices, sync, a, b); world_line(vertices, sync, b, c);
        world_line(vertices, sync, c, d); world_line(vertices, sync, d, a);
        break;
    }
    case SimulationConfig::GeometryKind::sphere: {
        const Float3 center = point(geometry.center);
        circle(vertices, sync, center, Float3(1.0f, 0.0f, 0.0f), geometry.radius);
        circle(vertices, sync, center, Float3(0.0f, 1.0f, 0.0f), geometry.radius);
        circle(vertices, sync, center, Float3(0.0f, 0.0f, 1.0f), geometry.radius);
        break;
    }
    case SimulationConfig::GeometryKind::square: {
        const Float3 center = point(geometry.center);
        const auto [u, v] = basis(point(geometry.normal));
        const float half = 0.5f * geometry.side_length;
        const Float3 a = center - half * u - half * v;
        const Float3 b = center + half * u - half * v;
        const Float3 c = center + half * u + half * v;
        const Float3 d = center - half * u + half * v;
        world_line(vertices, sync, a, b); world_line(vertices, sync, b, c);
        world_line(vertices, sync, c, d); world_line(vertices, sync, d, a);
        break;
    }
    case SimulationConfig::GeometryKind::triangle:
        world_line(vertices, sync, point(geometry.a), point(geometry.b));
        world_line(vertices, sync, point(geometry.b), point(geometry.c));
        world_line(vertices, sync, point(geometry.c), point(geometry.a));
        break;
    case SimulationConfig::GeometryKind::triangle_mesh:
        for (const auto& triangle : geometry.triangles) {
            world_line(vertices, sync, point(triangle[0]), point(triangle[1]));
            world_line(vertices, sync, point(triangle[1]), point(triangle[2]));
            world_line(vertices, sync, point(triangle[2]), point(triangle[0]));
        }
        break;
    case SimulationConfig::GeometryKind::polygonal_prism: {
        const Float3 center = point(geometry.center);
        const float half = 0.5f * geometry.height;
        const int sides = std::max(3, geometry.side_count);
        for (int index = 0; index < sides; ++index) {
            const float a = 2.0f * pi * static_cast<float>(index) / sides;
            const float b = 2.0f * pi * static_cast<float>(index + 1) / sides;
            const Float3 pa = center + Float3(geometry.radius * std::cos(a),
                                               geometry.radius * std::sin(a), 0.0f);
            const Float3 pb = center + Float3(geometry.radius * std::cos(b),
                                               geometry.radius * std::sin(b), 0.0f);
            world_line(vertices, sync, pa - Float3(0.0f, 0.0f, half),
                       pb - Float3(0.0f, 0.0f, half));
            world_line(vertices, sync, pa + Float3(0.0f, 0.0f, half),
                       pb + Float3(0.0f, 0.0f, half));
            world_line(vertices, sync, pa - Float3(0.0f, 0.0f, half),
                       pa + Float3(0.0f, 0.0f, half));
        }
        break;
    }
    }
}

}

GeometryLayer::~GeometryLayer() = default;

std::vector<Float3>
GeometryLayer::make_line_vertices(const GeometryRenderView& view,
                                  const Float3& domain_lower,
                                  const Float3& domain_upper) {
    std::vector<Float3> result;
    append_geometry_vertices(result, view, domain_lower, domain_upper);
    return result;
}

void
GeometryLayer::initialize() {
    // The direct OpenGL upload relies on Float3 matching a tightly packed vec3.
    static_assert(std::is_trivially_copyable_v<Float3>);
    static_assert(sizeof(Float3) == 3 * sizeof(float));
    _shader = opengl::Shader(vertex_shader, fragment_shader);
    _view_projection_location = _shader.uniform_location("view_projection");
    _color_location = _shader.uniform_location("color");
    glGenVertexArrays(1, &_vao);
}

void
GeometryLayer::render(const RenderState& state,
                      const Camera& camera,
                      const float aspect_ratio) {
    const glm::mat4 matrix = camera.view_projection(aspect_ratio);
    _shader.use();
    glUniformMatrix4fv(_view_projection_location, 1, GL_FALSE, glm::value_ptr(matrix));
    glBindVertexArray(_vao);
    for (const GeometryRenderView& geometry : state.geometries) {
        _vertices.clear();
        append_geometry_vertices(_vertices, geometry, state.lower_corner, state.upper_corner);
        if (_vertices.empty()) continue;
        _buffer.upload(_vertices.data(), _vertices.size() * sizeof(Float3));
        _buffer.bind();
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Float3), nullptr);
        if (geometry.role == GeometryRenderView::Role::source) {
            glUniform4f(_color_location, 0.20f, 0.90f, 0.35f, 1.0f);
        } else if (geometry.role == GeometryRenderView::Role::sink) {
            glUniform4f(_color_location, 0.95f, 0.30f, 0.25f, 1.0f);
        } else {
            glUniform4f(_color_location, 0.95f, 0.80f, 0.20f, 1.0f);
        }
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(_vertices.size()));
    }
    glBindVertexArray(0);
    opengl::Buffer::unbind();
}

void
GeometryLayer::shutdown() {
    if (_vao != 0) glDeleteVertexArrays(1, &_vao);
    _vao = 0;
    _vertices.clear();
    _buffer = opengl::Buffer();
    _shader = opengl::Shader();
    _view_projection_location = -1;
    _color_location = -1;
}

}
