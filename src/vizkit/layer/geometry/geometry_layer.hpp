#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <cmath>
#include <vizkit/shader/glsl.h>

namespace atlas::vizkit {

template <typename T>
GeometryLayer<T>::GeometryLayer(unsigned int primitive_mode,
                                const atlas::UnitHostPtr<T>& unit)

    : _unit(unit)
    , _primitive_mode(primitive_mode) { }

template <typename T>
void
GeometryLayer<T>::init(GLFWwindow* window, Camera& camera) {
    (void)window;
    (void)camera;

    _local_positions.clear();
    build_geometry(_local_positions);

    _world_positions = _local_positions;

    synchronize(_world_positions, T(0));

    _vertex_count = static_cast<int>(_world_positions.size());
    if (_vertex_count <= 0) return;

    _program = std::make_unique<ShaderProgram>(k_line_vs, k_line_fs);

    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);

    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);

    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(_world_positions.size() * sizeof(Vector3<T>)),
        _world_positions.data(),
        GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Vector3<T>),
        reinterpret_cast<void*>(0));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    _u_mvp   = _program->uniform_loc("MVP");
    _u_color = _program->uniform_loc("uColor");
}

template <typename T>
void
GeometryLayer<T>::update(GLFWwindow* window, Camera& camera, T dt) {

    if (!_program || !_vao || _vertex_count <= 0) return;

    if (_unit) {

        _unit->update(dt);
    }

    if (synchronize(_world_positions, dt)) {
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);

        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            static_cast<GLsizeiptr>(_world_positions.size() * sizeof(Vector3<T>)),
            _world_positions.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    int w = 1;
    int h = 1;
    glfwGetFramebufferSize(window, &w, &h);

    float mvp[16];
    camera.build_mvp(w, h, mvp);

    _program->use();

    glUniformMatrix4fv(_u_mvp, 1, GL_FALSE, mvp);

    if (_u_color >= 0) {

        glUniform4f(_u_color, 0.90f, 0.95f, 1.00f, 1.00f);
    }

    glBindVertexArray(_vao);
    glDrawArrays(_primitive_mode, 0, _vertex_count);
    glBindVertexArray(0);
}

template <typename T>
void
GeometryLayer<T>::shutdown() {

    if (_vbo) {
        glDeleteBuffers(1, &_vbo);
        _vbo = 0;
    }
    if (_vao) {
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }

    _vertex_count = 0;
    _u_mvp        = -1;
    _u_color      = -1;
    _local_positions.clear();
    _world_positions.clear();
    _program.reset();
}

template <typename T>
bool
GeometryLayer<T>::synchronize(std::vector<Vector3<T>>& world_positions, T dt) {
    (void)dt;

    if (world_positions.size() != _local_positions.size()) {
        world_positions.resize(_local_positions.size());
    }

    const T eps  = static_cast<T>(1e-6);
    bool changed = false;

    if (!_unit) {

        for (std::size_t i = 0; i < _local_positions.size(); ++i) {
            const auto& src = _local_positions[i];
            auto& dst       = world_positions[i];

            if (std::abs(dst.x - src.x) > eps || std::abs(dst.y - src.y) > eps || std::abs(dst.z - src.z) > eps) {
                changed = true;
            }

            dst = src;
        }
        return changed;
    }

    const auto& sync = _unit->sync_operator();

    for (std::size_t i = 0; i < _local_positions.size(); ++i) {

        const Vector3<T> transformed = sync.sync_to_world(_local_positions[i]);
        auto& dst                    = world_positions[i];

        if (std::abs(dst.x - transformed.x) > eps || std::abs(dst.y - transformed.y) > eps || std::abs(dst.z - transformed.z) > eps) {
            changed = true;
        }

        dst = transformed;
    }

    return changed;
}

}

#endif