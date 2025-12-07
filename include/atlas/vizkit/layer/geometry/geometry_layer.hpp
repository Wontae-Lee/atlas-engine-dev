#ifndef ATLAS_ENGINE_DEV_GEOMETRY_LAYER_HPP
#define ATLAS_ENGINE_DEV_GEOMETRY_LAYER_HPP
#include <atlas/vizkit/shader/glsl.h>
#include <stdexcept>
#include <type_traits>
namespace atlas::vizkit {

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
GeometryLayer<T>::GeometryLayer(unsigned int primitive_mode)
    : _primitive_mode(primitive_mode) {
    static_assert(std::is_same_v<T, float>,
                  "GeometryLayer<T> currently supports only T = float.");
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
GeometryLayer<T>::init(GLFWwindow*, Camera&) {

    std::vector<Vector3<T>> positions;
    build_geometry(positions);

    _vertex_count = static_cast<int>(positions.size());
    if (_vertex_count == 0) {
        throw std::runtime_error("GeometryLayer::init: no vertices generated.");
    }

    _program = std::make_unique<ShaderProgram>(k_line_vs, k_line_fs);
    _u_mvp   = _program->uniform_loc("MVP");

    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(_vertex_count * sizeof(Vector3<T>)),
        positions.data(),
        GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Vector3<T>),
        static_cast<void*>(nullptr));

    glBindVertexArray(0);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
GeometryLayer<T>::update(GLFWwindow* window, Camera& camera) {
    if (_vertex_count <= 0 || !_program || _vao == 0) {
        return;
    }

    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);

    float mvp[16];
    camera.build_mvp(w, h, mvp);
    glLineWidth(3.0f);
    _program->use();
    glBindVertexArray(_vao);
    glUniform4f(_program->uniform_loc("uColor"), 1.0f, 1.0f, 1.0f, 1.0f);
    glUniformMatrix4fv(_u_mvp, 1, GL_FALSE, mvp);
    glDrawArrays(_primitive_mode, 0, _vertex_count);
    glBindVertexArray(0);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
GeometryLayer<T>::shutdown() {
    if (_vbo) {
        glDeleteBuffers(1, &_vbo);
        _vbo = 0;
    }
    if (_vao) {
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }
    _program.reset();
    _u_mvp        = -1;
    _vertex_count = 0;
}

}

#endif