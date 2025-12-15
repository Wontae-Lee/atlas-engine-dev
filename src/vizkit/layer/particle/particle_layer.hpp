#pragma once

#include <vizkit/shader/glsl.h>
#include <stdexcept>

namespace atlas::vizkit {

template <typename T>
ParticleLayer<T>::ParticleLayer(ParticleSystemPtr psystem, size_t buffer_size)
    : _buffer_size(buffer_size)
    , _psystem(std::move(psystem)) {
    static_assert(std::is_same_v<T, float>,
                  "ParticleLayer<T> currently only supports T = float for OpenGL interop.");
}

template <typename T>
void
ParticleLayer<T>::init(GLFWwindow*, Camera&) {
    if (!_psystem) {
        throw std::runtime_error("ParticleLayer::init: particle system is null");
    }

    init_point_pipeline();
    init_interop_buffers();
}

template <typename T>
void
ParticleLayer<T>::update(GLFWwindow* window, Camera& camera) {

    _psystem->update();

    auto particles = _psystem->particles();
    auto probe     = particles->make_device_probe();
    auto count     = _psystem->alive();

    if (count <= 0) {
        return;
    }
    copy_positions_to_vbo(probe.pos, count);

    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);

    float mvp[16];
    camera.build_mvp(w, h, mvp);

    _point_prog->use();
    glBindVertexArray(_vao);
    glUniformMatrix4fv(_u_mvp_points, 1, GL_FALSE, mvp);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
}

template <typename T>
void
ParticleLayer<T>::shutdown() {
    destroy_interop_buffers();
}

template <typename T>
void
ParticleLayer<T>::init_point_pipeline() {
    _point_prog   = std::make_unique<ShaderProgram>(k_point_vs, k_point_fs);
    _u_mvp_points = _point_prog->uniform_loc("MVP");
}

template <typename T>
void
ParticleLayer<T>::init_interop_buffers() {
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        _buffer_size * sizeof(Vector3F),
        nullptr,
        GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Vector3F),
        static_cast<void*>(nullptr));

    glBindVertexArray(0);

    cudaGraphicsGLRegisterBuffer(&_cuda_vbo, _vbo, cudaGraphicsMapFlagsWriteDiscard);
}

template <typename T>
void
ParticleLayer<T>::destroy_interop_buffers() {
    if (_cuda_vbo) {
        cudaGraphicsUnregisterResource(_cuda_vbo);
        _cuda_vbo = nullptr;
    }
    if (_vbo) {
        glDeleteBuffers(1, &_vbo);
        _vbo = 0;
    }
    if (_vao) {
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }
}

template <typename T>
void
ParticleLayer<T>::copy_positions_to_vbo(const Vector3F* d_src, int count) {
    if (count <= 0 || !_cuda_vbo) {
        return;
    }

    cudaGraphicsMapResources(1, &_cuda_vbo);
    size_t bytes    = 0;
    Vector3F* d_dst = nullptr;
    cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&d_dst),
        &bytes,
        _cuda_vbo);

    cudaMemcpy(
        d_dst,
        d_src,
        static_cast<size_t>(count) * sizeof(Vector3F),
        cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &_cuda_vbo);
}

}