#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/logging/logging.h>
#include <vizkit/shader/glsl.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace atlas::vizkit {

template <typename T>
ParticleLayer<T>::ParticleLayer(const atlas::SystemHostPtr<T>& system,
                                const Vector4<T>& color)
    : _system(system)
    , _color(color) {
    static_assert(std::is_floating_point_v<T>, "ParticleLayer requires a floating-point T");
}

template <typename T>
typename ParticleLayer<T>::Builder
ParticleLayer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
ParticleLayer<T>::init(GLFWwindow* window, Camera& camera) {
    (void)window;
    (void)camera;

    validate_system();

    _capacity = _system->fluid()->buffer_size();
    _draw_count = 0;

    _program = std::make_unique<ShaderProgram>(k_point_vs, k_point_fs);

    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);

    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(_capacity * sizeof(Vector3<T>)),
        nullptr,
        GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        vertex_component_type(),
        GL_FALSE,
        sizeof(Vector3<T>),
        reinterpret_cast<void*>(0));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    _u_mvp   = _program->uniform_loc("MVP");
    _u_color = _program->uniform_loc("uColor");

#if defined(ATLAS_TASKING_CUDA)
    const cudaError_t register_error = cudaGraphicsGLRegisterBuffer(
        &_cuda_vbo_resource,
        _vbo,
        cudaGraphicsRegisterFlagsWriteDiscard);
    if (register_error != cudaSuccess) {
        shutdown();
        throw std::runtime_error(
            std::string("ParticleLayer: failed to register CUDA-GL buffer: ")
            + cudaGetErrorString(register_error));
    }
#else
    _host_positions.resize(_capacity);
#endif
}

template <typename T>
void
ParticleLayer<T>::update(GLFWwindow* window, Camera& camera, T dt) {
    (void)dt;

    if (!_program || !_vao || !_vbo || !_system) return;

    const auto& particle = _system->particle_probe();
    if (particle.pos == nullptr || particle.particle_count <= 0) return;

    const std::size_t active_count = std::min<std::size_t>(
        static_cast<std::size_t>(particle.particle_count),
        _capacity);
    if (active_count == 0) return;

#if defined(ATLAS_TASKING_CUDA)
    if (_cuda_vbo_resource == nullptr) return;

    cudaGraphicsMapResources(1, &_cuda_vbo_resource, 0);

    void* mapped_buffer = nullptr;
    std::size_t mapped_bytes = 0;
    cudaGraphicsResourceGetMappedPointer(&mapped_buffer, &mapped_bytes, _cuda_vbo_resource);

    const std::size_t required_bytes = active_count * sizeof(Vector3<T>);
    if (mapped_buffer != nullptr && mapped_bytes >= required_bytes) {
        cudaMemcpy(mapped_buffer, particle.pos, required_bytes, cudaMemcpyDeviceToDevice);
    }

    cudaGraphicsUnmapResources(1, &_cuda_vbo_resource, 0);
#else
    atlas::copy_device_to_host(particle.pos, _host_positions.data(), active_count);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        static_cast<GLsizeiptr>(active_count * sizeof(Vector3<T>)),
        _host_positions.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif

    _draw_count = static_cast<int>(active_count);

    int width = 1;
    int height = 1;
    glfwGetFramebufferSize(window, &width, &height);

    float mvp[16];
    camera.build_mvp(width, height, mvp);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    _program->use();
    glUniformMatrix4fv(_u_mvp, 1, GL_FALSE, mvp);
    if (_u_color >= 0) {
        glUniform4f(
            _u_color,
            static_cast<float>(_color.x),
            static_cast<float>(_color.y),
            static_cast<float>(_color.z),
            static_cast<float>(_color.w));
    }

    glBindVertexArray(_vao);
    glDrawArrays(GL_POINTS, 0, _draw_count);
    glBindVertexArray(0);
}

template <typename T>
void
ParticleLayer<T>::shutdown() {
#if defined(ATLAS_TASKING_CUDA)
    if (_cuda_vbo_resource != nullptr) {
        cudaGraphicsUnregisterResource(_cuda_vbo_resource);
        _cuda_vbo_resource = nullptr;
    }
#else
    _host_positions.clear();
#endif

    if (_vbo) {
        glDeleteBuffers(1, &_vbo);
        _vbo = 0;
    }

    if (_vao) {
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }

    _u_mvp = -1;
    _u_color = -1;
    _draw_count = 0;
    _capacity = 0;
    _program.reset();
}

template <typename T>
void
ParticleLayer<T>::validate_system() const {
    if (_system == nullptr) {
        atlas::logger::error()
            << "ParticleLayer: system must not be null.";
        throw std::runtime_error("ParticleLayer: system must not be null.");
    }

    if (_system->fluid() == nullptr) {
        atlas::logger::error()
            << "ParticleLayer: system fluid must not be null.";
        throw std::runtime_error("ParticleLayer: system fluid must not be null.");
    }
}

template <typename T>
GLenum
ParticleLayer<T>::vertex_component_type() const noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return GL_DOUBLE;
    }

    return GL_FLOAT;
}

template <typename T>
typename ParticleLayer<T>::Builder&
ParticleLayer<T>::Builder::with_color(const Vector4<T>& color) noexcept {
    _color = color;
    return *this;
}

template <typename T>
typename ParticleLayer<T>::Builder&
ParticleLayer<T>::Builder::with_system(const atlas::SystemHostPtr<T>& system) noexcept {
    _system = system;
    return *this;
}

template <typename T>
ParticleLayer<T>
ParticleLayer<T>::Builder::build() const {
    validate();
    return ParticleLayer<T>(_system, _color);
}

template <typename T>
std::shared_ptr<ParticleLayer<T>>
ParticleLayer<T>::Builder::make_shared() const {
    validate();
    return std::make_shared<ParticleLayer<T>>(_system, _color);
}

template <typename T>
void
ParticleLayer<T>::Builder::validate() const {
    if (_system == nullptr) {
        atlas::logger::error()
            << "ParticleLayer::Builder: system must not be null.";
        throw std::runtime_error("ParticleLayer::Builder: system must not be null.");
    }

    if (_system->fluid() == nullptr) {
        atlas::logger::error()
            << "ParticleLayer::Builder: system fluid must not be null.";
        throw std::runtime_error("ParticleLayer::Builder: system fluid must not be null.");
    }
}

}

#endif
