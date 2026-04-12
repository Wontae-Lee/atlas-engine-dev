#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/core/macros.h>
#include <atlas/math/vector/vector4.h>
#include <atlas/memory/copy.h>
#include <atlas/system/system.h>

#include <vizkit/layer/layer.h>
#include <vizkit/shader/shader_program.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

template <typename T>
class ParticleLayer final : public Layer<T> {
public:
    class Builder;

public:
    ATLAS_HOST ParticleLayer(const atlas::SystemHostPtr<T>& system,
                             const Vector4<T>& color = Vector4<T>(T(0.15), T(0.45), T(0.95), T(0.65)));

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ~ParticleLayer() override = default;

    ATLAS_HOST void
    init(GLFWwindow* window, Camera& camera) override;

    ATLAS_HOST void
    update(GLFWwindow* window, Camera& camera, T dt) override;

    ATLAS_HOST void
    shutdown() override;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate_system() const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GLenum
    vertex_component_type() const noexcept;

private:
    atlas::SystemHostPtr<T> _system {};

    GLuint _vao = 0;

    GLuint _vbo = 0;

    GLint _u_mvp = -1;

    GLint _u_color = -1;

    int _draw_count = 0;

    std::size_t _capacity = 0;

    Vector4<T> _color { T(0.15), T(0.45), T(0.95), T(0.65) };

    std::unique_ptr<ShaderProgram> _program;

#if defined(ATLAS_TASKING_CUDA)
    cudaGraphicsResource* _cuda_vbo_resource = nullptr;
#else
    std::vector<Vector3<T>> _host_positions;
#endif
};

template <typename T>
class ParticleLayer<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_system(const atlas::SystemHostPtr<T>& system) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_color(const Vector4<T>& color) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE ParticleLayer<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE std::shared_ptr<ParticleLayer<T>>
    make_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    atlas::SystemHostPtr<T> _system {};

    Vector4<T> _color { T(0.15), T(0.45), T(0.95), T(0.65) };
};

}

#include <vizkit/layer/particle/particle_layer.hpp>

#endif
