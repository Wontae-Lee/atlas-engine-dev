#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
#include <atlas/math/math.h>
#include <atlas/system/particle_system.h>
#include <vizkit/layer/layer.h>
#include <vizkit/shader/shader_program.h>
#include <memory>

namespace atlas::vizkit {

template <typename T>
class ParticleLayer final : public Layer<T> {
public:
    using ParticleSystemPtr = ParticleSystemHostPtr<T>;

    ATLAS_HOST ATLAS_FORCE_INLINE
    ParticleLayer(ParticleSystemPtr psystem, size_t buffer_size = 100000);

    ~ParticleLayer() override = default;

    ATLAS_HOST void
    init(GLFWwindow* window, Camera& camera) override;

    ATLAS_HOST void
    update(GLFWwindow* window, Camera& camera) override;

    ATLAS_HOST void
    shutdown() override;

private:
    size_t _buffer_size;
    ParticleSystemPtr _psystem;

    std::unique_ptr<ShaderProgram> _point_prog;
    GLint _u_mvp_points { -1 };

    GLuint _vao { 0 };
    GLuint _vbo { 0 };
    cudaGraphicsResource* _cuda_vbo { nullptr };

    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_point_pipeline();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_interop_buffers();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    destroy_interop_buffers();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    copy_positions_to_vbo(const Vector3F* d_src, int count);
};

}

#include <vizkit/layer/particle/particle_layer.hpp>

#endif
