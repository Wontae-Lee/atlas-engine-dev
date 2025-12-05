#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/vizkit/camera/camera.h>
#include <atlas/vizkit/layer/layer.h>

#include <atlas/vizkit/shader/shader_program.h>
#include <memory>
#include <vector>

namespace atlas::vizkit {

template <typename T>
class GeometryLayer : public Layer<T> {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE explicit GeometryLayer(unsigned int primitive_mode);

    ATLAS_HOST ATLAS_FORCE_INLINE ~GeometryLayer() override = default;

    ATLAS_HOST void
    init(GLFWwindow* window, Camera& camera) override;

    ATLAS_HOST void
    update(GLFWwindow* window, Camera& camera) override;

    ATLAS_HOST void
    shutdown() override;

protected:
    virtual void
    build_geometry(std::vector<Vector3<T>>& positions)
        = 0;

    std::unique_ptr<ShaderProgram> _program;
    int _u_mvp { -1 };

    unsigned int _primitive_mode { 0 };
    unsigned int _vao { 0 };
    unsigned int _vbo { 0 };
    int _vertex_count { 0 };
};

}

#include <atlas/vizkit/layer/geometry/geometry_layer.hpp>

#endif
