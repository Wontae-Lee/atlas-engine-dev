/**
 * @file
 * @brief Declares rendering of live particles.
 */

#pragma once

#include "rendering/layer/layer.h"
#include "rendering/opengl/shader.h"

namespace atlas::interactive {

/// Draws every live particle as an OpenGL point.
class ParticleLayer final : public Layer {
public:
    /// Creates a particle layer with the requested point diameter.
    explicit ParticleLayer(float point_size = 3.0f) noexcept;
    ~ParticleLayer() override;

    /// @copydoc Layer::initialize
    void initialize() override;
    /// @copydoc Layer::render
    void render(const RenderState& state,
                const Camera& camera,
                float aspect_ratio) override;
    /// @copydoc Layer::shutdown
    void shutdown() override;

private:
    opengl::Shader _shader; ///< Point-rendering shader program.
    unsigned int _vao = 0; ///< Vertex-array object describing position data.
    int _view_projection_location = -1; ///< Camera-transform uniform location.
    int _color_location = -1; ///< Point-color uniform location.
    float _point_size = 3.0f; ///< Point diameter in pixels.
};

}
