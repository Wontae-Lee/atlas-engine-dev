#pragma once

#include "rendering/layer/layer.h"
#include "rendering/opengl/shader.h"

namespace atlas::interactive {

class ParticleLayer final : public Layer {
public:
    explicit ParticleLayer(float point_size = 3.0f) noexcept;
    ~ParticleLayer() override;

    void initialize() override;
    void render(const RenderState& state,
                const Camera& camera,
                float aspect_ratio) override;
    void shutdown() override;

private:
    opengl::Shader _shader;
    unsigned int _vao = 0;
    int _view_projection_location = -1;
    int _color_location = -1;
    float _point_size = 3.0f;
};

}
