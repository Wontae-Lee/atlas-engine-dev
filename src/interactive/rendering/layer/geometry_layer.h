#pragma once

#include "rendering/layer/layer.h"

namespace atlas::interactive {

class GeometryLayer final : public Layer {
public:
    void initialize() override;
    void render(const RenderState& state,
                const Camera& camera,
                float aspect_ratio) override;
    void shutdown() override;
};

}
