#pragma once

namespace atlas::interactive {

class Camera;
struct RenderState;

class Layer {
public:
    virtual ~Layer() = default;
    virtual void initialize() = 0;
    virtual void render(const RenderState& state,
                        const Camera& camera,
                        float aspect_ratio) = 0;
    virtual void shutdown() = 0;
};

}
