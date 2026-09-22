#pragma once

#include "rendering/camera.h"

#include <memory>
#include <vector>

namespace atlas {
class System;
}

namespace atlas::interactive {

class Layer;
class RenderTarget;
class StateProvider;

class Renderer final {
public:
    explicit Renderer(std::unique_ptr<StateProvider> provider);
    ~Renderer();

    void initialize(RenderTarget& target);
    void add_layer(std::unique_ptr<Layer> layer);
    void render(const System& system, RenderTarget& target);
    void shutdown();

    Camera& camera() noexcept;
    const Camera& camera() const noexcept;

private:
    std::unique_ptr<StateProvider> _provider;
    std::vector<std::unique_ptr<Layer>> _layers;
    Camera _camera;
    bool _initialized = false;
};

}
