#include "rendering/renderer.h"

#include "rendering/layer/layer.h"
#include "rendering/state/render_state.h"
#include "rendering/state/state_provider.h"
#include "rendering/target/render_target.h"

#include <stdexcept>
#include <utility>

namespace atlas::interactive {

Renderer::Renderer(std::unique_ptr<StateProvider> provider)
    : _provider(std::move(provider)) {
    if (!_provider) throw std::invalid_argument("Renderer requires a StateProvider.");
}

Renderer::~Renderer() = default;

void
Renderer::initialize(RenderTarget&) {
    if (_initialized) return;
    for (auto& layer : _layers) layer->initialize();
    _initialized = true;
}

void
Renderer::add_layer(std::unique_ptr<Layer> layer) {
    if (_initialized) throw std::logic_error("Layers must be added before Renderer initialization.");
    if (!layer) throw std::invalid_argument("Renderer cannot add a null Layer.");
    _layers.push_back(std::move(layer));
}

void
Renderer::render(const SimulationSceneView& view, RenderTarget& target) {
    if (!_initialized || !_provider) throw std::logic_error("Renderer is not initialized.");
    const RenderState& state = _provider->update(view);
    target.begin_frame();
    const float aspect = static_cast<float>(target.width()) / static_cast<float>(target.height());
    for (auto& layer : _layers) layer->render(state, _camera, aspect);
    target.end_frame();
    target.present();
}

void
Renderer::shutdown() {
    if (!_initialized) return;
    for (auto iterator = _layers.rbegin(); iterator != _layers.rend(); ++iterator) {
        (*iterator)->shutdown();
    }
    _layers.clear();
    _provider.reset();
    _initialized = false;
}

Camera&
Renderer::camera() noexcept {
    return _camera;
}

const Camera&
Renderer::camera() const noexcept {
    return _camera;
}

}
