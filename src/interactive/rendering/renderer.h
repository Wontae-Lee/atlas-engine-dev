/**
 * @file
 * @brief Declares orchestration of render state, layers, and targets.
 */

#pragma once

#include "rendering/camera.h"

#include <memory>
#include <vector>

namespace atlas::interactive {

class Layer;
class RenderTarget;
class StateProvider;
struct SimulationSceneView;

/**
 * @brief Coordinates state preparation, layers, camera, and render targets.
 *
 * Renderer never advances the simulation. Its caller controls the relationship
 * between simulation steps and presented frames.
 */
class Renderer final {
public:
    /// Takes ownership of the render-state provider.
    explicit Renderer(std::unique_ptr<StateProvider> provider);
    ~Renderer();

    /// Initializes all layers against the target's current context.
    void initialize(RenderTarget& target);
    /// Adds an owned layer in draw order.
    void add_layer(std::unique_ptr<Layer> layer);
    /// Updates render state and draws one frame to target.
    void render(const SimulationSceneView& view, RenderTarget& target);
    /// Releases all layer resources.
    void shutdown();

    /// Returns the mutable renderer camera.
    Camera& camera() noexcept;
    /// Returns the renderer camera.
    const Camera& camera() const noexcept;

private:
    std::unique_ptr<StateProvider> _provider; ///< Owned simulation-to-render strategy.
    std::vector<std::unique_ptr<Layer>> _layers; ///< Owned layers in draw order.
    Camera _camera; ///< Camera shared by every layer.
    bool _initialized = false; ///< Whether layer resources are live.
};

}
