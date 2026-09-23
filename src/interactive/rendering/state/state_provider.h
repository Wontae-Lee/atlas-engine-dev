/**
 * @file
 * @brief Declares the simulation-to-render-state strategy interface.
 */

#pragma once

namespace atlas::interactive {

struct RenderState;
struct SimulationSceneView;

/// Strategy interface for producing graphics state from simulation views.
class StateProvider {
public:
    virtual ~StateProvider() = default;
    /**
     * @brief Updates graphics resources from the current simulation scene.
     * @return Provider-owned state valid until the next update or destruction.
     */
    virtual const RenderState& update(const SimulationSceneView& view) = 0;
};

}
