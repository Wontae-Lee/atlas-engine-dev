/**
 * @file
 * @brief Declares direct conversion of current simulation state into render state.
 */

#pragma once

#include "rendering/backend/state_bridge.h"
#include "rendering/state/render_state.h"
#include "rendering/state/state_provider.h"

namespace atlas::interactive {

/// Uploads every current live particle state without interpolation.
class RawStateProvider final : public StateProvider {
public:
    /// @copydoc StateProvider::update
    const RenderState& update(const SimulationSceneView& view) override;
    void finish_frame() override;

private:
    RenderState _state; ///< Persistent graphics resources reused across frames.
    StateBridge _bridge; ///< Backend-specific simulation-to-OpenGL transfer.
};

}
