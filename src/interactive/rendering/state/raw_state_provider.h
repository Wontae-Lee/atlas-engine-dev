#pragma once

#include "rendering/backend/state_bridge.h"
#include "rendering/state/render_state.h"
#include "rendering/state/state_provider.h"

namespace atlas::interactive {

class RawStateProvider final : public StateProvider {
public:
    const RenderState& update(const System& system) override;

private:
    RenderState _state;
    StateBridge _bridge;
};

}
