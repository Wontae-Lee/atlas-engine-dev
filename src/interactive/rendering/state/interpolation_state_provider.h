#pragma once

#include "rendering/state/state_provider.h"

namespace atlas::interactive {

class InterpolationStateProvider final : public StateProvider {
public:
    const RenderState& update(const SimulationRenderView& view) override;
};

}
