#pragma once

namespace atlas::interactive {

struct RenderState;
struct SimulationRenderView;

class StateProvider {
public:
    virtual ~StateProvider() = default;
    virtual const RenderState& update(const SimulationRenderView& view) = 0;
};

}
