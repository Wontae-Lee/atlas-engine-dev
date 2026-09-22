#pragma once

namespace atlas::interactive {

struct RenderState;
struct SimulationSceneView;

class StateProvider {
public:
    virtual ~StateProvider() = default;
    virtual const RenderState& update(const SimulationSceneView& view) = 0;
};

}
