#pragma once

namespace atlas {
class System;
}

namespace atlas::interactive {

struct RenderState;

class StateProvider {
public:
    virtual ~StateProvider() = default;
    virtual const RenderState& update(const System& system) = 0;
};

}
