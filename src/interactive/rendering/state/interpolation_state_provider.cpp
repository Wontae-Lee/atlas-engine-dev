#include "rendering/state/interpolation_state_provider.h"

#include <stdexcept>

namespace atlas::interactive {

const RenderState&
InterpolationStateProvider::update(const System&) {
    throw std::logic_error("Interpolation requires stable particle identity and is not implemented.");
}

}
