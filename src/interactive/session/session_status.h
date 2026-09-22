#pragma once

#include "session/session_state.h"

#include <cstddef>

namespace atlas::interactive {

struct SessionStatus final {
    SessionState state = SessionState::ready;
    std::size_t step = 0;
    double simulation_time = 0.0;
    std::size_t particle_count = 0;
    std::size_t source_count = 0;
    std::size_t sink_count = 0;
};

}
