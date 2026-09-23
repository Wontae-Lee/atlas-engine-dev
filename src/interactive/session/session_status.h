/**
 * @file
 * @brief Defines a snapshot of interactive session status.
 */

#pragma once

#include "session/session_state.h"

#include <cstddef>

namespace atlas::interactive {

/// Lightweight externally reportable snapshot of a session.
struct SessionStatus final {
    SessionState state = SessionState::ready; ///< Current lifecycle state.
    std::size_t step = 0; ///< Number of completed core updates.
    double simulation_time = 0.0; ///< Physical time represented by the system.
    std::size_t particle_count = 0; ///< Current live particle count.
    std::size_t source_count = 0; ///< Installed source count.
    std::size_t sink_count = 0; ///< Installed sink count.
};

}
