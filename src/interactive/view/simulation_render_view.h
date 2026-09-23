/**
 * @file
 * @brief Defines non-owning views of live particle-state buffers.
 */

#pragma once

#include "view/simulation_buffer_view.h"

#include <cstddef>
#include <optional>

namespace atlas::interactive {

/// Non-owning views of every live particle state exposed for rendering.
struct SimulationRenderView final {
    std::size_t particle_count = 0; ///< Number of live entries in every state view.
    SimulationBufferView position; ///< Mandatory packed atlas::Float3 positions.
    SimulationBufferView velocity; ///< Mandatory packed atlas::Float3 velocities.
    SimulationBufferView species; ///< Mandatory packed material indices.
    std::optional<SimulationBufferView> temperature; ///< Optional scalar temperatures.
    std::optional<SimulationBufferView> translational_energy; ///< Optional scalar energies.
    std::optional<SimulationBufferView> rotational_energy; ///< Optional scalar energies.
    std::optional<SimulationBufferView> vibrational_energy; ///< Optional scalar energies.
};

}
