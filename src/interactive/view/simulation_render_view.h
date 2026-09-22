#pragma once

#include "view/simulation_buffer_view.h"

#include <cstddef>
#include <optional>

namespace atlas::interactive {

struct SimulationRenderView final {
    std::size_t particle_count = 0;
    SimulationBufferView position;
    SimulationBufferView velocity;
    SimulationBufferView species;
    std::optional<SimulationBufferView> temperature;
    std::optional<SimulationBufferView> translational_energy;
    std::optional<SimulationBufferView> rotational_energy;
    std::optional<SimulationBufferView> vibrational_energy;
};

}
