#pragma once

#include <cstddef>
#include <vector>

namespace atlas::interactive {

struct SimulationSample final {
    std::size_t step = 0;
    double simulation_time = 0.0;
    std::size_t particle_count = 0;
    std::vector<std::size_t> source_spawned;
    std::vector<std::size_t> sink_removed;
};

}
