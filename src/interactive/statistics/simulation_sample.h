/**
 * @file
 * @brief Defines statistics collected after a simulation step.
 */

#pragma once

#include <cstddef>
#include <vector>

namespace atlas::interactive {

/// Per-step counters sampled from one Atlas system.
struct SimulationSample final {
    std::size_t step = 0; ///< Completed system step index.
    double simulation_time = 0.0; ///< Physical simulation time.
    std::size_t particle_count = 0; ///< Live particles after the step.
    std::vector<std::size_t> source_spawned; ///< Particles emitted by each source in this step.
    std::vector<std::size_t> sink_removed; ///< Particles removed by each sink in this step.
};

}
