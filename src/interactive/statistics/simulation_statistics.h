#pragma once

#include "statistics/simulation_sample.h"

#include <cstdint>
#include <vector>

namespace atlas::interactive {

class SimulationStatistics final {
public:
    void update(const SimulationSample& sample);
    void reset();

    const SimulationSample& current() const noexcept;
    const std::vector<std::uint64_t>& total_source_spawned() const noexcept;
    const std::vector<std::uint64_t>& total_sink_removed() const noexcept;
    std::size_t sample_count() const noexcept;

private:
    SimulationSample _current;
    std::vector<std::uint64_t> _total_source_spawned;
    std::vector<std::uint64_t> _total_sink_removed;
    std::size_t _sample_count = 0;
};

}
