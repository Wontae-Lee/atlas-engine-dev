#include "statistics/simulation_statistics.h"

#include <stdexcept>

namespace atlas::interactive {

void
SimulationStatistics::update(const SimulationSample& sample) {
    if (_sample_count == 0) {
        _total_source_spawned.assign(sample.source_spawned.size(), 0);
        _total_sink_removed.assign(sample.sink_removed.size(), 0);
    }
    if (sample.source_spawned.size() != _total_source_spawned.size()
        || sample.sink_removed.size() != _total_sink_removed.size()) {
        throw std::invalid_argument("Simulation sample topology changed without a statistics reset.");
    }

    _current = sample;
    for (std::size_t index = 0; index < sample.source_spawned.size(); ++index) {
        _total_source_spawned[index] += sample.source_spawned[index];
    }
    for (std::size_t index = 0; index < sample.sink_removed.size(); ++index) {
        _total_sink_removed[index] += sample.sink_removed[index];
    }
    ++_sample_count;
}

void
SimulationStatistics::reset() {
    _current = {};
    _total_source_spawned.clear();
    _total_sink_removed.clear();
    _sample_count = 0;
}

const SimulationSample&
SimulationStatistics::current() const noexcept {
    return _current;
}

const std::vector<std::uint64_t>&
SimulationStatistics::total_source_spawned() const noexcept {
    return _total_source_spawned;
}

const std::vector<std::uint64_t>&
SimulationStatistics::total_sink_removed() const noexcept {
    return _total_sink_removed;
}

std::size_t
SimulationStatistics::sample_count() const noexcept {
    return _sample_count;
}

}
