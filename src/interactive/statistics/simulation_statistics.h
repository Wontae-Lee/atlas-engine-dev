/**
 * @file
 * @brief Declares accumulation of interactive simulation statistics.
 */

#pragma once

#include "statistics/simulation_sample.h"

#include <cstdint>
#include <vector>

namespace atlas::interactive {

/// Retains the latest sample and accumulated boundary counters.
class SimulationStatistics final {
public:
    /// Records one sample and adds its source and sink counters.
    void update(const SimulationSample& sample);
    /// Clears the latest sample and all accumulated counters.
    void reset();

    /// Returns the most recently recorded sample.
    const SimulationSample& current() const noexcept;
    /// Returns lifetime emitted-particle totals indexed by source.
    const std::vector<std::uint64_t>& total_source_spawned() const noexcept;
    /// Returns lifetime removed-particle totals indexed by sink.
    const std::vector<std::uint64_t>& total_sink_removed() const noexcept;
    /// Returns the number of recorded samples.
    std::size_t sample_count() const noexcept;

private:
    SimulationSample _current; ///< Latest complete sample.
    std::vector<std::uint64_t> _total_source_spawned; ///< Totals indexed by source.
    std::vector<std::uint64_t> _total_sink_removed; ///< Totals indexed by sink.
    std::size_t _sample_count = 0; ///< Number of accumulated samples.
};

}
