/**
 * @file
 * @brief Declares CSV output for interactive simulation statistics.
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>

namespace atlas::interactive {

struct SimulationSample;
class SimulationStatistics;

/// Writes session statistics as rows in one CSV stream.
class CsvWriter final {
public:
    /**
     * @brief Opens a CSV file and writes a header matching the boundary counts.
     * @param path Destination file path.
     * @param source_count Number of per-source columns.
     * @param sink_count Number of per-sink columns.
     */
    CsvWriter(const std::filesystem::path& path,
              std::size_t source_count,
              std::size_t sink_count);

    /// Appends one step sample and its accumulated totals.
    void append(const SimulationSample& sample,
                const SimulationStatistics& statistics);

    /// Flushes buffered output to the underlying file.
    void flush();

private:
    std::ofstream _stream; ///< Owned output stream.
    std::size_t _source_count = 0; ///< Fixed number of source columns.
    std::size_t _sink_count = 0; ///< Fixed number of sink columns.
};

}
