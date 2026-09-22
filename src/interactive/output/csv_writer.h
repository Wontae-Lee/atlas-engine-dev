#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>

namespace atlas::interactive {

struct SimulationSample;
class SimulationStatistics;

class CsvWriter final {
public:
    CsvWriter(const std::filesystem::path& path,
              std::size_t source_count,
              std::size_t sink_count);

    void append(const SimulationSample& sample,
                const SimulationStatistics& statistics);
    void flush();

private:
    std::ofstream _stream;
    std::size_t _source_count = 0;
    std::size_t _sink_count = 0;
};

}
