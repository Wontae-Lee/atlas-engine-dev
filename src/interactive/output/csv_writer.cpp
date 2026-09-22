#include "output/csv_writer.h"

#include "statistics/simulation_sample.h"
#include "statistics/simulation_statistics.h"

#include <iomanip>
#include <stdexcept>

namespace atlas::interactive {

CsvWriter::CsvWriter(const std::filesystem::path& path,
                     const std::size_t source_count,
                     const std::size_t sink_count)
    : _source_count(source_count)
    , _sink_count(sink_count) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    _stream.open(path, std::ios::out | std::ios::trunc);
    if (!_stream) throw std::runtime_error("Failed to open interactive CSV output: " + path.string());

    _stream << "step,time,particle_count";
    for (std::size_t index = 0; index < source_count; ++index) {
        _stream << ",source_" << index << "_spawned";
    }
    for (std::size_t index = 0; index < sink_count; ++index) {
        _stream << ",sink_" << index << "_removed";
    }
    for (std::size_t index = 0; index < source_count; ++index) {
        _stream << ",source_" << index << "_total";
    }
    for (std::size_t index = 0; index < sink_count; ++index) {
        _stream << ",sink_" << index << "_total";
    }
    _stream << '\n';
}

void
CsvWriter::append(const SimulationSample& sample,
                  const SimulationStatistics& statistics) {
    if (sample.source_spawned.size() != _source_count
        || sample.sink_removed.size() != _sink_count) {
        throw std::invalid_argument("CSV sample topology does not match its header.");
    }

    _stream << sample.step << ',' << std::setprecision(17) << sample.simulation_time << ','
            << sample.particle_count;
    for (const std::size_t value : sample.source_spawned) _stream << ',' << value;
    for (const std::size_t value : sample.sink_removed) _stream << ',' << value;
    for (const std::uint64_t value : statistics.total_source_spawned()) _stream << ',' << value;
    for (const std::uint64_t value : statistics.total_sink_removed()) _stream << ',' << value;
    _stream << '\n';

    if (!_stream) throw std::runtime_error("Failed to append interactive CSV output.");
}

void
CsvWriter::flush() {
    _stream.flush();
    if (!_stream) throw std::runtime_error("Failed to flush interactive CSV output.");
}

}
