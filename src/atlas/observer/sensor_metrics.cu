#include <atlas/observer/sensor_metrics.h>

#include <fstream>
#include <string>

namespace atlas {

ParticleCountSensorMetrics::ParticleCountSensorMetrics(const std::size_t reserve_count) {
    if (reserve_count > 0) {
        _records.reserve(reserve_count);
    }
}

void
ParticleCountSensorMetrics::ensure_extra_capacity(const std::size_t additional_records) {
    const std::size_t required = _records.size() + additional_records;
    if (required <= _records.capacity()) {
        return;
    }
    std::size_t new_capacity = _records.capacity() == 0 ? std::size_t { 8 } : _records.capacity();
    while (new_capacity < required) {
        new_capacity *= 2;
    }
    _records.reserve(new_capacity);
}

void
ParticleCountSensorMetrics::record(const std::size_t step_index,
                                   const std::size_t unit_index,
                                   const std::size_t particle_count) {
    ensure_extra_capacity(1);
    _records.push_back(Record { step_index, unit_index, particle_count });
}

const HostBuffer<SensorMetrics::Record>&
ParticleCountSensorMetrics::records() const noexcept {
    return _records;
}

std::size_t
ParticleCountSensorMetrics::size() const noexcept {
    return _records.size();
}

void
SourceSensorMetrics::export_csv(const std::filesystem::path& output_directory) const {
    std::filesystem::create_directories(output_directory);
    std::ofstream out(output_directory / std::string(filename()));
    out << "step_index,source_unit_index,particle_count\n";
    for (const auto& record : _records) {
        out << record.step_index << ','
            << record.unit_index << ','
            << record.particle_count << '\n';
    }
}

std::string_view
SourceSensorMetrics::filename() const noexcept {
    return "source_sensor_metrics.csv";
}

void
SinkSensorMetrics::export_csv(const std::filesystem::path& output_directory) const {
    std::filesystem::create_directories(output_directory);
    std::ofstream out(output_directory / std::string(filename()));
    out << "step_index,sink_unit_index,particle_count\n";
    for (const auto& record : _records) {
        out << record.step_index << ','
            << record.unit_index << ','
            << record.particle_count << '\n';
    }
}

std::string_view
SinkSensorMetrics::filename() const noexcept {
    return "sink_sensor_metrics.csv";
}

}
