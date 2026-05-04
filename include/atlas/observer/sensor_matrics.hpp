#pragma once
#include <fstream>
namespace atlas::observer {
inline ParticleCountSensorMatrics::ParticleCountSensorMatrics(const std::size_t reserve_count) {
    if (reserve_count > 0) {
        _records.reserve(reserve_count);
    }
}

inline void
ParticleCountSensorMatrics::ensure_extra_capacity(const std::size_t additional_records) {
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

inline void
ParticleCountSensorMatrics::record(const std::size_t step_index,
                                   const std::size_t unit_index,
                                   const std::size_t particle_count) {
    ensure_extra_capacity(1);
    _records.push_back(Record { step_index, unit_index, particle_count });
}

inline const HostBuffer<SensorMatrics::Record>&
ParticleCountSensorMatrics::records() const noexcept {
    return _records;
}

inline std::size_t
ParticleCountSensorMatrics::size() const noexcept {
    return _records.size();
}

inline void
SourceSensorMatrics::export_csv(const std::filesystem::path& output_directory) const {
    std::filesystem::create_directories(output_directory);
    std::ofstream out(output_directory / std::string(filename()));
    out << "step_index,source_unit_index,particle_count\n";
    const HostBuffer<Record> records(_records.begin(), _records.end());
    for (const auto& record : records) {
        out << record.step_index << ','
            << record.unit_index << ','
            << record.particle_count << '\n';
    }
}

inline std::string_view
SourceSensorMatrics::filename() const noexcept {
    return "source_sensor_matrics.csv";
}

inline void
SinkSensorMatrics::export_csv(const std::filesystem::path& output_directory) const {
    std::filesystem::create_directories(output_directory);
    std::ofstream out(output_directory / std::string(filename()));
    out << "step_index,sink_unit_index,particle_count\n";
    const HostBuffer<Record> records(_records.begin(), _records.end());
    for (const auto& record : records) {
        out << record.step_index << ','
            << record.unit_index << ','
            << record.particle_count << '\n';
    }
}

inline std::string_view
SinkSensorMatrics::filename() const noexcept {
    return "sink_sensor_matrics.csv";
}

}