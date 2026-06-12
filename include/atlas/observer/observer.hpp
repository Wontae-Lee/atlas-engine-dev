#pragma once

#include <utility>

namespace atlas {

inline Observer::Builder
Observer::builder() noexcept {
    // Return a fresh builder for fluent Observer construction.
    return Builder {};
}

template <typename SensorMatricsT, typename... Args>
SensorMatricsT&
Observer::emplace_sensor_matrics(Args&&... args) {
    return _sensor_matrics.template emplace<SensorMatricsT>(std::forward<Args>(args)...);
}

template <typename SensorMatricsT>
void
Observer::set_sensor_matrics(std::unique_ptr<SensorMatricsT> sensor_matrics) {
    _sensor_matrics.template set<SensorMatricsT>(std::move(sensor_matrics));
}

template <typename SensorMatricsT>
SensorMatricsT*
Observer::sensor_matrics() noexcept {
    return _sensor_matrics.template get<SensorMatricsT>();
}

template <typename SensorMatricsT>
const SensorMatricsT*
Observer::sensor_matrics() const noexcept {
    return _sensor_matrics.template get<SensorMatricsT>();
}

template <typename SensorMatricsT>
bool
Observer::has_sensor_matrics() const noexcept {
    return _sensor_matrics.template contains<SensorMatricsT>();
}

template <typename SensorMatricsT>
std::unique_ptr<SensorMatricsT>
Observer::remove_sensor_matrics() {
    return _sensor_matrics.template remove<SensorMatricsT>();
}

inline void
Observer::export_csv(const std::filesystem::path& output_directory) const {
    // Export every registered sensor matrics object into the requested output directory.
    for (const auto& entry : _sensor_matrics) {
        const auto& value = entry.second;
        if (value) {
            // Delegate CSV export to the concrete matrics implementation.
            value->export_csv(output_directory);
        }
    }
}

inline SensorMatricsStore&
Observer::sensor_matrics() noexcept {
    return _sensor_matrics;
}

inline const SensorMatricsStore&
Observer::sensor_matrics() const noexcept {
    return _sensor_matrics;
}

inline Observer::Builder&
Observer::Builder::with_source_sensor_matrics(const std::size_t reserve_count) noexcept {
    // Enable construction of source sensor matrics in build().
    _with_source_sensor_matrics = true;

    // Store the requested preallocation size for source matrics data.
    _source_reserve_count = reserve_count;

    return *this;
}

inline Observer::Builder&
Observer::Builder::with_sink_sensor_matrics(const std::size_t reserve_count) noexcept {
    // Enable construction of sink sensor matrics in build().
    _with_sink_sensor_matrics = true;

    // Store the requested preallocation size for sink matrics data.
    _sink_reserve_count = reserve_count;

    return *this;
}

inline Observer
Observer::Builder::build() const {
    Observer observer;

    if (_with_source_sensor_matrics) {
        // Add source sensor matrics when requested by the builder configuration.
        observer.emplace_sensor_matrics<SourceSensorMatrics>(_source_reserve_count);
    }

    if (_with_sink_sensor_matrics) {
        // Add sink sensor matrics when requested by the builder configuration.
        observer.emplace_sensor_matrics<SinkSensorMatrics>(_sink_reserve_count);
    }

    return observer;
}

inline atlas::host_shared_ptr<Observer>
Observer::Builder::make_host_shared() const {
    // Build the observer first so all requested matrics are initialized before ownership wrapping.
    auto observer = build();

    // Store the constructed observer in host-managed shared ownership.
    return atlas::make_host_shared<Observer>(std::move(observer));
}

} // namespace atlas
