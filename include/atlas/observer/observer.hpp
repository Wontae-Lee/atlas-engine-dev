#pragma once

#include <utility>

namespace atlas {

inline Observer::Builder
Observer::builder() noexcept {

    return Builder {};
}

template <typename SensorMetricsT, typename... Args>
SensorMetricsT&
Observer::emplace_sensor_metrics(Args&&... args) {
    return _sensor_metrics.template emplace<SensorMetricsT>(std::forward<Args>(args)...);
}

template <typename SensorMetricsT>
void
Observer::set_sensor_metrics(std::unique_ptr<SensorMetricsT> sensor_metrics) {
    _sensor_metrics.template set<SensorMetricsT>(std::move(sensor_metrics));
}

template <typename SensorMetricsT>
SensorMetricsT*
Observer::sensor_metrics() noexcept {
    return _sensor_metrics.template get<SensorMetricsT>();
}

template <typename SensorMetricsT>
const SensorMetricsT*
Observer::sensor_metrics() const noexcept {
    return _sensor_metrics.template get<SensorMetricsT>();
}

template <typename SensorMetricsT>
bool
Observer::has_sensor_metrics() const noexcept {
    return _sensor_metrics.template contains<SensorMetricsT>();
}

template <typename SensorMetricsT>
std::unique_ptr<SensorMetricsT>
Observer::remove_sensor_metrics() {
    return _sensor_metrics.template remove<SensorMetricsT>();
}

inline void
Observer::export_csv(const std::filesystem::path& output_directory) const {

    for (const auto& entry : _sensor_metrics) {
        const auto& value = entry.second;
        if (value) {

            value->export_csv(output_directory);
        }
    }
}

inline SensorMetricsStore&
Observer::sensor_metrics() noexcept {
    return _sensor_metrics;
}

inline const SensorMetricsStore&
Observer::sensor_metrics() const noexcept {
    return _sensor_metrics;
}

inline Observer::Builder&
Observer::Builder::with_source_sensor_metrics(const std::size_t reserve_count) noexcept {

    _with_source_sensor_metrics = true;

    _source_reserve_count = reserve_count;

    return *this;
}

inline Observer::Builder&
Observer::Builder::with_sink_sensor_metrics(const std::size_t reserve_count) noexcept {

    _with_sink_sensor_metrics = true;

    _sink_reserve_count = reserve_count;

    return *this;
}

inline Observer
Observer::Builder::build() const {
    Observer observer;

    if (_with_source_sensor_metrics) {

        observer.emplace_sensor_metrics<SourceSensorMetrics>(_source_reserve_count);
    }

    if (_with_sink_sensor_metrics) {

        observer.emplace_sensor_metrics<SinkSensorMetrics>(_sink_reserve_count);
    }

    return observer;
}

inline atlas::host_shared_ptr<Observer>
Observer::Builder::make_host_shared() const {

    auto observer = build();

    return atlas::make_host_shared<Observer>(std::move(observer));
}

}