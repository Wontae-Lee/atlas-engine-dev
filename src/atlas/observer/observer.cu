#include <atlas/observer/observer.h>

#include <utility>

namespace atlas {

Observer::Builder
Observer::builder() noexcept {
    return Builder {};
}

void
Observer::export_csv(const std::filesystem::path& output_directory) const {
    for (const auto& entry : _sensor_metrics) {
        const auto& value = entry.second;
        if (value) {
            value->export_csv(output_directory);
        }
    }
}

SensorMetricsStore&
Observer::sensor_metrics() noexcept {
    return _sensor_metrics;
}

const SensorMetricsStore&
Observer::sensor_metrics() const noexcept {
    return _sensor_metrics;
}

Observer::Builder&
Observer::Builder::with_source_sensor_metrics(const std::size_t reserve_count) noexcept {
    _with_source_sensor_metrics = true;

    _source_reserve_count = reserve_count;

    return *this;
}

Observer::Builder&
Observer::Builder::with_sink_sensor_metrics(const std::size_t reserve_count) noexcept {
    _with_sink_sensor_metrics = true;

    _sink_reserve_count = reserve_count;

    return *this;
}

Observer
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

atlas::host_shared_ptr<Observer>
Observer::Builder::make_host_shared() const {
    auto observer = build();

    return atlas::make_host_shared<Observer>(std::move(observer));
}

}
