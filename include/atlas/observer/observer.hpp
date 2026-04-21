#pragma once

#include <stdexcept>
#include <utility>

namespace atlas::observer {

inline Observer::Builder
Observer::builder() noexcept {
    return Builder {};
}

template <typename SensorMatricsT, typename... Args>
SensorMatricsT&
Observer::emplace_sensor_matrics(Args&&... args) {
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::observer::SensorMatrics.");

    auto sensor_matrics = std::make_unique<SensorMatricsT>(std::forward<Args>(args)...);
    auto* ptr = sensor_matrics.get();
    _sensor_matrics.insert_or_assign(typeid(SensorMatricsT), std::move(sensor_matrics));
    return *ptr;
}

template <typename SensorMatricsT>
void
Observer::set_sensor_matrics(std::unique_ptr<SensorMatricsT> sensor_matrics) {
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::observer::SensorMatrics.");

    if (sensor_matrics == nullptr) {
        throw std::invalid_argument("Observer::set_sensor_matrics failed: sensor_matrics must not be null.");
    }

    _sensor_matrics.insert_or_assign(typeid(SensorMatricsT), std::move(sensor_matrics));
}

template <typename SensorMatricsT>
SensorMatricsT*
Observer::sensor_matrics() noexcept {
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::observer::SensorMatrics.");

    const auto it = _sensor_matrics.find(typeid(SensorMatricsT));
    return it == _sensor_matrics.end() ? nullptr : static_cast<SensorMatricsT*>(it->second.get());
}

template <typename SensorMatricsT>
const SensorMatricsT*
Observer::sensor_matrics() const noexcept {
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::observer::SensorMatrics.");

    const auto it = _sensor_matrics.find(typeid(SensorMatricsT));
    return it == _sensor_matrics.end() ? nullptr : static_cast<const SensorMatricsT*>(it->second.get());
}

template <typename SensorMatricsT>
bool
Observer::has_sensor_matrics() const noexcept {
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::observer::SensorMatrics.");

    return _sensor_matrics.contains(typeid(SensorMatricsT));
}

template <typename SensorMatricsT>
std::unique_ptr<SensorMatricsT>
Observer::remove_sensor_matrics() {
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::observer::SensorMatrics.");

    const auto it = _sensor_matrics.find(typeid(SensorMatricsT));
    if (it == _sensor_matrics.end()) {
        return nullptr;
    }

    auto sensor_matrics = std::unique_ptr<SensorMatricsT>(static_cast<SensorMatricsT*>(it->second.release()));
    _sensor_matrics.erase(it);
    return sensor_matrics;
}

inline void
Observer::export_csv(const std::filesystem::path& output_directory) const {
    for (const auto& sensor_matrics : _sensor_matrics) {
        if (sensor_matrics.second) {
            sensor_matrics.second->export_csv(output_directory);
        }
    }
}

inline std::unordered_map<std::type_index, std::unique_ptr<SensorMatrics>>&
Observer::sensor_matrics() noexcept {
    return _sensor_matrics;
}

inline const std::unordered_map<std::type_index, std::unique_ptr<SensorMatrics>>&
Observer::sensor_matrics() const noexcept {
    return _sensor_matrics;
}

inline Observer::Builder&
Observer::Builder::with_source_sensor_matrics(const std::size_t reserve_count) noexcept {
    _with_source_sensor_matrics = true;
    _source_reserve_count = reserve_count;
    return *this;
}

inline Observer::Builder&
Observer::Builder::with_sink_sensor_matrics(const std::size_t reserve_count) noexcept {
    _with_sink_sensor_matrics = true;
    _sink_reserve_count = reserve_count;
    return *this;
}

inline Observer
Observer::Builder::build() const {
    Observer observer;

    if (_with_source_sensor_matrics) {
        observer.emplace_sensor_matrics<SourceSensorMatrics>(_source_reserve_count);
    }

    if (_with_sink_sensor_matrics) {
        observer.emplace_sensor_matrics<SinkSensorMatrics>(_sink_reserve_count);
    }

    return observer;
}

inline atlas::host_shared_ptr<Observer>
Observer::Builder::make_host_shared() const {
    auto observer = build();
    return atlas::make_host_shared<Observer>(std::move(observer));
}

} // namespace atlas::observer
