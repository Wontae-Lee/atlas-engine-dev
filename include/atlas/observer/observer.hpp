#pragma once

#include <ranges>
#include <stdexcept>
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
    // Ensure only SensorMatrics-derived types can be stored in the observer.
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::SensorMatrics.");

    // Construct the requested sensor matrics type with forwarded arguments.
    auto sensor_matrics = std::make_unique<SensorMatricsT>(std::forward<Args>(args)...);

    // Keep a raw pointer before moving ownership into the type-indexed container.
    auto* ptr = sensor_matrics.get();

    // Store or replace the sensor matrics instance associated with its concrete type.
    _sensor_matrics.insert_or_assign(typeid(SensorMatricsT), std::move(sensor_matrics));

    // Return a reference to the stored instance for immediate configuration or use.
    return *ptr;
}

template <typename SensorMatricsT>
void
Observer::set_sensor_matrics(std::unique_ptr<SensorMatricsT> sensor_matrics) {
    // Ensure only SensorMatrics-derived types can be stored in the observer.
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::SensorMatrics.");

    // Reject null ownership transfer because the observer must store valid matrics objects.
    if (sensor_matrics == nullptr) {
        throw std::invalid_argument("Observer::set_sensor_matrics failed: sensor_matrics must not be null.");
    }

    // Store or replace the sensor matrics instance associated with its concrete type.
    _sensor_matrics.insert_or_assign(typeid(SensorMatricsT), std::move(sensor_matrics));
}

template <typename SensorMatricsT>
SensorMatricsT*
Observer::sensor_matrics() noexcept {
    // Ensure callers request a valid SensorMatrics-derived type.
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::SensorMatrics.");

    // Look up the stored matrics object by its concrete runtime type.
    const auto it = _sensor_matrics.find(typeid(SensorMatricsT));

    // Return nullptr when the requested matrics type has not been registered.
    return it == _sensor_matrics.end() ? nullptr : static_cast<SensorMatricsT*>(it->second.get());
}

template <typename SensorMatricsT>
const SensorMatricsT*
Observer::sensor_matrics() const noexcept {
    // Ensure callers request a valid SensorMatrics-derived type.
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::SensorMatrics.");

    // Look up the stored matrics object by its concrete runtime type.
    const auto it = _sensor_matrics.find(typeid(SensorMatricsT));

    // Return nullptr when the requested matrics type has not been registered.
    return it == _sensor_matrics.end() ? nullptr : static_cast<const SensorMatricsT*>(it->second.get());
}

template <typename SensorMatricsT>
bool
Observer::has_sensor_matrics() const noexcept {
    // Ensure callers query a valid SensorMatrics-derived type.
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::SensorMatrics.");

    // Check whether a matrics instance of the requested concrete type is registered.
    return _sensor_matrics.contains(typeid(SensorMatricsT));
}

template <typename SensorMatricsT>
std::unique_ptr<SensorMatricsT>
Observer::remove_sensor_matrics() {
    // Ensure only SensorMatrics-derived types can be removed through this API.
    static_assert(std::is_base_of_v<SensorMatrics, SensorMatricsT>,
                  "SensorMatricsT must derive from atlas::SensorMatrics.");

    // Locate the matrics object associated with the requested concrete type.
    const auto it = _sensor_matrics.find(typeid(SensorMatricsT));

    // Return null ownership when no matching matrics object exists.
    if (it == _sensor_matrics.end()) {
        return nullptr;
    }

    // Release ownership from the type-erased base pointer and restore the concrete unique_ptr type.
    auto sensor_matrics = std::unique_ptr<SensorMatricsT>(static_cast<SensorMatricsT*>(it->second.release()));

    // Remove the now-empty map entry after ownership has been transferred out.
    _sensor_matrics.erase(it);

    return sensor_matrics;
}

inline void
Observer::export_csv(const std::filesystem::path& output_directory) const {
    // Export every registered sensor matrics object into the requested output directory.
    for (const auto& value : _sensor_matrics | std::views::values) {
        if (value) {
            // Delegate CSV export to the concrete matrics implementation.
            value->export_csv(output_directory);
        }
    }
}

inline std::unordered_map<std::type_index, std::unique_ptr<SensorMatrics>>&
Observer::sensor_matrics() noexcept {
    // Expose the mutable type-indexed matrics container for advanced management.
    return _sensor_matrics;
}

inline const std::unordered_map<std::type_index, std::unique_ptr<SensorMatrics>>&
Observer::sensor_matrics() const noexcept {
    // Expose the read-only type-indexed matrics container.
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