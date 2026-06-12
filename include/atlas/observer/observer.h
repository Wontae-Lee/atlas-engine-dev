#pragma once

/**
 * @file observer.h
 * @brief Declares the Observer class used to own and export heterogeneous simulation metrics.
 */

#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/sensor_matrics.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

namespace atlas {

/**
 * @brief Runtime registry of heterogeneous metric containers.
 *
 * Observer mirrors the state-registry pattern used by Fluid and Universe:
 * metric sets are stored by exact concrete type and can be queried or exported
 * collectively.
 */
class Observer final {
public:
    class Builder;

public:
    Observer()                    = default;
    Observer(const Observer&)     = delete;
    Observer(Observer&&) noexcept = default;
    ~Observer()                   = default;

    Observer&
    operator=(const Observer&)
        = delete;

    Observer&
    operator=(Observer&&) noexcept = default;

    /**
     * @brief Creates a Builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Constructs and registers a metric container in place.
     */
    template <typename SensorMatricsT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE SensorMatricsT&
    emplace_sensor_matrics(Args&&... args);

    /**
     * @brief Registers or replaces a metric container.
     */
    template <typename SensorMatricsT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_sensor_matrics(std::unique_ptr<SensorMatricsT> sensor_matrics);

    /**
     * @brief Returns mutable access to a stored metric container.
     */
    template <typename SensorMatricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SensorMatricsT*
    sensor_matrics() noexcept;

    /**
     * @brief Returns read-only access to a stored metric container.
     */
    template <typename SensorMatricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SensorMatricsT*
    sensor_matrics() const noexcept;

    /**
     * @brief Returns whether a metric container is registered.
     */
    template <typename SensorMatricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_sensor_matrics() const noexcept;

    /**
     * @brief Removes and returns a metric container.
     */
    template <typename SensorMatricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<SensorMatricsT>
    remove_sensor_matrics();

    /**
     * @brief Exports every registered metric container as CSV.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    export_csv(const std::filesystem::path& output_directory) const;

    /**
     * @brief Returns the full mutable metric registry.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unordered_map<std::type_index, std::unique_ptr<SensorMatrics>>&
    sensor_matrics() noexcept;

    /**
     * @brief Returns the full read-only metric registry.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::unordered_map<std::type_index, std::unique_ptr<SensorMatrics>>&
    sensor_matrics() const noexcept;

private:
    std::unordered_map<std::type_index, std::unique_ptr<SensorMatrics>> _sensor_matrics;
};

/**
 * @brief Builder for Observer.
 */
class Observer::Builder final {
public:
    Builder() = default;

    /**
     * @brief Enables source metrics and optionally reserves space for records.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_source_sensor_matrics(std::size_t reserve_count = 0) noexcept;

    /**
     * @brief Enables sink metrics and optionally reserves space for records.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sink_sensor_matrics(std::size_t reserve_count = 0) noexcept;

    /**
     * @brief Builds an Observer value.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Observer
    build() const;

    /**
     * @brief Builds an Observer in host-managed shared storage.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::host_shared_ptr<Observer>
    make_host_shared() const;

private:
    bool _with_source_sensor_matrics  = false;
    bool _with_sink_sensor_matrics    = false;
    std::size_t _source_reserve_count = 0;
    std::size_t _sink_reserve_count   = 0;
};

} // namespace atlas

namespace atlas {
using ObserverHostPtr   = host_shared_ptr<Observer>;
using ObserverDevicePtr = device_shared_ptr<Observer>;

} // namespace atlas

#include <atlas/observer/observer.hpp>
