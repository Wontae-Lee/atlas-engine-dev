#pragma once

#include <atlas/container/type_store.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/sensor_matrics.h>

#include <cstddef>
#include <filesystem>
#include <memory>

namespace atlas {

using SensorMatricsStore = TypeStore<SensorMatrics>;

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

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    template <typename SensorMatricsT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE SensorMatricsT&
    emplace_sensor_matrics(Args&&... args);

    template <typename SensorMatricsT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_sensor_matrics(std::unique_ptr<SensorMatricsT> sensor_matrics);

    template <typename SensorMatricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SensorMatricsT*
    sensor_matrics() noexcept;

    template <typename SensorMatricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SensorMatricsT*
    sensor_matrics() const noexcept;

    template <typename SensorMatricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_sensor_matrics() const noexcept;

    template <typename SensorMatricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<SensorMatricsT>
    remove_sensor_matrics();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    export_csv(const std::filesystem::path& output_directory) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SensorMatricsStore&
    sensor_matrics() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SensorMatricsStore&
    sensor_matrics() const noexcept;

private:
    SensorMatricsStore _sensor_matrics;
};

class Observer::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_source_sensor_matrics(std::size_t reserve_count = 0) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sink_sensor_matrics(std::size_t reserve_count = 0) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Observer
    build() const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::host_shared_ptr<Observer>
    make_host_shared() const;

private:
    bool _with_source_sensor_matrics  = false;
    bool _with_sink_sensor_matrics    = false;
    std::size_t _source_reserve_count = 0;
    std::size_t _sink_reserve_count   = 0;
};

}

namespace atlas {
using ObserverHostPtr   = host_shared_ptr<Observer>;
using ObserverDevicePtr = device_shared_ptr<Observer>;

}

#include <atlas/observer/observer.hpp>