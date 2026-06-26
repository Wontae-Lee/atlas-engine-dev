#pragma once

#include <atlas/container/type_store.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/sensor_metrics.h>

#include <cstddef>
#include <filesystem>
#include <memory>

namespace atlas {

using SensorMetricsStore = TypeStore<SensorMetrics>;

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

    template <typename SensorMetricsT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE SensorMetricsT&
    emplace_sensor_metrics(Args&&... args);

    template <typename SensorMetricsT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_sensor_metrics(std::unique_ptr<SensorMetricsT> sensor_metrics);

    template <typename SensorMetricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SensorMetricsT*
    sensor_metrics() noexcept;

    template <typename SensorMetricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SensorMetricsT*
    sensor_metrics() const noexcept;

    template <typename SensorMetricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_sensor_metrics() const noexcept;

    template <typename SensorMetricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<SensorMetricsT>
    remove_sensor_metrics();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    export_csv(const std::filesystem::path& output_directory) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SensorMetricsStore&
    sensor_metrics() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SensorMetricsStore&
    sensor_metrics() const noexcept;

private:
    SensorMetricsStore _sensor_metrics;
};

class Observer::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_source_sensor_metrics(std::size_t reserve_count = 0) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sink_sensor_metrics(std::size_t reserve_count = 0) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Observer
    build() const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::host_shared_ptr<Observer>
    make_host_shared() const;

private:
    bool _with_source_sensor_metrics  = false;
    bool _with_sink_sensor_metrics    = false;
    std::size_t _source_reserve_count = 0;
    std::size_t _sink_reserve_count   = 0;
};

}

namespace atlas {
using ObserverHostPtr   = host_shared_ptr<Observer>;
using ObserverDevicePtr = device_shared_ptr<Observer>;

}

#include <atlas/observer/observer.hpp>