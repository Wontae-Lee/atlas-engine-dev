#pragma once

#include <atlas/container/type_store.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/sensor_metrics.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <utility>

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

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    template <typename SensorMetricsT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE SensorMetricsT&
    emplace_sensor_metrics(Args&&... args) {
        return _sensor_metrics.template emplace<SensorMetricsT>(std::forward<Args>(args)...);
    }

    template <typename SensorMetricsT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_sensor_metrics(std::unique_ptr<SensorMetricsT> sensor_metrics) {
        _sensor_metrics.template set<SensorMetricsT>(std::move(sensor_metrics));
    }

    template <typename SensorMetricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SensorMetricsT*
    sensor_metrics() noexcept {
        return _sensor_metrics.template get<SensorMetricsT>();
    }

    template <typename SensorMetricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SensorMetricsT*
    sensor_metrics() const noexcept {
        return _sensor_metrics.template get<SensorMetricsT>();
    }

    template <typename SensorMetricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_sensor_metrics() const noexcept {
        return _sensor_metrics.template contains<SensorMetricsT>();
    }

    template <typename SensorMetricsT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<SensorMetricsT>
    remove_sensor_metrics() {
        return _sensor_metrics.template remove<SensorMetricsT>();
    }

    ATLAS_HOST void
    export_csv(const std::filesystem::path& output_directory) const;

    ATLAS_HOST ATLAS_NODISCARD SensorMetricsStore&
    sensor_metrics() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const SensorMetricsStore&
    sensor_metrics() const noexcept;

private:
    SensorMetricsStore _sensor_metrics;
};

class Observer::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_source_sensor_metrics(std::size_t reserve_count = 0) noexcept;

    ATLAS_HOST Builder&
    with_sink_sensor_metrics(std::size_t reserve_count = 0) noexcept;

    ATLAS_HOST ATLAS_NODISCARD Observer
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<Observer>
    make_host_shared() const;

private:
    bool _with_source_sensor_metrics  = false;
    bool _with_sink_sensor_metrics    = false;
    std::size_t _source_reserve_count = 0;
    std::size_t _sink_reserve_count   = 0;
};

using ObserverHostPtr = host_shared_ptr<Observer>;

using ObserverDevicePtr = device_shared_ptr<Observer>;

}
