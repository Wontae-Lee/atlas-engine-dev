#pragma once

#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measure.h>
#include <atlas/measure/thermometer/thermometer_type.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

namespace atlas::system {

template <typename T>
struct VarianceThermometerOperator final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(const DomainDeviceProbe<T>& domain,
            const SpatialHashingProbe<T>& searcher,
            const FluidDeviceProbe<T>& particle,
            MeasureModeType measure_mode = MeasureModeType::All) const;
};

template <typename T>
struct AverageThermometerOperator final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(const DomainDeviceProbe<T>& domain,
            const SpatialHashingProbe<T>& searcher,
            const FluidDeviceProbe<T>& particle,
            MeasureModeType measure_mode = MeasureModeType::All) const;
};

template <typename T>
struct ThermometerOperator final {
    ThermometerType type = ThermometerType::Average;

    union {
        VarianceThermometerOperator<T> variance;
        AverageThermometerOperator<T> average;
    };

    ATLAS_HOST ATLAS_FORCE_INLINE
    ThermometerOperator() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit ThermometerOperator(ThermometerType type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    ThermometerOperator(const ThermometerOperator& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE ThermometerOperator&
    operator=(const ThermometerOperator& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE ~ThermometerOperator() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit ThermometerOperator(const VarianceThermometerOperator<T>& op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit ThermometerOperator(const AverageThermometerOperator<T>& op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(const DomainDeviceProbe<T>& domain,
            const SpatialHashingProbe<T>& searcher,
            const FluidDeviceProbe<T>& particle,
            MeasureModeType measure_mode = MeasureModeType::All) const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    copy_from(const ThermometerOperator& other) noexcept;
};

}

#include <atlas/measure/thermometer/thermometer_operator.hpp>
