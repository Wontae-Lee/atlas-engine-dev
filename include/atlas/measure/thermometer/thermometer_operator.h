#pragma once

#include <atlas/measure/thermometer/average_thermometer.h>
#include <atlas/measure/thermometer/thermometer_type.h>
#include <atlas/measure/thermometer/variance_thermometer.h>

namespace atlas::system {

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
