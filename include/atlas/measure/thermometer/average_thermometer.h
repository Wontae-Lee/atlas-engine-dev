#pragma once

#include <atlas/measure/thermometer.h>

namespace atlas::system {

template <typename T>
class AverageThermometer final : public Thermometer<T> {
public:
    class Builder;

public:
    AverageThermometer() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit AverageThermometer(
        const atlas::system::ThermometerOperator<T>& thermometer_operator,
        MeasureModeType measure_mode = MeasureModeType::All) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit AverageThermometer(
        const atlas::system::AverageThermometerOperator<T>& thermometer_operator,
        MeasureModeType measure_mode = MeasureModeType::All) noexcept;

    ~AverageThermometer() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(DomainDeviceProbe<T> domain, SpatialHashingProbe<T> searcher, FluidDeviceProbe<T> particle)
        override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE ThermometerType
    type() const noexcept override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_thermometer_operator(const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::system::ThermometerOperator<T>&
    thermometer_operator() const noexcept;

private:
    atlas::system::ThermometerOperator<T> _thermometer_operator {};
};

template <typename T>
class AverageThermometer<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measure_mode(MeasureModeType measure_mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_thermometer_operator(const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_thermometer_operator(const atlas::system::AverageThermometerOperator<T>& thermometer_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE AverageThermometer<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<AverageThermometer<T>>
    make_host_shared() const;

private:
    MeasureModeType _measure_mode { MeasureModeType::All };

    atlas::system::ThermometerOperator<T> _thermometer_operator { ThermometerType::Average };
};

}

namespace atlas {

template <typename T>
using AverageThermometer = atlas::system::AverageThermometer<T>;

template <typename T>
using AverageThermometerHostPtr = atlas::host_shared_ptr<atlas::system::AverageThermometer<T>>;

template <typename T>
using AverageThermometerDevicePtr = atlas::device_shared_ptr<atlas::system::AverageThermometer<T>>;

}

#include <atlas/measure/thermometer/average_thermometer.hpp>
