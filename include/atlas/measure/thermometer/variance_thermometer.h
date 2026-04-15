#pragma once

#include <atlas/measure/thermometer.h>
#include <atlas/measure/thermometer/thermometer_operator.h>

namespace atlas::system {

template <typename T>
class VarianceThermometer final : public Thermometer<T> {
public:
    class Builder;

public:
    VarianceThermometer() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit VarianceThermometer(
        const atlas::system::ThermometerOperator<T>& thermometer_operator,
        MeasureModeType measure_mode = MeasureModeType::All) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit VarianceThermometer(
        const atlas::system::VarianceThermometerOperator<T>& thermometer_operator,
        MeasureModeType measure_mode = MeasureModeType::All) noexcept;

    ~VarianceThermometer() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(Universe<T>& domain, SpatialHashingProbe<T> searcher, FluidDeviceProbe<T> particle)
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
class VarianceThermometer<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measure_mode(MeasureModeType measure_mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_thermometer_operator(const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_thermometer_operator(const atlas::system::VarianceThermometerOperator<T>& thermometer_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE VarianceThermometer<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<VarianceThermometer<T>>
    make_host_shared() const;

private:
    MeasureModeType _measure_mode { MeasureModeType::All };

    atlas::system::ThermometerOperator<T> _thermometer_operator {};
};
}

namespace atlas {

template <typename T>
using VarianceThermometer = atlas::system::VarianceThermometer<T>;

template <typename T>
using VarianceThermometerHostPtr = atlas::host_shared_ptr<atlas::system::VarianceThermometer<T>>;

template <typename T>
using VarianceThermometerDevicePtr = atlas::device_shared_ptr<atlas::system::VarianceThermometer<T>>;

}

#include <atlas/measure/thermometer/variance_thermometer.hpp>