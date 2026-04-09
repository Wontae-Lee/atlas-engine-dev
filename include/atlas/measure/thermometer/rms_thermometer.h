#pragma once

#include <atlas/measure/thermometer.h>

namespace atlas::system {

template <typename T>
class RmsThermometer final : public Thermometer<T> {
public:
    class Builder;

public:
    RmsThermometer() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit RmsThermometer(
        const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit RmsThermometer(
        const atlas::system::RmsThermometerOperator<T>& thermometer_operator) noexcept;

    ~RmsThermometer() override = default;

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
    atlas::system::ThermometerOperator<T> _thermometer_operator { ThermometerType::Rms };
};

template <typename T>
class RmsThermometer<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_thermometer_operator(const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_thermometer_operator(const atlas::system::RmsThermometerOperator<T>& thermometer_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE RmsThermometer<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<RmsThermometer<T>>
    make_host_shared() const;

private:
    atlas::system::ThermometerOperator<T> _thermometer_operator { ThermometerType::Rms };
};

}

namespace atlas {

template <typename T>
using RmsThermometer = atlas::system::RmsThermometer<T>;

template <typename T>
using RmsThermometerHostPtr = atlas::host_shared_ptr<atlas::system::RmsThermometer<T>>;

template <typename T>
using RmsThermometerDevicePtr = atlas::device_shared_ptr<atlas::system::RmsThermometer<T>>;

}

#include <atlas/measure/thermometer/rms_thermometer.hpp>
