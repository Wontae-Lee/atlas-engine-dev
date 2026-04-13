#pragma once

#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measure.h>
#include <atlas/measure/thermometer/thermometer_type.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
struct ThermometerOperator;

template <typename T>
class Thermometer : public Measure<T> {
    static_assert(std::is_floating_point_v<T>, "Thermometer requires a floating-point T");

public:
    Thermometer() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Thermometer(MeasureModeType measure_mode) noexcept
        : _measure_mode(measure_mode) {
    }

    ~Thermometer() override = default;


    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(DomainDeviceProbe<T> domain, SpatialHashingProbe<T> searcher, FluidDeviceProbe<T> particle)
        override = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_valid() const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual ThermometerType
    type() const noexcept = 0;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_measure_mode(MeasureModeType measure_mode) noexcept {
        _measure_mode = measure_mode;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE MeasureModeType
    measure_mode() const noexcept override {
        return _measure_mode;
    }

private:
    MeasureModeType _measure_mode { MeasureModeType::All };
};

}

namespace atlas {

template <typename T>
using ThermometerOperator = atlas::system::ThermometerOperator<T>;

template <typename T>
using Thermometer = atlas::system::Thermometer<T>;

template <typename T>
using ThermometerHostPtr = atlas::host_shared_ptr<atlas::system::Thermometer<T>>;

template <typename T>
using ThermometerDevicePtr = atlas::device_shared_ptr<atlas::system::Thermometer<T>>;

}
