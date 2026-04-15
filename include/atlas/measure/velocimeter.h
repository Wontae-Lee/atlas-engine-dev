#pragma once

#include <atlas/core/macros.h>
#include <atlas/measure/measure.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class Velocimeter : public Measure<T> {
    static_assert(std::is_floating_point_v<T>, "Velocimeter requires a floating-point T");

public:
    Velocimeter() = default;
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Velocimeter(MeasureModeType measure_mode) noexcept;
    ~Velocimeter() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(Universe<T>& domain,
            SpatialHashingProbe<T> searcher,
            FluidDeviceProbe<T> particle) override
        = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_valid() const noexcept
        = 0;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_measure_mode(MeasureModeType measure_mode) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE MeasureModeType
    measure_mode() const noexcept override;

protected:
    MeasureModeType _measure_mode { MeasureModeType::All };
};

}

namespace atlas {

template <typename T>
using Velocimeter = atlas::system::Velocimeter<T>;

template <typename T>
using VelocimeterHostPtr = atlas::host_shared_ptr<atlas::system::Velocimeter<T>>;

template <typename T>
using VelocimeterDevicePtr = atlas::device_shared_ptr<atlas::system::Velocimeter<T>>;

}

#include <atlas/measure/velocimeter.hpp>