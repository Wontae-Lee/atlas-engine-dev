#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

enum class MeasureModeType : int {
    Field,
    Fluid,
    All
};

template <typename T>
class Measure {
public:
    Measure() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Measure(UniverseHostPtr<T> universe,
            FluidHostPtr<T> fluid,
            SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    virtual ~Measure() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    measure()
        = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual MeasureModeType
    measure_mode() const noexcept = 0;

protected:
    UniverseHostPtr<T> _universe {};
    FluidHostPtr<T> _fluid {};
    SpatialHashingSearcherHostPtr<T> _searcher {};

};

}

namespace atlas {

using MeasureModeType = atlas::system::MeasureModeType;

template <typename T>
using Measure = atlas::system::Measure<T>;

template <typename T>
using MeasureHostPtr = atlas::host_shared_ptr<atlas::system::Measure<T>>;

template <typename T>
using MeasureDevicePtr = atlas::device_shared_ptr<atlas::system::Measure<T>>;
}

#include <atlas/measure/measure.hpp>
