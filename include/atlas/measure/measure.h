#pragma once

#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

namespace atlas::system {

enum class MeasureModeType : int {
    Field,
    Fluid,
    All
};

template <typename T>
class Measure {
public:
    Measure()          = default;
    virtual ~Measure() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    measure(DomainDeviceProbe<T> domain, SpatialHashingProbe<T> searcher, FluidDeviceProbe<T> particle)
        = 0;
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
