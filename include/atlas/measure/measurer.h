#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measurer_probe.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

namespace atlas {

enum class MeasureModeType : int {

    Field,

    Fluid,

    All
};

template <typename T>
class Measurer {
public:
    Measurer() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Measurer(UniverseHostPtr<T> universe,
             FluidHostPtr<T> fluid,
             SearcherHostPtr<T> searcher) noexcept;

    virtual ~Measurer() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    measure()
        = 0;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    measure(T dt);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual MeasureModeType
    measure_mode() const noexcept = 0;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

protected:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SearcherHostPtr<T> _searcher {};

    MeasurerProbe<T> _probe {};
};

}

namespace atlas {

template <typename T>
using MeasurerHostPtr = atlas::host_shared_ptr<atlas::Measurer<T>>;

template <typename T>
using MeasurerDevicePtr = atlas::device_shared_ptr<atlas::Measurer<T>>;

}

#include <atlas/measure/measurer.hpp>