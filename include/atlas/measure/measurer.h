#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measurer_probe.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

namespace atlas {

enum class MeasureModeType : int {

    field,

    fluid,

    all
};

class Measurer {
public:
    Measurer() = default;

    ATLAS_HOST
    Measurer(UniverseHostPtr universe,
             FluidHostPtr fluid,
             SearcherHostPtr searcher) noexcept;

    virtual ~Measurer() = default;

    Measurer(const Measurer&) = default;
    Measurer&
    operator=(const Measurer&)
        = default;
    Measurer(Measurer&&) noexcept = default;
    Measurer&
    operator=(Measurer&&) noexcept = default;

    ATLAS_HOST virtual void
    measure()
        = 0;

    ATLAS_HOST virtual void
    measure(float dt);

    ATLAS_NODISCARD ATLAS_HOST virtual MeasureModeType
    measure_mode() const noexcept = 0;

    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe() noexcept;

protected:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    MeasurerProbe _probe {};
};

using MeasurerHostPtr = atlas::host_shared_ptr<Measurer>;

using MeasurerDevicePtr = atlas::device_shared_ptr<Measurer>;

}
