#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec_probe.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

namespace atlas {

class Codec {
public:
    Codec() = default;

    ATLAS_HOST
    Codec(UniverseHostPtr domain,
          FluidHostPtr fluid,
          SearcherHostPtr searcher);

    virtual ~Codec() = default;

    Codec(const Codec&) = default;
    Codec&
    operator=(const Codec&)
        = default;
    Codec(Codec&&) noexcept = default;
    Codec&
    operator=(Codec&&) noexcept = default;

    ATLAS_HOST virtual void
    update();

    ATLAS_HOST virtual void
    encode()
        = 0;

    ATLAS_HOST virtual void
    decode()
        = 0;

    ATLAS_HOST void
    reset() noexcept;

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    allocated_solver() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    allocated_solver() const noexcept;

    ATLAS_HOST void
    set_fixed_solver(DeviceBuffer<int> fixed_solver);

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    fixed_solver() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    fixed_solver() const noexcept;

    ATLAS_HOST void
    set_fixed_region(DeviceBuffer<int> fixed_region);

    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    fixed_region() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    fixed_region() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe() noexcept;

protected:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    DeviceBuffer<int> d_allocated_solver;

    DeviceBuffer<int> d_fixed_solver;

    DeviceBuffer<int> d_fixed_region;

    CodecProbe _probe {};
};

using CodecHostPtr = atlas::host_shared_ptr<Codec>;

using CodecDevicePtr = atlas::device_shared_ptr<Codec>;

}
