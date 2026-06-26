#pragma once

#include <atlas/codec/codec_probe.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

namespace atlas {

template <typename T>
class Codec {
public:
    Codec() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Codec(UniverseHostPtr<T> domain,
          FluidHostPtr<T> fluid,
          SearcherHostPtr<T> searcher);

    virtual ~Codec() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    update();

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    encode()
        = 0;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    decode()
        = 0;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
    allocated_solver() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    allocated_solver() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fixed_solver(DeviceBuffer<int> fixed_solver);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
    fixed_solver() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    fixed_solver() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fixed_region(DeviceBuffer<int> fixed_region);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
    fixed_region() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    fixed_region() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

protected:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SearcherHostPtr<T> _searcher {};

    DeviceBuffer<int> d_allocated_solver;

    DeviceBuffer<int> d_fixed_solver;

    DeviceBuffer<int> d_fixed_region;

    CodecProbe<T> _probe {};
};

}

namespace atlas {

template <typename T>
using CodecHostPtr = atlas::host_shared_ptr<Codec<T>>;

template <typename T>
using CodecDevicePtr = atlas::device_shared_ptr<Codec<T>>;

}

#include <atlas/codec/codec.hpp>