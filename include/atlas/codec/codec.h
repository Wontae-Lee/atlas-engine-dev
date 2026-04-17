#pragma once

#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

template <typename T>
class Codec {
public:
    Codec() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Codec(UniverseHostPtr<T> domain,
          FluidHostPtr<T> fluid,
          SpatialHashingSearcherHostPtr<T> searcher);

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

protected:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SpatialHashingSearcherHostPtr<T> _searcher {};

    DeviceBuffer<int> d_allocated_solver;
};

}

namespace atlas {

template <typename T>
using Codec = system::Codec<T>;

template <typename T>
using CodecHostPtr = atlas::host_shared_ptr<system::Codec<T>>;

template <typename T>
using CodecDevicePtr = atlas::device_shared_ptr<system::Codec<T>>;

}

#include <atlas/codec/codec.hpp>
