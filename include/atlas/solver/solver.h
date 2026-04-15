#pragma once

#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

template <typename T>
class Solver {
public:
    Solver()          = default;
    virtual ~Solver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    solve(Universe<T>& domain,
          SpatialHashingProbe<T>& searcher,
          FluidDeviceProbe<T>& particle,
          CodecDeviceProbe<T>& codec)
        = 0;
};

}

namespace atlas {

template <typename T>
using Solve = atlas::system::Solver<T>;

template <typename T>
using SolveHostPtr = atlas::host_shared_ptr<atlas::system::Solver<T>>;

template <typename T>
using SolveDevicePtr = atlas::device_shared_ptr<atlas::system::Solver<T>>;

}