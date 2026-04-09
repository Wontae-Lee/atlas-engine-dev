#pragma once

#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

namespace atlas::system {

template <typename T>
class Solver {
public:
    Solver()          = default;
    virtual ~Solver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    solve(DomainDeviceProbe<T> domain,
          SpatialHashingProbe<T> searcher,
          FluidDeviceProbe<T> particle,
          CodecDeviceProbe<T> codec) {
    }
};

}

namespace atlas {

template <typename T>
using Solver = atlas::system::Solver<T>;

template <typename T>
using SolverHostPtr = atlas::host_shared_ptr<atlas::system::Solver<T>>;

template <typename T>
using SolverDevicePtr = atlas::device_shared_ptr<atlas::system::Solver<T>>;

}
