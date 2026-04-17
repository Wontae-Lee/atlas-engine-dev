#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

template <typename T>
class Solver {
public:
    Solver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Solver(UniverseHostPtr<T> universe,
           FluidHostPtr<T> fluid,
           SpatialHashingSearcherHostPtr<T> searcher) noexcept
        : _universe(std::move(universe))
        , _fluid(std::move(fluid))
        , _searcher(std::move(searcher)) { }

    virtual ~Solver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    solve(const T dt) { }

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    solve(const DeviceBuffer<int>* allocated_solver, const int index, const T dt) { }

protected:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SpatialHashingSearcherHostPtr<T> _searcher {};
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
