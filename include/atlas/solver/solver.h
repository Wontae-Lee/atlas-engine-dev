#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher_view.h>
#include <atlas/solver/solver_type.h>
#include <atlas/universe/universe.h>

namespace atlas {

class Solver {
public:
    Solver() = default;

    Solver(const Solver&) = default;

    Solver(Solver&&) noexcept = default;

    virtual ~Solver() = default;

    Solver&
    operator=(const Solver&)
        = default;

    Solver&
    operator=(Solver&&) noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST virtual SolverType
    type() const noexcept = 0;

    ATLAS_HOST virtual void
    solve(Fluid& fluid,
          Universe& universe,
          const SpatialHashingSearcherView& searcher_view,
          int index,
          float dt)
        = 0;
};

using SolverHostPtr = atlas::host_shared_ptr<Solver>;

using SolverDevicePtr = atlas::device_shared_ptr<Solver>;

}