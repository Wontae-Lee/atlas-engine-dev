#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

namespace atlas {

class Solver {
public:
    Solver() = default;

    ATLAS_HOST
    Solver(UniverseHostPtr universe, FluidHostPtr fluid, SearcherHostPtr searcher) noexcept;

    virtual ~Solver() = default;

    Solver(const Solver&) = default;
    Solver&
    operator=(const Solver&)
        = default;
    Solver(Solver&&) noexcept = default;
    Solver&
    operator=(Solver&&) noexcept = default;

    ATLAS_HOST virtual void
    solve(float dt);

    ATLAS_HOST virtual void
    solve(const DeviceBuffer<int>* allocated_solver, int index, float dt);

protected:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};
};

using Solve = atlas::Solver;

using SolveHostPtr = atlas::host_shared_ptr<atlas::Solver>;

using SolveDevicePtr = atlas::device_shared_ptr<atlas::Solver>;

}
