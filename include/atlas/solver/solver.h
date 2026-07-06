#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

/**
 * @file solver.h
 * @brief Host-only base interface every physics solver (`DsmcSolver`,
 *        `SphSolver`, ...) implements, plus the
 *        `allocated_solver` mechanism that lets several solver instances
 *        share one universe by cell.
 *
 * @details
 * ### Background — per-cell solver allocation
 * A single flow field can span multiple physical regimes: near-continuum
 * regions are cheaper and more accurate to advance with a continuum-style
 * solver (e.g. `SphSolver`), while rarefied regions need particle-based
 * DSMC (`DsmcSolver`). Atlas supports this by letting a `Codec` (see
 * `codec/codec.h`, e.g. `KnudsenCodec`) classify each cell by a local
 * criterion — Knudsen number for `KnudsenCodec` — into a small integer
 * *solver index* per cell (`Codec::allocated_solver()`, a
 * `DeviceBuffer<int>` parallel to the universe's cells). Several `Solver`
 * instances then share the same `Universe`/`Fluid`/`Searcher`, each
 * called with that shared `allocated_solver` buffer and its own `index`;
 * `solve(allocated_solver, index, dt)` restricts a solver to only the
 * cells where `allocated_solver[cell] == index`, leaving every other
 * cell's particles untouched for another solver's turn.
 * `solve(dt)` (no `allocated_solver`) is the simple case: a solver owning
 * its universe outright, with every cell implicitly allocated to it.
 *
 * ### Operating principle
 * `Solver` itself is intentionally inert — its `solve` overloads are
 * empty ATLAS_HOST virtuals (not pure), so it establishes an interface
 * without forcing every derived class to implement both overloads
 * meaningfully (a solver that never runs in a multi-solver universe can
 * simply not override `solve(allocated_solver, index, dt)`). It owns the
 * three handles every concrete solver needs — `_universe`/`_fluid`/
 * `_searcher` — as shared, non-owning-by-value host pointers, so several
 * solvers (and other systems, e.g. `Collider`) can reference the same
 * live objects.
 */

namespace atlas {

/**
 * @brief Base interface for a physics solver operating on one
 *        `Fluid`/`Universe`/`Searcher` triple. See this file's
 *        top-of-file documentation for the `allocated_solver` per-cell
 *        multi-solver mechanism.
 */
class Solver {
public:
    Solver() = default;

    ATLAS_HOST Solver(UniverseHostPtr universe, FluidHostPtr fluid, SearcherHostPtr searcher) noexcept;

    virtual ~Solver() = default;

    Solver(const Solver&) = default;
    Solver&
    operator=(const Solver&)
        = default;
    Solver(Solver&&) noexcept = default;
    Solver&
    operator=(Solver&&) noexcept = default;

    /** @brief Advances the whole universe by `dt`, assuming this solver
     *  owns every cell. Base implementation is a no-op; overridden by
     *  every concrete solver. */
    ATLAS_HOST virtual void
    solve(float dt);

    /** @brief Advances only the cells allocated to this solver
     *  (`allocated_solver[cell] == index`) by `dt` — see this file's
     *  top-of-file documentation for the per-cell allocation mechanism.
     *  Base implementation is a no-op. */
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
