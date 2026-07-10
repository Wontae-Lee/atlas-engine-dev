#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher_view.h>
#include <atlas/solver/solver_type.h>
#include <atlas/universe/universe.h>

namespace atlas {

/**
 * @brief Abstract host-side interface for a per-step particle-interaction solver.
 *
 * A Solver advances one physical interaction (currently only DSMC collisions, see
 * @ref DsmcSolver) over the fluid, in place, once per System step. Unlike the engine's
 * @c DeviceVariant / @c HostVariant leaves, solvers are heavyweight, own device scratch
 * buffers, and are dispatched dynamically: the System holds a list of
 * @ref SolverHostPtr and calls @ref solve on each, so classic virtual dispatch is used
 * rather than a tagged union. @ref type lets a caller recover the concrete kind without
 * RTTI.
 *
 * The class is an interface only: it stores no state and its special members are
 * defaulted so derived classes remain freely copyable/movable. It is used on the host;
 * the actual per-particle work is launched onto the device from inside @ref solve.
 */
class Solver {
public:
    /** @brief Default-constructs an empty solver base; derived state is set up by the subclass. */
    Solver() = default;

    /** @brief Defaulted copy constructor; the base carries no state to copy. */
    Solver(const Solver&) = default;

    /** @brief Defaulted move constructor; @c noexcept so containers can relocate solvers. */
    Solver(Solver&&) noexcept = default;

    /** @brief Virtual destructor so a @ref SolverHostPtr to the base destroys the derived solver. */
    virtual ~Solver() = default;

    /** @brief Defaulted copy assignment; the base carries no state to assign. */
    Solver&
    operator=(const Solver&)
        = default;

    /** @brief Defaulted move assignment; @c noexcept counterpart to the move constructor. */
    Solver&
    operator=(Solver&&) noexcept = default;

    /**
     * @brief Reports the concrete solver kind behind this base pointer.
     *
     * Lets the System branch on the dynamic type without RTTI (e.g. when deciding which
     * per-cell state a solver needs). Pure virtual — every leaf must return its own tag.
     *
     * @return The @ref SolverType enumerator naming this solver's concrete class.
     */
    ATLAS_NODISCARD ATLAS_HOST virtual SolverType
    type() const noexcept = 0;

    /**
     * @brief Advances this solver's interaction over the fluid by one sub-step, in place.
     *
     * Called once per System step for each registered solver. The implementation reads
     * the current particle state through views gathered on the host and mutates the
     * fluid (typically velocities) via device kernels. Pure virtual.
     *
     * @param fluid         The particle population to read and update in place.
     * @param universe      The background grid supplying per-cell state (occupancy,
     *                      collision counters, ownership) the solver reads and writes.
     * @param searcher_view Device-capturable snapshot of the spatial hash: the sorted
     *                      particle order and per-cell `[start, end)` ranges. Must have
     *                      been classified for this step; solvers skip work if it is null.
     * @param index         This solver's position in the System's solver list, used to
     *                      claim only the cells the codec assigned to it
     *                      (`allocated_solver[cell] == index`); a null ownership buffer
     *                      means every solver owns every cell.
     * @param dt            Sub-step duration in seconds; must be positive to have effect.
     */
    ATLAS_HOST virtual void
    solve(Fluid& fluid,
          Universe& universe,
          const SpatialHashingSearcherView& searcher_view,
          int index,
          float dt)
        = 0;
};

/** @brief Shared-ownership host handle to a polymorphic @ref Solver; the System's storage type. */
using SolverHostPtr = atlas::host_shared_ptr<Solver>;

/** @brief Device-resident shared handle to a @ref Solver; provided for symmetry with other modules. */
using SolverDevicePtr = atlas::device_shared_ptr<Solver>;

}