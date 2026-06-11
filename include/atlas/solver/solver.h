#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

/**
 * @brief Base interface for simulation solvers executed by the system orchestrator.
 *
 * @tparam T Floating-point scalar type used by the simulation, such as `float` or `double`.
 *
 * @details
 * `Solver` stores shared access to the simulation universe, fluid storage, and
 * particle searcher used by concrete solver implementations. Derived
 * solvers override one or both `solve()` overloads to update particle or
 * universe state for a simulation time step.
 *
 * The allocation-filtered overload supports orchestrators that assign different
 * cells to different solvers. A solver that does not use allocation filtering can
 * implement the all-cells overload or forward it to the filtered overload with a
 * null allocation buffer.
 *
 * @note
 * The default implementations intentionally do no work. They allow optional
 * solver slots and delayed specialization without requiring a concrete no-op
 * solver type.
 */
template <typename T>
class Solver {
public:
    /**
     * @brief Creates an empty solver.
     *
     * @details
     * The default-constructed solver has no universe, fluid, or searcher
     * references. It is mainly useful for delayed initialization or containers
     * that require default construction.
     */
    Solver() = default;

    /**
     * @brief Constructs a solver with shared simulation objects.
     *
     * @param universe Shared host pointer to the simulation universe.
     * @param fluid Shared host pointer to the particle fluid storage.
     * @param searcher Shared host pointer to the particle searcher.
     *
     * @details
     * The pointers are stored for derived solver implementations. The base class
     * does not validate ownership or initialize solver-specific state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Solver(UniverseHostPtr<T> universe,
           FluidHostPtr<T> fluid,
           SearcherHostPtr<T> searcher) noexcept
        : _universe(std::move(universe))
        , _fluid(std::move(fluid))
        , _searcher(std::move(searcher)) { }

    /**
     * @brief Destroys the solver through the base interface.
     */
    virtual ~Solver() = default;

    /**
     * @brief Advances solver state for all particles or cells.
     *
     * @param dt Simulation time step.
     *
     * @details
     * Concrete solvers override this overload when they process the whole domain
     * without per-cell solver allocation. The base implementation is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    solve(const T dt) { }

    /**
     * @brief Advances solver state for cells assigned to this solver.
     *
     * @param allocated_solver Optional per-cell solver assignment buffer.
     * @param index Solver index used to select cells from `allocated_solver`.
     * @param dt Simulation time step.
     *
     * @details
     * Concrete solvers override this overload when they support orchestrator-side
     * cell allocation. If `allocated_solver` is null, derived implementations
     * typically process all cells. The base implementation is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    solve(const DeviceBuffer<int>* allocated_solver, const int index, const T dt) { }

protected:
    /**
     * @brief Shared simulation universe containing domain and per-cell states.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Shared particle fluid storage and material state.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Shared particle searcher used for neighborhood and cell access.
     */
    SearcherHostPtr<T> _searcher {};
};

}

namespace atlas {

/**
 * @brief Public alias for the system-level solver base interface.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Solve = atlas::system::Solver<T>;

/**
 * @brief Host shared-pointer alias for `atlas::system::Solver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SolveHostPtr = atlas::host_shared_ptr<atlas::system::Solver<T>>;

/**
 * @brief Device shared-pointer alias for `atlas::system::Solver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SolveDevicePtr = atlas::device_shared_ptr<atlas::system::Solver<T>>;

}
