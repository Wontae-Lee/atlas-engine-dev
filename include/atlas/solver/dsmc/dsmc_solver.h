#pragma once

/**
 * @file dsmc_solver.h
 * @brief Declares the abstract DSMC solver base class, collision-kernel binding, and species-pair table utilities.
 *
 * @details
 * This header defines @ref atlas::system::DsmcSolver, an abstract solver base
 * class for Direct Simulation Monte Carlo (DSMC)-style particle dynamics.
 *
 * `DsmcSolver` extends the generic @ref atlas::system::Solver interface and
 * provides common infrastructure needed by concrete DSMC solvers, including:
 * - access to the associated fluid species database,
 * - storage of the active collision kernel/operator,
 * - precomputed effective pairwise material-property tables,
 * - field-force application helpers,
 * - species-pair indexing support.
 *
 * ## Role in the solver hierarchy
 * `DsmcSolver` is an abstract intermediate base:
 * - it is more specialized than @ref atlas::system::Solver,
 * - it is not directly instantiable because @ref solve remains pure virtual,
 * - concrete DSMC implementations derive from it and implement the actual
 *   collision/integration algorithm.
 *
 * ## Pairwise species tables
 * DSMC collision logic often depends on effective pair properties derived from
 * two species, such as:
 * - effective collision diameter,
 * - effective viscosity index,
 * - effective scattering parameter,
 * - reduced mass.
 *
 * To avoid recomputing these values during every collision step, the solver can
 * precompute and cache them in device buffers through @ref rebuild_pair_tables.
 *
 * ## Fluid dependency
 * The associated @ref FluidHostPtr provides:
 * - species material properties,
 * - species count,
 * - runtime context needed to derive pair tables.
 *
 * A valid fluid with configured species is therefore generally required for a
 * usable DSMC solver instance.
 *
 * ## Field-force application
 * In addition to collision processing, the base class provides
 * @ref apply_field_force, a convenience routine that applies domain field forces
 * to particles. This allows concrete DSMC solvers to share common external-force
 * handling logic.
 *
 * ## Collision kernel
 * The active @ref DsmcKernel encapsulates the low-level collision model or
 * kernel used by the DSMC solver. The base class stores and exposes this kernel,
 * but leaves the orchestration of collision steps to derived classes.
 *
 * ## Host/device split
 * The orchestration functions in this class are host-side entry points, while
 * the cached tables they manage are stored in device buffers for efficient use
 * by backend computation paths.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the solver, runtime probes, and material parameters.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/indexer/device_pair_indexer.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/solver.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Abstract base class for DSMC solvers.
 *
 * @details
 * `DsmcSolver` provides shared state and utilities for concrete Direct
 * Simulation Monte Carlo solvers.
 *
 * It stores:
 * - a host-side fluid dependency describing species properties,
 * - a collision kernel used for pairwise interaction logic,
 * - device-resident lookup tables for pairwise effective properties,
 * - a pair-indexing helper for compact species-pair addressing,
 * - the current species count inferred from the fluid.
 *
 * ## Responsibilities of the base class
 * The base class is responsible for:
 * - storing and exposing the configured fluid and collision kernel,
 * - rebuilding effective pair-property tables when fluid/species data changes,
 * - applying external field forces from the domain to particles,
 * - providing common data needed by derived DSMC implementations.
 *
 * ## Responsibilities of derived classes
 * Concrete derived classes must implement @ref solve, which defines the actual
 * DSMC step logic, such as:
 * - selecting collision candidates,
 * - evaluating collision probabilities,
 * - updating particle states,
 * - integrating with codec or auxiliary runtime stages.
 *
 * ## Pair-table layout
 * The pairwise material-property tables are stored in compact one-dimensional
 * buffers indexed through @ref DevicePairIndexer. This avoids storing a full
 * dense square matrix when only unique unordered species pairs are needed.
 *
 * ## Typical usage
 * A derived solver may:
 * 1. ensure the fluid is configured,
 * 2. call @ref rebuild_pair_tables after species changes,
 * 3. use the cached tables during @ref solve,
 * 4. optionally call @ref apply_field_force as part of the update step.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the simulation.
 */
template <typename T>
class DsmcSolver : public Solver<T> {
    static_assert(std::is_floating_point_v<T>, "DsmcSolver requires a floating-point T");

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an empty DSMC solver with:
     * - no associated fluid,
     * - a default-initialized collision kernel,
     * - empty pairwise property tables,
     * - zero species count.
     *
     * A default-constructed solver is typically incomplete until a valid fluid
     * is assigned and pair tables are rebuilt.
     */
    DsmcSolver() = default;

    /**
     * @brief Construct a DSMC solver base from a fluid and collision kernel.
     *
     * @details
     * Initializes the base solver state with:
     * - the supplied fluid dependency,
     * - the supplied DSMC collision kernel,
     * - empty or not-yet-rebuilt pairwise lookup tables.
     *
     * Concrete implementations may call @ref rebuild_pair_tables after
     * construction if immediate pair-table availability is required.
     *
     * @param fluid Host-side fluid dependency providing species data.
     * @param op Active DSMC collision kernel.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSolver(FluidHostPtr<T> fluid,
               DsmcKernel<T> op = DsmcKernel<T> {});

    /**
     * @brief Virtual destructor.
     */
    ~DsmcSolver() override = default;

    /**
     * @brief Perform one DSMC solve step.
     *
     * @details
     * This pure virtual function defines the solver-specific DSMC update logic.
     * Derived classes are expected to use the provided probes and the base-class
     * cached pairwise data to implement particle collision and related updates.
     *
     * ## Inputs
     * - @p domain provides field/grid information and optional external forces,
     * - @p searcher provides neighborhood lookup structures,
     * - @p particle provides particle buffers and counts,
     * - @p codec provides mutable codec-side runtime state.
     *
     * ## Typical derived behavior
     * A derived implementation may:
     * - apply field forces,
     * - traverse cells or neighbor sets,
     * - select collision pairs,
     * - invoke the stored collision kernel,
     * - update particle velocities/temperatures/species state,
     * - consume or update codec state.
     *
     * @param domain Mutable domain probe.
     * @param searcher Mutable spatial hashing probe.
     * @param particle Mutable particle/fluid probe.
     * @param codec Mutable codec probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(DomainDeviceProbe<T>& domain,
          SpatialHashingProbe<T>& searcher,
          FluidDeviceProbe<T>& particle,
          CodecDeviceProbe<T>& codec) override = 0;

    /**
     * @brief Replace the active DSMC collision kernel.
     *
     * @param op New collision kernel.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_operator(DsmcKernel<T> op) noexcept;

    /**
     * @brief Replace the associated fluid dependency.
     *
     * @details
     * After changing the fluid, the cached pairwise property tables may no
     * longer be valid. Call @ref rebuild_pair_tables to refresh them.
     *
     * @param fluid New fluid dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid);

    /**
     * @brief Return const access to the associated fluid dependency.
     *
     * @return Const reference to the stored fluid pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    /**
     * @brief Return const access to the active DSMC collision kernel.
     *
     * @return Const reference to the stored collision kernel.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DsmcKernel<T>&
    collision_operator() const noexcept;

    /**
     * @brief Return mutable access to the active DSMC collision kernel.
     *
     * @return Mutable reference to the stored collision kernel.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernel<T>&
    collision_operator() noexcept;

    /**
     * @brief Apply the domain field force to particle state.
     *
     * @details
     * This helper applies the domain's external field force information to the
     * supplied particle probe.
     *
     * Typical use cases include:
     * - acceleration by a per-cell force field,
     * - shared pre-collision force integration,
     * - common external-force handling across multiple DSMC solver variants.
     *
     * The exact force-integration scheme is implementation-defined in
     * `dsmc_solver.hpp`.
     *
     * @param domain Domain probe containing field-force information.
     * @param particle Particle probe to update.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_field_force(DomainDeviceProbe<T> domain,
                      FluidDeviceProbe<T> particle) const;

    /**
     * @brief Rebuild cached pairwise species-property lookup tables.
     *
     * @details
     * This function derives effective unordered species-pair properties from the
     * associated fluid and stores them in device buffers for efficient runtime use.
     *
     * The rebuilt tables typically include:
     * - @ref _effective_collision_diameters
     * - @ref _effective_viscosity_indices
     * - @ref _effective_scattering_parameters
     * - @ref _reduced_masses
     *
     * ## When to call
     * This function should generally be called whenever:
     * - the associated fluid is first assigned,
     * - the fluid species list changes,
     * - any species material properties relevant to collisions change.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_pair_tables();

protected:
    /**
     * @brief Associated fluid dependency.
     *
     * @details
     * Provides species material properties and species-count information used to
     * build pairwise lookup tables and drive DSMC behavior.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Active DSMC collision kernel.
     *
     * @details
     * Encapsulates the low-level pairwise collision logic used by derived
     * solver implementations.
     */
    DsmcKernel<T> _operator {};

    /**
     * @brief Device-resident effective collision diameters per unique species pair.
     *
     * @details
     * Indexed through @ref _pair_indexer using unordered species indices.
     */
    DeviceBuffer<T> _effective_collision_diameters {};

    /**
     * @brief Device-resident effective viscosity indices per unique species pair.
     *
     * @details
     * Indexed through @ref _pair_indexer using unordered species indices.
     */
    DeviceBuffer<T> _effective_viscosity_indices {};

    /**
     * @brief Device-resident effective scattering parameters per unique species pair.
     *
     * @details
     * Indexed through @ref _pair_indexer using unordered species indices.
     */
    DeviceBuffer<T> _effective_scattering_parameters {};

    /**
     * @brief Device-resident reduced masses per unique species pair.
     *
     * @details
     * Indexed through @ref _pair_indexer using unordered species indices.
     */
    DeviceBuffer<T> _reduced_masses {};

    /**
     * @brief Helper for mapping unordered species pairs to compact linear indices.
     *
     * @details
     * Allows pairwise tables to be stored as 1D buffers instead of dense 2D matrices.
     */
    DevicePairIndexer<int> _pair_indexer {};

    /**
     * @brief Number of species currently represented by the associated fluid.
     *
     * @details
     * Used together with @ref _pair_indexer to size and access pairwise tables.
     */
    int _species_count = 0;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::system::DsmcSolver.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
using DsmcHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSolver<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::system::DsmcSolver.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
using DsmcDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_solver.hpp>