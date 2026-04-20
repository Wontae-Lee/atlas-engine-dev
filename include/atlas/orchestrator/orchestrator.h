#pragma once

/**
 * @file orchestrator.h
 * @brief Declares the Orchestrator class used to coordinate codec-aware solver execution.
 */

#include <atlas/buffer/host_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/measure/measurer.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/solver.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

/**
 * @brief Coordinates execution of a sequence of solvers, optionally with a codec.
 *
 * An Orchestrator stores:
 * - an optional universe,
 * - an optional fluid,
 * - an optional searcher,
 * - an optional codec,
 * - an optional measure,
 * - an ordered list of solver objects.
 *
 * Its primary responsibility is to run the search -> classify -> measure ->
 * solve pipeline according to the current orchestration mode:
 * - if no codec is configured, each solver is invoked with solve(dt)
 * - if a codec is configured, each solver is invoked with solve(allocated_solver, index, dt)
 *
 * This allows the solver pipeline to adapt its behavior depending on whether
 * a codec-aware execution path is available.
 *
 * @tparam T Scalar type associated with the simulation system.
 */
template <typename T>
class Orchestrator final {
public:
    /**
     * @brief Builder for configuring and constructing Orchestrator instances.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     */
    Orchestrator() = default;

    /**
     * @brief Destructor.
     */
    ~Orchestrator() = default;

    /**
     * @brief Constructs an orchestrator from pipeline dependencies and solver list.
     *
     * @param universe Optional host-side shared pointer to a universe.
     * @param fluid Optional host-side shared pointer to a fluid.
     * @param searcher Optional host-side shared pointer to a spatial searcher.
     * @param codec Optional host-side shared pointer to a codec.
     * @param measurer Optional host-side shared pointer to a measurer.
     * @param solvers Ordered list of host-side shared solver pointers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Orchestrator(UniverseHostPtr<T> universe,
                 FluidHostPtr<T> fluid,
                 SpatialHashingSearcherHostPtr<T> searcher,
                 CodecHostPtr<T> codec,
                 MeasurerHostPtr<T> measurer,
                 HostBuffer<SolveHostPtr<T>> solvers) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    search();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    classify();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt);

    /**
     * @brief Runs the full orchestrator pipeline for one simulation step.
     *
     * This is the high-level entry point used by @ref atlas::system::System so
     * the system does not need to invoke the individual orchestration stages directly.
     *
     * @param dt Time step forwarded to field-force and solver stages.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Applies a cell-wise field force to particle velocities when available.
     *
     * If the universe exposes a @ref atlas::universe::UniverseFieldForceState and
     * the orchestrator has the universe, fluid, and searcher dependencies needed
     * to map particles into cells, this function updates particle velocity using:
     *
     * @code
     * v += dt * (F / m)
     * @endcode
     *
     * where @c F is the force vector stored for the particle's current cell and
     * @c m is the particle mass obtained from its species material properties.
     *
     * @param dt Time step used for the explicit velocity update.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_field_force(T dt);

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent orchestrator construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    orchestrate(T dt);

    /**
     * @brief Sets or replaces the universe dependency.
     *
     * @param universe Host-side shared pointer to the universe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets or replaces the fluid dependency.
     *
     * @param fluid Host-side shared pointer to the fluid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets or replaces the searcher.
     *
     * @param searcher Host-side shared pointer to the searcher.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets or replaces the codec.
     *
     * @param codec Host-side shared pointer to the codec.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_codec(CodecHostPtr<T> codec) noexcept;

    /**
     * @brief Sets or replaces the measure.
     *
     * @param measurer Host-side shared pointer to the measure.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_measurer(MeasurerHostPtr<T> measurer) noexcept;

    /**
     * @brief Appends a solver to the orchestration list.
     *
     * @param solver Host-side shared pointer to the solver.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_solver(SolveHostPtr<T> solver) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SpatialHashingSearcherHostPtr<T>&
    searcher() const noexcept;

    /**
     * @brief Returns the configured universe.
     *
     * @return Const reference to the universe shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const UniverseHostPtr<T>&
    universe() const noexcept;

    /**
     * @brief Returns the configured fluid.
     *
     * @return Const reference to the fluid shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    /**
     * @brief Returns the configured codec.
     *
     * @return Const reference to the codec shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const CodecHostPtr<T>&
    codec() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const MeasurerHostPtr<T>&
    measurer() const noexcept;

    /**
     * @brief Returns the configured solver list.
     *
     * @return Const reference to the solver container.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<SolveHostPtr<T>>&
    solvers() const noexcept;

private:
    /**
     * @brief Optional universe used to access cell-wise field states.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Optional fluid used to access particle states updated by orchestration.
     */
    FluidHostPtr<T> _fluid {};

    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Optional codec used to control codec-aware solver execution.
     */
    CodecHostPtr<T> _codec {};

    MeasurerHostPtr<T> _measurer {};

    /**
     * @brief Ordered list of solvers managed by this orchestrator.
     */
    HostBuffer<SolveHostPtr<T>> _solvers {};
};

/**
 * @brief Builder for Orchestrator.
 *
 * This builder collects:
 * - an optional codec,
 * - an ordered sequence of solvers.
 *
 * Validation currently ensures that no stored solver pointer is null.
 *
 * @tparam T Scalar type associated with the simulation system.
 */
template <typename T>
class Orchestrator<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the universe used by the orchestrator.
     *
     * @param universe Host-side shared pointer to the universe.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid used by the orchestrator.
     *
     * @param fluid Host-side shared pointer to the fluid.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the codec used by the orchestrator.
     *
     * @param codec Host-side shared pointer to the codec.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_codec(CodecHostPtr<T> codec) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measurer(MeasurerHostPtr<T> measurer) noexcept;

    /**
     * @brief Appends a solver to the orchestrator configuration.
     *
     * @param solver Host-side shared pointer to the solver.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(SolveHostPtr<T> solver) noexcept;

    /**
     * @brief Builds a validated Orchestrator object.
     *
     * @return Constructed Orchestrator object.
     *
     * @throw std::runtime_error Thrown if any configured solver is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Orchestrator<T>
    build() const;

    /**
     * @brief Builds a host-side shared Orchestrator object.
     *
     * @return Host shared pointer to a constructed Orchestrator object.
     *
     * @throw std::runtime_error Thrown if any configured solver is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Orchestrator<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the current builder state.
     *
     * @throw std::runtime_error Thrown if any solver pointer is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Universe collected by the builder.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Fluid collected by the builder.
     */
    FluidHostPtr<T> _fluid {};

    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Codec collected by the builder.
     */
    CodecHostPtr<T> _codec {};

    MeasurerHostPtr<T> _measurer {};

    /**
     * @brief Solver list collected by the builder.
     */
    HostBuffer<SolveHostPtr<T>> _solvers {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Alias for atlas::system::Orchestrator.
 *
 * @tparam T Scalar type associated with the orchestrator.
 */
template <typename T>
using Orchestrator = atlas::system::Orchestrator<T>;

/**
 * @brief Host-side shared pointer alias for Orchestrator.
 *
 * @tparam T Scalar type associated with the orchestrator.
 */
template <typename T>
using OrchestratorHostPtr = atlas::host_shared_ptr<atlas::system::Orchestrator<T>>;

/**
 * @brief Device-side shared pointer alias for Orchestrator.
 *
 * @tparam T Scalar type associated with the orchestrator.
 */
template <typename T>
using OrchestratorDevicePtr = atlas::device_shared_ptr<atlas::system::Orchestrator<T>>;

} // namespace atlas

#include <atlas/orchestrator/orchestrator.hpp>
