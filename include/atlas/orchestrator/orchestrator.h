#pragma once

/**
 * @file orchestrator.h
 * @brief Declares the host-side solver orchestration wrapper used by Atlas runtime systems.
 *
 * @details
 * This header defines @ref atlas::system::Orchestrator, a lightweight runtime
 * coordination object responsible for forwarding the currently active simulation
 * probes to a configured solver implementation.
 *
 * The orchestrator is intentionally minimal. It does not implement any physical,
 * numerical, or integration logic by itself. Instead, it serves as a stable
 * indirection layer between higher-level runtime code and a concrete solver.
 *
 * ## Purpose
 * In the Atlas runtime architecture, the orchestrator exists to:
 * - hold a solver dependency in a uniform and replaceable way,
 * - expose a single `solve(...)` entry point for system-level code,
 * - decouple high-level runtime flow from the exact concrete solver type,
 * - allow runtime stages to omit solver logic entirely when no solver is configured.
 *
 * ## Runtime role
 * A typical simulation runtime, such as @ref atlas::system::System, owns the
 * canonical device probes for:
 * - the active domain,
 * - the active spatial search structure,
 * - the active particle/fluid storage,
 * - the active codec state.
 *
 * During the solve stage, those probes are forwarded into the orchestrator,
 * which in turn forwards them to the configured solver.
 *
 * ## No-op behavior
 * If no solver is configured, the orchestrator behaves as a no-op dispatcher:
 * - `solve(...)` accepts the probes,
 * - no solver call is performed,
 * - the runtime can continue without special-case branching at the call site.
 *
 * This keeps the higher-level update loop simpler and makes solver presence an
 * optional configuration concern rather than a mandatory runtime invariant.
 *
 * ## Ownership model
 * The orchestrator stores the solver as a host-side shared pointer:
 * - @ref SolveHostPtr
 *
 * This allows:
 * - shared ownership across runtime components,
 * - late replacement of the active solver,
 * - uniform builder-based configuration.
 *
 * ## Construction
 * An orchestrator may be:
 * - default-constructed with no solver,
 * - explicitly constructed from a solver handle,
 * - configured through the nested fluent @ref Builder.
 *
 * ## Design philosophy
 * The orchestrator is intentionally thin and policy-free:
 * - it does not own domain/fluid/searcher/codec state,
 * - it does not cache or transform probes,
 * - it does not choose between multiple solvers,
 * - it simply forwards the current runtime state to exactly one configured solver.
 *
 * This makes it suitable as a stable extension point in the runtime pipeline.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the coordinated solver and runtime probes.
 */

#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/solver.h>

namespace atlas::system {

/**
 * @brief Host-side wrapper that forwards runtime probes to a configured solver.
 *
 * @details
 * @ref Orchestrator acts as a thin runtime adapter between high-level simulation
 * control code and a concrete solver implementation.
 *
 * It stores a solver handle and exposes a uniform @ref solve entry point that
 * accepts the current runtime probes:
 * - @ref DomainDeviceProbe
 * - @ref SpatialHashingProbe
 * - @ref FluidDeviceProbe
 * - @ref CodecDeviceProbe
 *
 * ## Why this abstraction exists
 * Higher-level runtime code such as @ref atlas::system::System should not need
 * to know:
 * - which specific solver subtype is active,
 * - whether a solver is present at all,
 * - how solver ownership is managed.
 *
 * By routing solver access through @ref Orchestrator, that complexity is
 * isolated to a very small host-side wrapper.
 *
 * ## Solve forwarding semantics
 * When @ref solve is called:
 * - if a solver is present, the orchestrator forwards the probes to it,
 * - if no solver is present, the function performs no work.
 *
 * This preserves a clean runtime update pipeline and allows solver stages to be
 * optional without requiring repeated null checks at the call site.
 *
 * ## Mutability of probes
 * The probes are accepted by non-const reference because a solver is expected
 * to mutate simulation state, for example by:
 * - updating field values,
 * - modifying particle state,
 * - consuming or producing codec-side runtime data.
 *
 * ## Typical usage
 * @code
 * auto orchestrator = atlas::Orchestrator<float>::builder()
 *     .with_solver(solver)
 *     .build();
 *
 * orchestrator.solve(domain_probe, searcher_probe, particle_probe, codec_probe);
 * @endcode
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the runtime probes and solver.
 */
template <typename T>
class Orchestrator final {
public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Orchestrator.
     *
     * @details
     * The builder stages the solver dependency, validates the staged state, and
     * constructs either:
     * - an orchestrator by value, or
     * - a host-owned shared pointer to an orchestrator.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an orchestrator with no configured solver.
     *
     * In this state:
     * - @ref solver returns an empty handle,
     * - @ref solve behaves as a no-op.
     */
    Orchestrator()  = default;

    /**
     * @brief Destructor.
     *
     * @details
     * Defaulted because the orchestrator only owns managed handle state.
     */
    ~Orchestrator() = default;

    /**
     * @brief Construct an orchestrator with the provided solver handle.
     *
     * @details
     * Stores the supplied host-side shared solver pointer as the active solver
     * dependency of the orchestrator.
     *
     * This constructor does not itself invoke validation logic; any validation
     * policy is implementation-defined and may instead be applied through the
     * builder path.
     *
     * @param solver Host-side shared pointer to the solver implementation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Orchestrator(SolveHostPtr<T> solver) noexcept;

    /**
     * @brief Create a fluent builder for @ref Orchestrator.
     *
     * @details
     * Returns a default-initialized builder with no staged solver handle.
     *
     * @return A default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Forward the active runtime probes to the configured solver.
     *
     * @details
     * This function is the orchestrator's main runtime entry point.
     *
     * If a solver is currently configured:
     * - the probes are forwarded to that solver,
     * - the solver may mutate simulation state through those probes.
     *
     * If no solver is configured:
     * - the function performs no work,
     * - all probes remain unchanged by the orchestrator itself.
     *
     * ## Probe roles
     * - @p domain provides domain/grid metadata and field access,
     * - @p searcher provides neighborhood-search access structures,
     * - @p particle provides particle/state buffers,
     * - @p codec provides runtime codec-side mutable state.
     *
     * ## Exception/ownership notes
     * The orchestrator does not take ownership of the supplied probes and does
     * not cache them. They are forwarded only for the duration of this call.
     *
     * @param domain Active domain probe.
     * @param searcher Active spatial searcher probe.
     * @param particle Active fluid/particle probe.
     * @param codec Active codec probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(DomainDeviceProbe<T>& domain,
          SpatialHashingProbe<T>& searcher,
          FluidDeviceProbe<T>& particle,
          CodecDeviceProbe<T>& codec);

    /**
     * @brief Replace the solver handle stored by the orchestrator.
     *
     * @details
     * After this call:
     * - subsequent @ref solve calls use the new solver handle,
     * - an empty handle causes the orchestrator to revert to no-op solve behavior.
     *
     * @param solver New host-side shared solver pointer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_solver(SolveHostPtr<T> solver) noexcept;

    /**
     * @brief Return the currently configured solver handle.
     *
     * @details
     * This accessor exposes the stored solver dependency without transferring
     * ownership.
     *
     * @return Const reference to the stored solver pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SolveHostPtr<T>&
    solver() const noexcept;

private:
    /**
     * @brief Stored host-side solver dependency.
     *
     * @details
     * When empty, the orchestrator acts as a no-op solver stage.
     */
    SolveHostPtr<T> _solver {};
};

/**
 * @brief Fluent builder for @ref Orchestrator.
 *
 * @details
 * The builder provides a controlled construction path for the orchestrator by
 * staging its solver dependency before creating the final object.
 *
 * ## Responsibilities
 * The builder is responsible for:
 * - storing a staged solver handle,
 * - validating that staged configuration according to Atlas policy,
 * - constructing an orchestrator by value or shared ownership.
 *
 * ## Typical usage
 * @code
 * auto orchestrator = atlas::system::Orchestrator<double>::builder()
 *     .with_solver(solver)
 *     .make_host_shared();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared.
 *
 * A typical validation policy may check that:
 * - a non-null solver handle is present when required,
 * - the configured solver is ready for runtime use.
 *
 * The exact validation behavior is implementation-defined in
 * `orchestrator.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the orchestrated solver.
 */
template <typename T>
class Orchestrator<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with no staged solver dependency.
     */
    Builder() = default;

    /**
     * @brief Store the solver handle to be used by the built orchestrator.
     *
     * @details
     * Replaces any previously staged solver handle.
     *
     * @param solver Host-side shared solver pointer.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(SolveHostPtr<T> solver) noexcept;

    /**
     * @brief Build an @ref Orchestrator value from the staged configuration.
     *
     * @details
     * Validates the staged builder state and returns a fully constructed
     * orchestrator by value.
     *
     * @return A configured orchestrator instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Orchestrator<T>
    build() const;

    /**
     * @brief Build an @ref Orchestrator wrapped in host shared ownership.
     *
     * @details
     * Validates the staged builder state and returns a
     * `atlas::host_shared_ptr<Orchestrator<T>>`.
     *
     * @return Host-side shared pointer to a configured orchestrator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Orchestrator<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate the staged builder configuration.
     *
     * @details
     * Performs any required pre-construction checks on the staged solver
     * dependency.
     *
     * The exact validation rules are implementation-defined in
     * `orchestrator.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Staged solver dependency.
     *
     * @details
     * This handle is consumed by @ref build and @ref make_host_shared.
     */
    SolveHostPtr<T> _solver {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::Orchestrator.
 *
 * @tparam T Floating-point scalar type used by the orchestrated solver.
 */
template <typename T>
using Orchestrator = atlas::system::Orchestrator<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to
 *        @ref atlas::system::Orchestrator.
 *
 * @tparam T Floating-point scalar type used by the orchestrated solver.
 */
template <typename T>
using OrchestratorHostPtr = atlas::host_shared_ptr<atlas::system::Orchestrator<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to
 *        @ref atlas::system::Orchestrator.
 *
 * @details
 * Although the orchestrator is fundamentally a host-side coordination object,
 * this alias is provided for API symmetry with other Atlas runtime components.
 *
 * @tparam T Floating-point scalar type used by the orchestrated solver.
 */
template <typename T>
using OrchestratorDevicePtr = atlas::device_shared_ptr<atlas::system::Orchestrator<T>>;

} // namespace atlas

#include <atlas/orchestrator/orchestrator.hpp>