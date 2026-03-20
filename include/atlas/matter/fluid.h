#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/generator/generator.h>
#include <atlas/matter/fluidic_particle.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

/**
 * @brief Host-side container of fluidic (DSMC) particles with per-species amount and generator data.
 *
 * @details
 * `Fluid` is a **host-side** aggregate container that owns a collection of fluidic particles
 * and parallel per-species buffers called **amounts** and **generators**.
 *
 * ## Ownership model
 * Particles are stored as `FluidicParticleHostPtr<T>` (i.e., `atlas::host_shared_ptr<FluidicParticle<T>>`)
 * inside a @ref HostBuffer. This design has several practical benefits:
 *
 * - **Stable ownership**: each particle is heap-allocated and reference-counted.
 * - **Cheap container copies**: copying a `Fluid` typically copies shared pointers, not particle values.
 * - **Shared access**: multiple host subsystems can safely refer to the same particle instances
 *   (e.g., logging/IO, diagnostics, preprocessing, sampling).
 *
 * ## Parallel buffers
 * In addition to particles, `Fluid` stores:
 * - an `amounts` buffer,
 * - a `generators` buffer.
 *
 * These are all parallel arrays:
 * - `_particles[i]` corresponds to `_amounts[i]` and `_generators[i]`
 * - All buffers always represent the same logical length and indexing domain
 *
 * The semantic meaning of `amount` is intentionally left **simulation-specific**. Common uses include:
 * - macroparticle multiplicity / number of represented molecules
 * - injected amount / particle weight for source terms
 * - mixture fraction / mass fraction scalars carried alongside particles
 * - per-particle scaling for sampling or collision frequency adjustments
 *
 * By default, builder APIs that do not accept an explicit `amount` will assign `T(1)` to the new entry.
 *
 * ## Construction
 * `Fluid` is intended to be constructed through its nested fluent @ref Builder, which provides:
 * - append-by-value (deep-copy into a new shared object),
 * - append-by-pointer (adopt an existing `host_shared_ptr`),
 * - append-many overloads with optional explicit `amounts` and `generators`,
 * - build-time validation policies (non-empty, null-pointer rejection, negative amount rejection).
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., `float`, `double`).
 *
 * @note
 * - Host-only: the builder may throw exceptions; the container relies on host allocation.
 * - `Fluid` itself does not enforce invariants at mutation time; invariants are expected
 *   to be established by @ref Builder and respected by subsequent users of the container.
 * - Thread-safety depends on `host_shared_ptr` and @ref HostBuffer; read-only sharing is
 *   usually safe, but concurrent mutation must follow your project’s synchronization rules.
 */
template <typename T>
class Fluid final {
public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Fluid.
     *
     * @details
     * Provides an append-oriented API for adding particles and their associated amounts,
     * plus optional validation policies at build time.
     *
     * See @ref Fluid<T>::Builder for the full set of overloads and policies.
     */
    class Builder;

    /**
     * @brief Default-constructed fluid is empty.
     *
     * @details
     * Creates an empty container with zero particles and zero amounts.
     * No particles are allocated and buffers remain in their default states.
     */
    Fluid() = default;

    /**
     * @brief Builder entry point.
     *
     * @details
     * Returns a builder used to populate `_particles` and `_amounts` and set validation policy.
     *
     * @return A default-initialized @ref Builder.
     *
     * @note Host-only: builder can throw at build time.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /// @brief Default destructor.
    ~Fluid() = default;

    // ------------------------------------------------------------
    // Convenience
    // ------------------------------------------------------------

    /**
     * @brief Number of particles currently stored.
     *
     * @return Count of particles (and also the length of @ref amounts() and @ref generators()).
     *
     * @note
     * This returns the number of entries in the parallel buffers.
     * The invariants expected by this type require:
     * - `particles().size() == amounts().size() == generators().size()`
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    size() const noexcept;

    /**
     * @brief Returns whether the container holds no particles.
     *
     * @return `true` if @ref size() is zero; otherwise `false`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

    /**
     * @brief Immutable access to the particle pointer buffer.
     *
     * @details
     * Returns a const reference to the underlying @ref HostBuffer containing
     * `FluidicParticleHostPtr<T>` entries.
     *
     * @return Const reference to particle pointer buffer.
     *
     * @note
     * The returned buffer contains shared pointers; even if the buffer is const,
     * the pointed-to particle object may still be mutable depending on pointer constness.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<FluidicParticleHostPtr<T>>&
    particles() const noexcept;

    /**
     * @brief Mutable access to the particle pointer buffer.
     *
     * @details
     * Returns a mutable reference to the internal particle pointer buffer.
     * This allows modifying the list (append/remove/replace pointers), but doing so
     * may violate invariants unless `_amounts` is updated consistently.
     *
     * @return Mutable reference to particle pointer buffer.
     *
     * @warning
     * If you modify the particle buffer directly, you must maintain the invariant:
     * `particles().size() == amounts().size() == generators().size()`.
     * Prefer using the @ref Builder for construction and controlled population.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<FluidicParticleHostPtr<T>>&
    particles() noexcept;

    /**
     * @brief Immutable access to per-particle amounts (parallel to @ref particles()).
     *
     * @details
     * Returns a const reference to the amount buffer. The amount at index `i`
     * corresponds to the particle at index `i` in @ref particles().
     *
     * @return Const reference to amount buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<T>&
    amounts() const noexcept;

    /**
     * @brief Mutable access to per-particle amounts (parallel to @ref particles()).
     *
     * @details
     * Returns a mutable reference to the amount buffer. The amount at index `i`
     * corresponds to the particle at index `i` in @ref particles().
     *
     * @return Mutable reference to amount buffer.
     *
     * @warning
     * If you modify amounts directly, ensure indices remain aligned with @ref particles().
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<T>&
    amounts() noexcept;

    /**
     * @brief Immutable access to per-species velocity generators.
     *
     * @details
     * Returns a const reference to the generator buffer. The entry at index `i`
     * corresponds to the particle definition at index `i` in @ref particles().
     *
     * @return Const reference to generator buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<GeneratorHostPtr<T>>&
    generators() const noexcept;

    /**
     * @brief Mutable access to per-species velocity generators.
     *
     * @details
     * Returns a mutable reference to the generator buffer. The entry at index `i`
     * corresponds to the particle definition at index `i` in @ref particles().
     *
     * @return Mutable reference to generator buffer.
     *
     * @warning
     * If you modify generators directly, ensure indices remain aligned with
     * @ref particles() and @ref amounts().
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<GeneratorHostPtr<T>>&
    generators() noexcept;

    /**
     * @brief Returns whether @ref amounts() forms a normalized mixture ratio vector.
     *
     * @details
     * This predicate is useful when a `Fluid` is used as a species-mixture descriptor
     * for source emission rather than as a generic weighted particle list.
     *
     * The stored amounts are considered normalized when:
     * - the buffer is non-empty,
     * - every amount is finite,
     * - every amount is non-negative,
     * - the total sum differs from `1` by at most `eps`.
     *
     * @param eps Absolute tolerance used for the total-sum check.
     * @return `true` if the amount buffer satisfies the normalized-ratio invariant,
     *         otherwise `false`.
     *
     * @note
     * This function does not mutate the object and does not perform renormalization.
     * It is only a structural/semantic check over the currently stored values.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    normalized(T eps = atlas::eps) const noexcept;

private:
    /// @brief Allow @ref Builder to populate internals without exposing mutators publicly.
    friend class Builder;

private:
    /**
     * @brief Owned host-side particles (each entry is a `host_shared_ptr`).
     *
     * @details
     * Each pointer is expected to own (or share ownership of) a heap-allocated particle.
     * Validation policies may forbid null pointers depending on builder settings.
     */
    HostBuffer<FluidicParticleHostPtr<T>> _particles;

    /**
     * @brief Per-particle amount/weight buffer.
     *
     * @details
     * Maintains a 1:1 index correspondence with @ref _particles.
     * The meaning of "amount" is domain-specific and chosen by the simulation.
     */
    HostBuffer<T> _amounts;

    /**
     * @brief Per-species velocity-generator buffer.
     *
     * @details
     * Maintains a 1:1 index correspondence with @ref _particles and @ref _amounts.
     * A null pointer indicates that no species-specific generator was provided.
     */
    HostBuffer<GeneratorHostPtr<T>> _generators;
};

/* ====================================================================== */
/* Builder                                                                 */
/* ====================================================================== */

/**
 * @brief Fluent builder for @ref Fluid.
 *
 * @details
 * The builder provides **append semantics** for particle population and exposes
 * build-time validation policies. It accumulates particles, amounts, and generators
 * into parallel buffers which are transferred into the final @ref Fluid instance.
 *
 * ## Parallel buffers
 * Internally, the builder maintains:
 * - `_particles[i]`  : `FluidicParticleHostPtr<T>` (shared pointer to a particle)
 * - `_amounts[i]`    : `T` (amount associated with that particle)
 * - `_generators[i]` : `GeneratorHostPtr<T>` (velocity generator for that particle/species)
 *
 * All buffers are intended to remain the same length at all times.
 *
 * ## Default amount behavior
 * Many `add_*` overloads do not require an explicit `amount`. In those cases:
 * - the builder appends `T(1)` as the amount for the newly added particle(s)
 *
 * ## Validation policies
 * The builder supports independent policy switches:
 * - @ref require_non_empty : reject building if no particles were added
 * - @ref reject_null_particles : reject building if any particle pointer is null
 * - @ref reject_negative_amounts : reject building if any amount is negative
 *
 * Validation is applied in @ref validate and is triggered by @ref build and
 * @ref make_host_shared.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 *
 * @note
 * Host-only: builder may throw exceptions (`std::runtime_error`) and uses host memory utilities.
 */
template <typename T>
class Fluid<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates an empty builder with default policies:
     * - `_require_non_empty = false` (empty fluids allowed)
     * - `_reject_null_particles = true` (null pointers rejected by default)
     * - `_reject_negative_amounts = true` (negative amounts rejected by default)
     *
     * No particles are added and no allocation is performed beyond buffer defaults.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Fluid (by value).
     *
     * @details
     * Performs validation and returns a `Fluid<T>` whose internal buffers contain the
     * accumulated particles, amounts, and generators.
     *
     * @return Fully constructed fluid.
     *
     * @throws std::runtime_error if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Fluid<T>
    build() const;

    /**
     * @brief Build a configured @ref Fluid in a `host_shared_ptr`.
     *
     * @details
     * Performs the same validation as @ref build and returns a shared pointer owning
     * the resulting fluid container.
     *
     * @return Shared pointer owning a constructed fluid.
     *
     * @throws std::runtime_error if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Fluid<T>>
    make_host_shared() const;

    // ------------------------------------------------------------
    // Inputs (append semantics)
    // ------------------------------------------------------------

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particle(const FluidicParticle<T>& p);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particle(const FluidicParticle<T>& p, T amount);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particle(const FluidicParticle<T>& p, T amount, GeneratorHostPtr<T> generator);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particle(FluidicParticleHostPtr<T> p);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particle(FluidicParticleHostPtr<T> p, T amount);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particle(FluidicParticleHostPtr<T> p, T amount, GeneratorHostPtr<T> generator);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particles(const HostBuffer<FluidicParticle<T>>& ps);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particles(const HostBuffer<FluidicParticle<T>>& ps, const HostBuffer<T>& amounts);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particles(const HostBuffer<FluidicParticle<T>>& ps,
                  const HostBuffer<T>& amounts,
                  const HostBuffer<GeneratorHostPtr<T>>& generators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particles(const HostBuffer<FluidicParticleHostPtr<T>>& ps);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particles(const HostBuffer<FluidicParticleHostPtr<T>>& ps, const HostBuffer<T>& amounts);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_particles(const HostBuffer<FluidicParticleHostPtr<T>>& ps,
                  const HostBuffer<T>& amounts,
                  const HostBuffer<GeneratorHostPtr<T>>& generators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    require_non_empty(bool on = true) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    reject_null_particles(bool on = true) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    reject_negative_amounts(bool on = true) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    require_normalized_amounts(bool on = true) noexcept;

private:
    /**
     * @brief Validate builder configuration or throw.
     *
     * @details
     * This function enforces the builder policy flags:
     *
     * - `_require_non_empty`:
     *   - if true, `_particles.size()` must be > 0
     * - `_reject_null_particles`:
     *   - if true, no entry in `_particles` may be null
     * - `_reject_negative_amounts`:
     *   - if true, all entries in `_amounts` must be >= 0
     * - `_require_normalized_amounts`:
     *   - if true, `_amounts` must be non-empty, finite, non-negative, and sum to 1
     *
     * It is also responsible for basic structural invariants:
     * - `_particles.size() == _amounts.size() == _generators.size()`
     *
     * @throws std::runtime_error if any check fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /// @brief Accumulated particle pointers (parallel to @ref _amounts).
    HostBuffer<FluidicParticleHostPtr<T>> _particles;

    /// @brief Accumulated amounts (parallel to @ref _particles).
    HostBuffer<T> _amounts;

    /// @brief Accumulated generators (parallel to @ref _particles and @ref _amounts).
    HostBuffer<GeneratorHostPtr<T>> _generators;

    /// @brief Policy: reject building empty fluids when enabled.
    bool _require_non_empty = false;

    /// @brief Policy: reject null pointers in `_particles` when enabled.
    bool _reject_null_particles = true;

    /// @brief Policy: reject negative values in `_amounts` when enabled.
    bool _reject_negative_amounts = true;

    /// @brief Policy: require `_amounts` to form a normalized ratio vector.
    bool _require_normalized_amounts = false;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::Fluid<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Fluid = system::Fluid<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to `atlas::system::Fluid<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using FluidHostPtr = atlas::host_shared_ptr<system::Fluid<T>>;

} // namespace atlas

// Implementation header for templates.
#include <atlas/matter/fluid.hpp>
