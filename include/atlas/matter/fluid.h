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
 * - a `mole_fractions` buffer (normalized, sums to 1),
 * - a `generators` buffer.
 *
 * These are all parallel arrays:
 * - `_particles[i]` corresponds to `_mole_fractions[i]` and `_generators[i]`
 * - All buffers always represent the same logical length and indexing domain
 *
 * Mole fractions represent the normalized proportion of each species in the mixture,
 * used for stochastic species assignment during particle emission.
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
     * @brief Number of species currently stored.
     *
     * @return Count of species (and also the length of @ref mole_fractions() and @ref generators()).
     *
     * @note
     * This returns the number of entries in the parallel buffers.
     * The invariants expected by this type require:
     * - `particles().size() == mole_fractions().size() == generators().size()`
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
     * `particles().size() == mole_fractions().size() == generators().size()`.
     * Prefer using the @ref Builder for construction and controlled population.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<FluidicParticleHostPtr<T>>&
    particles() noexcept;

    /**
     * @brief Immutable access to per-species mole fractions (parallel to @ref particles()).
     *
     * @details
     * Returns a const reference to the mole fraction buffer. The mole fraction at index `i`
     * corresponds to the particle at index `i` in @ref particles().
     * Mole fractions are normalized and sum to 1.
     *
     * @return Const reference to mole fraction buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<T>&
    mole_fractions() const noexcept;

    /**
     * @brief Mutable access to per-species mole fractions (parallel to @ref particles()).
     *
     * @details
     * Returns a mutable reference to the mole fraction buffer. The mole fraction at index `i`
     * corresponds to the particle at index `i` in @ref particles().
     *
     * @return Mutable reference to mole fraction buffer.
     *
     * @warning
     * If you modify mole fractions directly, ensure indices remain aligned with @ref particles()
     * and that they sum to 1.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<T>&
    mole_fractions() noexcept;

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
     * @brief Per-species mole fraction buffer (normalized, sums to 1).
     *
     * @details
     * Maintains a 1:1 index correspondence with @ref _particles.
     * Mole fractions represent the relative proportion of each species in the mixture.
     */
    HostBuffer<T> _mole_fractions;

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
 * - `_particles[i]`      : `FluidicParticleHostPtr<T>` (shared pointer to a particle)
 * - `_mole_fractions[i]` : `T` (mole fraction for that species)
 * - `_generators[i]`     : `GeneratorHostPtr<T>` (velocity generator for that particle/species)
 *
 * All buffers are intended to remain the same length at all times.
 *
 * ## Default mole fraction behavior
 * Many `add_*` overloads do not require an explicit mole fraction. In those cases:
 * - the builder will automatically normalize all mole fractions to sum to 1 at build time
 *
 * ## Validation policies
 * The builder supports independent policy switches:
 * - @ref require_non_empty : reject building if no particles were added
 * - @ref reject_null_particles : reject building if any particle pointer is null
 *
 * Validation is applied in @ref validate and is triggered by @ref build and
 * @ref make_host_shared. Mole fractions are automatically normalized at build time.
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
     *
     * No particles are added and no allocation is performed beyond buffer defaults.
     * Mole fractions will be automatically normalized at build time.
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
    add_species(const FluidicParticle<T>& p);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const FluidicParticle<T>& p, T mole_fraction);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const FluidicParticle<T>& p, T mole_fraction, GeneratorHostPtr<T> generator);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(FluidicParticleHostPtr<T> p);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(FluidicParticleHostPtr<T> p, T mole_fraction);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(FluidicParticleHostPtr<T> p, T mole_fraction, GeneratorHostPtr<T> generator);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticle<T>>& ps);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticle<T>>& ps, const HostBuffer<T>& mole_fractions);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticle<T>>& ps,
                     const HostBuffer<T>& mole_fractions,
                     const HostBuffer<GeneratorHostPtr<T>>& generators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticleHostPtr<T>>& ps);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticleHostPtr<T>>& ps, const HostBuffer<T>& mole_fractions);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticleHostPtr<T>>& ps,
                     const HostBuffer<T>& mole_fractions,
                     const HostBuffer<GeneratorHostPtr<T>>& generators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    require_non_empty(bool on = true) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    reject_null_particles(bool on = true) noexcept;

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
     *
     * It is also responsible for basic structural invariants:
     * - `_particles.size() == _mole_fractions.size() == _generators.size()`
     * - Normalizes mole fractions to sum to 1
     *
     * @throws std::runtime_error if any check fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /// @brief Accumulated particle pointers (parallel to @ref _mole_fractions).
    HostBuffer<FluidicParticleHostPtr<T>> _particles;

    /// @brief Accumulated mole fractions (parallel to @ref _particles).
    HostBuffer<T> _mole_fractions;

    /// @brief Accumulated generators (parallel to @ref _particles and @ref _mole_fractions).
    HostBuffer<GeneratorHostPtr<T>> _generators;

    /// @brief Policy: reject building empty fluids when enabled.
    bool _require_non_empty = false;

    /// @brief Policy: reject null pointers in `_particles` when enabled.
    bool _reject_null_particles = true;
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
