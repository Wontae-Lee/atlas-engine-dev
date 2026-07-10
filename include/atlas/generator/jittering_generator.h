#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/random/seed.h>

#include <cstddef>
#include <utility>

namespace atlas {

/**
 * @brief Generator leaf that emits a constant base velocity perturbed by a small
 *        uniform jitter, plus a bulk drift.
 *
 * Each velocity component is @c base_value + U(-|jitter_radius|, +|jitter_radius|)
 * with the bulk drift added on top; the radius is taken as an absolute value so a
 * negative setting is treated symmetrically. Species selection is the same
 * weighted draw as the other leaves (@c sample_weighted_choice over ratios and
 * numbers). This is a non-physical "roughly here, with a little spread" model;
 * @c _temperature is stored and validated but does not affect the sampling.
 *
 * Owns two @c DeviceBuffer members, so the leaf is move-only. Build via @c Builder.
 *
 * @note Satisfies @c ConceptGenerator; stored in the @c Generator union.
 */
class JitteringGenerator final {
public:
    /// Validating factory that assembles a @c JitteringGenerator; see below.
    class Builder;

public:
    /// Constructs an empty leaf with no species data; @c generate yields 0.
    JitteringGenerator() = default;

    /**
     * @brief Constructs a ready-to-run leaf, taking ownership of the buffers.
     *
     * @param species_ratios  Per-species selection weights (device).
     * @param species_numbers Species-id values (device), parallel to the ratios.
     * @param temperature     Gas temperature in kelvin; stored, not used to sample.
     * @param base_value      Center value applied to every velocity component.
     * @param jitter_radius   Half-width of the uniform perturbation; used as |r|.
     * @param bulk_velocity   Constant drift added to every sampled velocity.
     * @param seed            Base seed folded with the particle index per draw.
     */
    ATLAS_HOST
    JitteringGenerator(DeviceBuffer<float> species_ratios,
                       DeviceBuffer<float> species_numbers,
                       float temperature,
                       float base_value,
                       float jitter_radius,
                       Float3 bulk_velocity,
                       unsigned int seed) noexcept;

    /// @return A fresh, empty @c Builder.
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Retargets the drift added to every sampled velocity.
     * @param bulk_velocity New bulk velocity.
     */
    ATLAS_HOST void
    set_bulk_velocity(const Float3& bulk_velocity) noexcept {
        _bulk_velocity = bulk_velocity;
    }

    /// @return The current bulk (drift) velocity.
    ATLAS_NODISCARD ATLAS_HOST Float3
    bulk_velocity() const noexcept {
        return _bulk_velocity;
    }

    /// @return The stored temperature in kelvin (informational for this leaf).
    ATLAS_NODISCARD ATLAS_HOST float
    temperature() const noexcept {
        return _temperature;
    }

    /**
     * @brief Fills @c [offset, offset + count) with sampled species and velocities.
     *
     * Runs a device kernel over the clamped range; each thread seeds an RNG from
     * @c shuffle_key(index, seed), picks a species, and writes
     * base + jitter + bulk. Silently no-ops on null pointers, zero count, or an
     * offset past the buffers' shared capacity.
     *
     * @param velocities Target velocity state.
     * @param species    Target species state.
     * @param offset     First slot to write.
     * @param count      Slots requested.
     * @return Number of slots actually written (may be clamped below @p count).
     */
    ATLAS_NODISCARD ATLAS_HOST int
    generate(FluidVelocityState* velocities,
             FluidSpeciesState* species,
             std::size_t offset,
             std::size_t count) const;

private:
    DeviceBuffer<float> _species_ratios; ///< Per-species selection weights (device).

    DeviceBuffer<float> _species_numbers; ///< Species-id values, parallel to ratios.

    float _temperature { 273.15f }; ///< Kelvin; stored but unused when sampling.

    float _base_value { 0.0f }; ///< Center of each velocity component before jitter.

    float _jitter_radius { 0.0f }; ///< Half-width of the uniform jitter; used as |r|.

    Float3 _bulk_velocity { 0.0f, 0.0f, 0.0f }; ///< Constant drift added to velocity.

    unsigned int _seed { atlas::DEFAULT_UNSIGNED_INT_SEED }; ///< Base RNG seed.
};

/**
 * @brief Fluent, validating builder for @c JitteringGenerator.
 *
 * Stages species data and scalars on the host; @c build() validates them, moves
 * the buffers onto the device, returns the leaf, and resets the builder.
 */
class JitteringGenerator::Builder final {
public:
    /// Constructs a builder with all fields at their defaults.
    Builder() = default;

    /**
     * @brief Sets the per-species selection weights (copied from host).
     * @param species_ratios Weights; must be non-empty and match numbers in size.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_species_ratios(const HostBuffer<float>& species_ratios);

    /**
     * @brief Sets the species-id values parallel to the ratios (copied from host).
     * @param species_numbers Species ids; must match @c with_species_ratios size.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_species_numbers(const HostBuffer<float>& species_numbers);

    /**
     * @brief Sets the stored temperature (kelvin); not used by this leaf's sampling.
     * @param temperature Must be finite and non-negative at @c build().
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

    /**
     * @brief Sets the center value of each velocity component.
     * @param base_value Must be finite at @c build().
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_base_value(float base_value) noexcept;

    /**
     * @brief Sets the jitter half-width; the kernel uses its absolute value.
     * @param jitter_radius Must be finite at @c build().
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_jitter_radius(float jitter_radius) noexcept;

    /**
     * @brief Sets the bulk drift added to every sampled velocity.
     * @param bulk_velocity Drift velocity.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_bulk_velocity(const Float3& bulk_velocity) noexcept;

    /**
     * @brief Sets the base RNG seed.
     * @param seed Base seed folded with each particle index.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_seed(unsigned int seed) noexcept;

    /**
     * @brief Validates inputs, moves host data to device, and builds the leaf.
     * @return A ready-to-run @c JitteringGenerator.
     * @throws std::runtime_error if @c validate() fails.
     * @note Resets this builder to defaults on success.
     */
    ATLAS_NODISCARD ATLAS_HOST JitteringGenerator
    build();

    /**
     * @brief Convenience wrapper that builds and wraps in a host shared pointer.
     * @return Shared handle to a newly built leaf.
     * @throws std::runtime_error if @c validate() fails.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<JitteringGenerator>
    make_host_shared();

private:
    /**
     * @brief Enforces the leaf's invariants before construction.
     * @throws std::runtime_error on empty or size-mismatched species buffers, a
     *         non-finite/negative temperature, or a non-finite base/radius.
     */
    ATLAS_HOST void
    validate() const;

private:
    HostBuffer<float> _species_ratios; ///< Staged selection weights (host).

    HostBuffer<float> _species_numbers; ///< Staged species-id values (host).

    float _temperature { 273.15f }; ///< Staged temperature in kelvin.

    float _base_value { 0.0f }; ///< Staged velocity center.

    float _jitter_radius { 0.0f }; ///< Staged jitter half-width.

    Float3 _bulk_velocity { 0.0f, 0.0f, 0.0f }; ///< Staged bulk drift.

    unsigned int _seed { atlas::DEFAULT_UNSIGNED_INT_SEED }; ///< Staged RNG seed.
};

/// Shared owning handle to a host-resident @c JitteringGenerator.
using JitteringGeneratorHostPtr = atlas::host_shared_ptr<JitteringGenerator>;

/// Shared handle to a device-resident @c JitteringGenerator.
using JitteringGeneratorDevicePtr = atlas::device_shared_ptr<JitteringGenerator>;

}