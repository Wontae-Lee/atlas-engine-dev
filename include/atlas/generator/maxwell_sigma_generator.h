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
 * @brief Generator leaf that draws velocities from an isotropic Gaussian with a
 *        caller-supplied standard deviation, plus a bulk drift.
 *
 * The velocity is @c sample_normal_vector(sigma) + bulk_velocity when
 * @c sigma > 0; when @c sigma == 0 every particle gets exactly the bulk drift.
 * Unlike @c MaxwellBoltzmannGenerator, the spread here is given directly rather
 * than derived from temperature and mass, so @c _temperature is stored and
 * validated but does not influence the sampled velocity. Species selection is
 * the shared weighted draw over ratios and numbers.
 *
 * Owns two @c DeviceBuffer members, so the leaf is move-only. Build via @c Builder.
 *
 * @note Satisfies @c ConceptGenerator; stored in the @c Generator union.
 */
class MaxwellSigmaGenerator final {
public:
    /// Validating factory that assembles a @c MaxwellSigmaGenerator; see below.
    class Builder;

public:
    /// Constructs an empty leaf with no species data; @c generate yields 0.
    MaxwellSigmaGenerator() = default;

    /**
     * @brief Constructs a ready-to-run leaf, taking ownership of the buffers.
     *
     * @param species_ratios  Per-species selection weights (device).
     * @param species_numbers Species-id values (device), parallel to the ratios.
     * @param temperature     Gas temperature in kelvin; stored, not used to sample.
     * @param sigma           Gaussian standard deviation per component; 0 disables.
     * @param bulk_velocity   Constant drift added to every sampled velocity.
     * @param seed            Base seed folded with the particle index per draw.
     */
    ATLAS_HOST
    MaxwellSigmaGenerator(DeviceBuffer<float> species_ratios,
                          DeviceBuffer<float> species_numbers,
                          float temperature,
                          float sigma,
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
     * @c shuffle_key(index, seed), picks a species, and writes a Gaussian
     * velocity (or just the bulk drift when @c sigma <= 0). Silently no-ops on
     * null pointers, zero count, or an offset past the buffers' shared capacity.
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

    float _sigma { 0.0f }; ///< Gaussian standard deviation; 0 means bulk-only.

    Float3 _bulk_velocity { 0.0f, 0.0f, 0.0f }; ///< Constant drift added to velocity.

    unsigned int _seed { atlas::DEFAULT_UNSIGNED_INT_SEED }; ///< Base RNG seed.
};

/**
 * @brief Fluent, validating builder for @c MaxwellSigmaGenerator.
 *
 * Stages species data and scalars on the host; @c build() validates them, moves
 * the buffers onto the device, returns the leaf, and resets the builder.
 */
class MaxwellSigmaGenerator::Builder final {
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
     * @brief Sets the Gaussian standard deviation of the velocity components.
     * @param sigma Must be finite and non-negative at @c build(); 0 means bulk-only.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_sigma(float sigma) noexcept;

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
     * @return A ready-to-run @c MaxwellSigmaGenerator.
     * @throws std::runtime_error if @c validate() fails.
     * @note Resets this builder to defaults on success.
     */
    ATLAS_NODISCARD ATLAS_HOST MaxwellSigmaGenerator
    build();

    /**
     * @brief Convenience wrapper that builds and wraps in a host shared pointer.
     * @return Shared handle to a newly built leaf.
     * @throws std::runtime_error if @c validate() fails.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<MaxwellSigmaGenerator>
    make_host_shared();

private:
    /**
     * @brief Enforces the leaf's invariants before construction.
     * @throws std::runtime_error on empty or size-mismatched species buffers, a
     *         non-finite/negative temperature, or a non-finite/negative sigma.
     */
    ATLAS_HOST void
    validate() const;

private:
    HostBuffer<float> _species_ratios; ///< Staged selection weights (host).

    HostBuffer<float> _species_numbers; ///< Staged species-id values (host).

    float _temperature { 273.15f }; ///< Staged temperature in kelvin.

    float _sigma { 0.0f }; ///< Staged Gaussian standard deviation.

    Float3 _bulk_velocity { 0.0f, 0.0f, 0.0f }; ///< Staged bulk drift.

    unsigned int _seed { atlas::DEFAULT_UNSIGNED_INT_SEED }; ///< Staged RNG seed.
};

/// Shared owning handle to a host-resident @c MaxwellSigmaGenerator.
using MaxwellSigmaGeneratorHostPtr = atlas::host_shared_ptr<MaxwellSigmaGenerator>;

/// Shared handle to a device-resident @c MaxwellSigmaGenerator.
using MaxwellSigmaGeneratorDevicePtr = atlas::device_shared_ptr<MaxwellSigmaGenerator>;

}