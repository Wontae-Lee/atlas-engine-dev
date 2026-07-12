#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/random/seed.h>

#include <cstddef>
#include <utility>

namespace atlas {

/**
 * @brief Generator leaf that draws velocities from a Maxwell-Boltzmann
 *        distribution whose spread is derived from temperature and species mass.
 *
 * This is the only leaf that turns @c _temperature into physics: for a chosen
 * species of molecular mass @c m the per-component standard deviation is
 * @c sqrt(k_B * T / m), and the velocity is @c sample_normal_vector(sigma) plus
 * the bulk drift. When either temperature or mass is non-positive the particle
 * receives only the bulk drift.
 *
 * Species selection differs subtly from the other leaves: it draws a weighted
 * *index* into the parallel arrays (@c sample_weighted_index over the ratios),
 * then reads both the species id (@c species_numbers[index]) and the mass
 * (@c species_mass[index]) at that index, so id and mass stay consistent.
 *
 * Owns three @c DeviceBuffer members, so the leaf is move-only. Build via
 * @c Builder, which can source per-species masses either directly or by looking
 * them up in a @c MaterialDictionary.
 *
 * @note Satisfies @c ConceptGenerator; stored in the @c Generator union.
 */
class MaxwellBoltzmannGenerator final {
public:
    /// Validating factory that assembles a @c MaxwellBoltzmannGenerator; see below.
    class Builder;

public:
    /// Constructs an empty leaf with no species data; @c generate yields 0.
    MaxwellBoltzmannGenerator() = default;

    /**
     * @brief Constructs a ready-to-run leaf, taking ownership of the buffers.
     *
     * @param species_ratios  Per-species selection weights (device).
     * @param species_numbers Species-id values (device), parallel to the ratios.
     * @param species_mass    Molecular mass per species (device), same indexing.
     * @param temperature     Gas temperature in kelvin; drives the Gaussian sigma.
     * @param bulk_velocity   Constant drift added to every sampled velocity.
     * @param seed            Base seed folded with the particle index per draw.
     */
    ATLAS_HOST
    MaxwellBoltzmannGenerator(DeviceBuffer<float> species_ratios,
                              DeviceBuffer<float> species_numbers,
                              DeviceBuffer<float> species_mass,
                              float temperature,
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

    /// @return The temperature in kelvin used to derive the Gaussian sigma.
    ATLAS_NODISCARD ATLAS_HOST float
    temperature() const noexcept {
        return _temperature;
    }

    /**
     * @brief Fills @c [offset, offset + count) with sampled species and velocities.
     *
     * Runs a device kernel over the clamped range; each thread seeds an RNG from
     * @c shuffle_key(index, seed), draws a species index, writes that species'
     * id, and writes a temperature/mass-derived Gaussian velocity plus bulk (or
     * just the bulk drift when temperature or mass is non-positive). Silently
     * no-ops on null pointers, zero count, or an offset past the shared capacity.
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

    DeviceBuffer<float> _species_mass; ///< Molecular mass per species (device).

    float _temperature { 273.15f }; ///< Kelvin; drives the per-species Gaussian sigma.

    Float3 _bulk_velocity { 0.0f, 0.0f, 0.0f }; ///< Constant drift added to velocity.

    unsigned int _seed { atlas::DEFAULT_UNSIGNED_INT_SEED }; ///< Base RNG seed.
};

/**
 * @brief Fluent, validating builder for @c MaxwellBoltzmannGenerator.
 *
 * Besides the shared species/scalar staging, this builder resolves the
 * per-species molecular masses the leaf needs. Masses may be supplied two ways:
 * directly via @c with_species_mass (one value per selectable species, indexed
 * like the ratios), or indirectly via @c with_material_dictionary, which caches
 * a mass-by-material-id table so @c build() can look up the mass for each
 * species id. If both are given, the explicit species masses win.
 */
class MaxwellBoltzmannGenerator::Builder final {
public:
    /// Constructs a builder with all fields at their defaults.
    Builder() = default;

    /**
     * @brief Sets the per-species selection weights (copied from host).
     *
     * The weights need not be normalized: @c build() rescales them to sum to 1, so
     * raw population counts or percentages (e.g. @c {70, 30}) select each species
     * with the intended probability.
     *
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
     * @brief Caches a mass-by-material-id table from a material dictionary.
     *
     * Reads @c mass() for every material into an internal table indexed by
     * material (species) id. Used at @c build() to resolve masses when no
     * explicit @c with_species_mass was provided.
     *
     * @param material_dictionary Dictionary whose materials supply the masses.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_material_dictionary(const MaterialDictionary& material_dictionary);

    /**
     * @brief Sets explicit per-species molecular masses (copied from host).
     *
     * When set, this takes precedence over the material-dictionary lookup and
     * must have one entry per selectable species, indexed like the ratios.
     *
     * @param species_mass Masses parallel to the ratios/numbers arrays.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_species_mass(const HostBuffer<float>& species_mass);

    /**
     * @brief Sets the gas temperature that drives the Gaussian sigma.
     * @param temperature Kelvin; must be finite and non-negative at @c build().
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

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
     * @brief Validates inputs, resolves masses, normalizes ratios, builds.
     *
     * The staged selection weights are rescaled to sum to 1 before upload (see
     * @c with_species_ratios), so @c sample_weighted_index draws each species with
     * its intended probability regardless of the input scale.
     *
     * @return A ready-to-run @c MaxwellBoltzmannGenerator.
     * @throws std::runtime_error if @c validate() fails or a species id is out of
     *         the dictionary's range during mass resolution.
     * @note Resets this builder to defaults on success.
     */
    ATLAS_NODISCARD ATLAS_HOST MaxwellBoltzmannGenerator
    build();

    /**
     * @brief Convenience wrapper that builds and wraps in a host shared pointer.
     * @return Shared handle to a newly built leaf.
     * @throws std::runtime_error if @c build() fails.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<MaxwellBoltzmannGenerator>
    make_host_shared();

private:
    /**
     * @brief Produces the per-species mass array the leaf will own.
     *
     * Returns the explicit @c _species_mass verbatim when present; otherwise maps
     * each species id in @c _species_numbers through the cached @c _material_mass
     * table to build a parallel mass array.
     *
     * @return One mass per selectable species, indexed like the ratios/numbers.
     * @throws std::runtime_error if a species id falls outside the mass table.
     */
    ATLAS_HOST HostBuffer<float>
    resolve_species_mass() const;

    /**
     * @brief Enforces the leaf's invariants before construction.
     * @throws std::runtime_error on empty or size-mismatched species buffers, an
     *         explicit-mass array whose size differs from the species count, the
     *         absence of both a mass array and a dictionary, or a
     *         non-finite/negative temperature.
     */
    ATLAS_HOST void
    validate() const;

private:
    HostBuffer<float> _species_ratios; ///< Staged selection weights (host).

    HostBuffer<float> _species_numbers; ///< Staged species-id values (host).

    HostBuffer<float> _species_mass; ///< Optional explicit per-species masses.

    HostBuffer<float> _material_mass; ///< Cached mass-by-material-id lookup table.

    float _temperature { 273.15f }; ///< Staged temperature in kelvin.

    Float3 _bulk_velocity { 0.0f, 0.0f, 0.0f }; ///< Staged bulk drift.

    unsigned int _seed { atlas::DEFAULT_UNSIGNED_INT_SEED }; ///< Staged RNG seed.
};

/// Shared owning handle to a host-resident @c MaxwellBoltzmannGenerator.
using MaxwellBoltzmannGeneratorHostPtr = atlas::host_shared_ptr<MaxwellBoltzmannGenerator>;

/// Shared handle to a device-resident @c MaxwellBoltzmannGenerator.
using MaxwellBoltzmannGeneratorDevicePtr = atlas::device_shared_ptr<MaxwellBoltzmannGenerator>;

}