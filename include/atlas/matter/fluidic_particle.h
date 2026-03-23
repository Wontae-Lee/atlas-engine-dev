#pragma once

#include <atlas/matter/matter.h>
#include <atlas/memory/memory.h>

#include <optional>

namespace atlas::system {

/**
 * @brief Fluidic (DSMC-style) particle with extended physical and statistical properties.
 *
 * @details
 * `FluidicParticle` models a single **simulation particle** used in rarefied-gas / DSMC
 * pipelines. In DSMC, one simulation particle commonly represents a *cloud* of real
 * molecules and may carry additional state beyond position/velocity (which is expected
 * to live in the @ref Matter<T> base class).
 *
 * This type stores:
 * - A **mandatory** physical identity parameter: @ref molecular_mass
 * - Optional **statistical** representation: @ref statistical_weight
 * - Optional **energy mode decomposition** (translational/rotational/vibrational/electronic)
 * - Optional **species/state** identifiers and collision model parameters (VHS/VSS)
 * - Optional **charge** for plasma-like extensions
 *
 * The class is intended to be a lightweight aggregate of parameters, while **construction
 * and validation** is performed via the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., `float`, `double`).
 *
 * @par Host/device notes
 * - The particle data container itself can be used in host/device contexts.
 * - The @ref Builder is **host-only** because it may throw exceptions and uses
 *   host-side ownership helpers (e.g., `host_shared_ptr`).
 *
 * @warning
 * Many algorithms assume that a particle has a physically valid configuration.
 * If you bypass the @ref Builder and mutate members directly, you are responsible
 * for ensuring:
 * - @ref molecular_mass is set and positive (or at least non-zero),
 * - collision parameters are consistent with the chosen model,
 * - energy values (if set) follow your simulation’s conventions.
 */
template <typename T>
class FluidicParticle final : public Matter<T> {
public:
    /**
     * @brief Fluent builder for configuring and constructing @ref FluidicParticle.
     *
     * @details
     * The builder enforces required fields and provides optional parameter setters.
     * See @ref FluidicParticle<T>::Builder for the full API.
     */
    class Builder;

public:
    // ------------------------------------------------------------------
    // Mandatory physical property
    // ------------------------------------------------------------------

    /**
     * @brief Molecular mass of a single real molecule.
     *
     * @details
     * This is the **only mandatory** scalar in this class. It defines the physical
     * identity of the gas species at the molecule level.
     *
     * Typical usage in DSMC:
     * - Used to relate velocities to momentum and energy.
     * - Used when sampling macroscopic properties (e.g., density, temperature).
     *
     * @note
     * - Default construction leaves this value at `T{}` (commonly 0).
     * - For physically meaningful particles, set this via @ref Builder::with_molecular_mass.
     */
    T molecular_mass {};

    // ------------------------------------------------------------------
    // Statistical / DSMC-specific
    // ------------------------------------------------------------------

    /**
     * @brief Statistical weight (number of real molecules represented).
     *
     * @details
     * In DSMC, a simulation particle often represents many real molecules.
     * This parameter (sometimes called "weight" or "representative number")
     * encodes that multiplicity.
     *
     * Common interpretations:
     * - If set, `statistical_weight = N` means the particle stands for `N` real molecules.
     * - If unset, your code may assume an implicit weight (often 1), or treat it as unknown.
     *
     * @note
     * - Unset (`std::nullopt`) is semantically different from zero.
     * - If your physics requires weights, treat unset as an error at the call site or enforce
     *   it in the builder validation.
     */
    std::optional<T> statistical_weight;

    // ------------------------------------------------------------------
    // Energy modes
    // ------------------------------------------------------------------

    /**
     * @brief Translational energy of the particle.
     *
     * @details
     * Represents the translational kinetic contribution (often related to the particle's
     * velocity distribution). The unit and definition are simulation-convention dependent.
     *
     * Examples of conventions:
     * - Energy per real molecule
     * - Energy per simulation particle (already multiplied by statistical weight)
     *
     * @note
     * This is optional. If unset, the simulation may derive it from velocity or treat it as
     * not tracked explicitly.
     */
    std::optional<T> translational_energy;

    /**
     * @brief Rotational internal energy.
     *
     * @details
     * Used for multi-DOF gases where rotational modes are modeled separately.
     * Typically used in DSMC with internal energy exchange models.
     */
    std::optional<T> rotational_energy;

    /**
     * @brief Vibrational internal energy.
     *
     * @details
     * Used for vibrational modes in high-temperature flows, often with relaxation models.
     */
    std::optional<T> vibrational_energy;

    // ------------------------------------------------------------------
    // Species / state information
    // ------------------------------------------------------------------

    /**
     * @brief Species identifier.
     *
     * @details
     * Associates this particle with a species table (collision parameters, internal DOF,
     * reaction models, etc.). The interpretation is framework-specific.
     *
     * Typical usage:
     * - Index into arrays of species constants (mass, collision diameter, VHS/VSS params)
     * - Partitioning particles by species for collision pairing
     *
     * @note
     * If unset, the particle is considered "untyped" unless your simulation provides a default.
     */
    std::optional<int> species_id;

    // ------------------------------------------------------------------
    // Collision model parameters (species-fixed)
    // ------------------------------------------------------------------

    /**
     * @brief Reference collision diameter (`d_ref`).
     *
     * @details
     * Collision diameter used by collision models such as VHS (Variable Hard Sphere)
     * and VSS (Variable Soft Sphere).
     *
     * @note
     * - If your simulation uses a species database, you may prefer storing these parameters
     *   externally and only keep @ref species_id here.
     * - If stored here, ensure consistency with @ref viscosity_index and
     *   @ref scattering_parameter (for VSS).
     */
    std::optional<T> collision_diameter;

    /**
     * @brief Viscosity index ω for VHS/VSS.
     *
     * @details
     * Used to model temperature dependence of viscosity and collision cross sections.
     */
    std::optional<T> viscosity_index;

    /**
     * @brief VSS scattering parameter α (VSS only).
     *
     * @details
     * Determines the angular scattering characteristics in the VSS model.
     *
     * @note
     * This parameter is generally not used for VHS.
     */
    std::optional<T> scattering_parameter;

    // ------------------------------------------------------------------
    // Advanced / optional models
    // ------------------------------------------------------------------

    /**
     * @brief Electronic excitation energy.
     *
     * @details
     * For high-temperature gases, electronic modes may be relevant and sometimes modeled
     * explicitly. This field stores the electronic energy contribution if tracked.
     */
    std::optional<T> electronic_energy;

    /**
     * @brief Particle charge.
     *
     * @details
     * Optional integer charge state used for plasma models or charged species.
     *
     * @note
     * Interpretation (units, sign convention) is simulation-specific.
     */
    std::optional<int> charge;

public:
    /**
     * @brief Default constructor creates an "unset" particle.
     *
     * @details
     * Constructs a particle whose optional properties are unset (`std::nullopt`) and whose
     * mandatory scalar @ref molecular_mass is default-initialized.
     *
     * @note
     * Prefer building particles via @ref builder() / @ref Builder for stronger invariants.
     *
     * @par Performance
     * Marked `ATLAS_ALL_DEVICE` and `ATLAS_FORCE_INLINE` to allow trivial construction in
     * both host and device contexts (where supported).
     */
    FluidicParticle() = default;

    /**
     * @brief Entry point for fluent particle construction.
     *
     * @details
     * Returns a builder used to set required and optional parameters in a fluent style.
     *
     * @return A default-initialized @ref Builder.
     *
     * @note
     * Host-only because the builder may throw exceptions at build time.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;
};

/* ====================================================================== */
/* Builder                                                                 */
/* ====================================================================== */

/**
 * @brief Fluent builder for @ref FluidicParticle.
 *
 * @details
 * The builder is responsible for constructing a physically meaningful particle configuration.
 *
 * ## Responsibilities
 * - Enforce presence of required parameters (currently @ref with_molecular_mass).
 * - Offer a fluent interface for setting optional DSMC parameters.
 * - Validate the configuration at build time via @ref validate.
 *
 * ## Error handling
 * - If required fields are missing, @ref build and @ref make_host_shared throw `std::runtime_error`.
 *
 * ## Ownership
 * - @ref build constructs a particle by value.
 * - @ref make_host_shared constructs a particle managed by `atlas::host_shared_ptr`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 *
 * @note
 * Host-only: this class uses exceptions and host memory utilities.
 */
template <typename T>
class FluidicParticle<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates an empty builder with no required fields set. You must call
     * @ref with_molecular_mass before building.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref FluidicParticle (by value).
     *
     * @details
     * Validates the builder state and returns a particle populated with all configured
     * parameters. Required parameters are copied into the resulting particle; unset
     * optionals remain unset.
     *
     * @return Fully constructed particle instance.
     *
     * @throws std::runtime_error
     * If required fields (e.g., molecular mass) are missing or validation fails.
     *
     * @note
     * This is a host-only routine.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE FluidicParticle<T>
    build() const;

    /**
     * @brief Build a configured @ref FluidicParticle in a @ref atlas::host_shared_ptr.
     *
     * @details
     * Equivalent to `host_shared_ptr<FluidicParticle<T>>(new FluidicParticle<T>(build()))`
     * (exact allocation strategy depends on `atlas::host_shared_ptr` implementation).
     *
     * @return Shared pointer owning the constructed particle.
     *
     * @throws std::runtime_error
     * If required fields are missing or validation fails.
     *
     * @note
     * Host-only routine; not intended for device code.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<FluidicParticle<T>>
    make_host_shared() const;

    /* ------------------------------------------------------------------
     * Required
     * ------------------------------------------------------------------ */

    /**
     * @brief Set the molecular mass (mandatory).
     *
     * @details
     * Sets the molecular mass of the species. This is required for building.
     *
     * @param mass Molecular mass of a single molecule.
     * @return `*this` for fluent chaining.
     *
     * @note
     * This function does not validate sign or unit; validation policy is left to
     * @ref validate (and/or your calling conventions).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_molecular_mass(T mass);

    /* ------------------------------------------------------------------
     * Optional (fluent)
     * ------------------------------------------------------------------ */

    /**
     * @brief Set statistical weight.
     *
     * @param w Statistical weight (number of real molecules represented).
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_statistical_weight(T w);

    /**
     * @brief Set translational energy.
     *
     * @param e Translational energy value (convention-dependent).
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_translational_energy(T e);

    /**
     * @brief Set rotational energy.
     *
     * @param e Rotational internal energy.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rotational_energy(T e);

    /**
     * @brief Set vibrational energy.
     *
     * @param e Vibrational internal energy.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vibrational_energy(T e);

    /**
     * @brief Set species identifier.
     *
     * @param id Species ID (index into a species table, etc.).
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_species_id(int id);

    /**
     * @brief Set reference collision diameter (`d_ref`).
     *
     * @param d_ref Reference collision diameter.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collision_diameter(T d_ref);

    /**
     * @brief Set viscosity index ω.
     *
     * @param omega Viscosity index ω used in VHS/VSS.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_viscosity_index(T omega);

    /**
     * @brief Set VSS scattering parameter α.
     *
     * @param alpha Scattering parameter α (VSS).
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_scattering_parameter(T alpha);

    /**
     * @brief Set electronic excitation energy.
     *
     * @param e Electronic energy value.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_electronic_energy(T e);

    /**
     * @brief Set charge.
     *
     * @param q Charge state.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_charge(int q);

private:
    /**
     * @brief Validate builder state or throw an exception.
     *
     * @details
     * Performs build-time validation, at minimum ensuring that required parameters are set.
     * Additional policy checks may be implemented in `fluidic_particle.hpp`, e.g.:
     * - `molecular_mass` is positive,
     * - collision parameters are provided consistently,
     * - energy values are non-negative,
     * - species_id is valid in a given registry (if applicable).
     *
     * @throws std::runtime_error if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    // ------------------------------------------------------------------
    // Builder storage
    // ------------------------------------------------------------------
    // The builder stores required and optional parameters as optionals so
    // it can distinguish between "unset" and "set to zero".
    // ------------------------------------------------------------------

    /// @brief Required: molecular mass must be set before building.
    std::optional<T> _molecular_mass;

    /// @brief Optional: statistical weight.
    std::optional<T> _statistical_weight;

    /// @brief Optional: translational energy.
    std::optional<T> _translational_energy;

    /// @brief Optional: rotational energy.
    std::optional<T> _rotational_energy;

    /// @brief Optional: vibrational energy.
    std::optional<T> _vibrational_energy;

    /// @brief Optional: species identifier.
    std::optional<int> _species_id;

    /// @brief Optional: collision diameter parameter.
    std::optional<T> _collision_diameter;

    /// @brief Optional: viscosity index parameter.
    std::optional<T> _viscosity_index;

    /// @brief Optional: scattering parameter (VSS).
    std::optional<T> _scattering_parameter;

    /// @brief Optional: electronic energy.
    std::optional<T> _electronic_energy;

    /// @brief Optional: particle charge.
    std::optional<int> _charge;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using FluidicParticle = system::FluidicParticle<T>;

template <typename T>
using FluidicParticleHostPtr = atlas::host_shared_ptr<system::FluidicParticle<T>>;

template <typename T>
using FluidicParticleDevicePtr = atlas::device_shared_ptr<system::FluidicParticle<T>>;

}

// Implementation header for templates.
#include <atlas/matter/fluidic_particle.hpp>
