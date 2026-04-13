#pragma once

/**
 * @file matrail_properties.h
 * @brief Declares material classification tags, material-property storage, and builder utilities.
 *
 * @details
 * This header defines:
 * - @ref atlas::system::MaterialType, a lightweight categorical tag describing
 *   the physical class of a material or species,
 * - @ref atlas::system::MatrialProperties, a value object that stores the
 *   physical parameters associated with a simulation material/species,
 * - a nested fluent @ref Builder used to stage and validate material-property
 *   construction.
 *
 * ## Purpose
 * Material-property records are used throughout Atlas to describe the physical
 * characteristics of particles or species that participate in simulation.
 *
 * Depending on the simulation model, these properties may be used for:
 * - thermodynamic interpretation,
 * - kinetic-theory calculations,
 * - collision models,
 * - transport-property evaluation,
 * - continuum-style constitutive parameters,
 * - species identification and charge-state handling.
 *
 * ## Optional-property model
 * Many material attributes are represented using `std::optional` because not all
 * simulation workflows require the same physical fields.
 *
 * This allows a material record to describe a broad range of entities, including:
 * - molecules,
 * - atoms,
 * - ions,
 * - neutrons,
 * - solids,
 * while only populating the subset of properties relevant to the active model.
 *
 * ## Builder-based construction
 * Although @ref MatrialProperties is a simple value type, the nested
 * @ref Builder provides a controlled construction path that can:
 * - stage required and optional parameters,
 * - validate consistency before finalization,
 * - return either a value instance or a host-owned shared pointer.
 *
 * ## Naming note
 * The type name is spelled `MatrialProperties` in the current API and is kept as-is
 * for compatibility with the surrounding codebase.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for physical quantities such as mass,
 *         energy, viscosity, density, and geometric scales.
 */

#include <atlas/memory/memory.h>

#include <optional>

namespace atlas::system {

/**
 * @brief Categorical classification of a material/species entity.
 *
 * @details
 * @ref MaterialType provides a compact runtime tag describing the physical class
 * of a material record.
 *
 * The tag can be used to distinguish between materially different simulation
 * entities whose relevant optional properties may differ significantly.
 */
struct MaterialType final {
    /**
     * @brief Enumerates the supported material categories.
     */
    enum Value : int {
        /**
         * @brief Molecular species.
         *
         * @details
         * Typically used for gas-phase molecular particles in kinetic or
         * thermodynamic simulations.
         */
        Molecule,

        /**
         * @brief Atomic species.
         *
         * @details
         * Represents neutral atomic particles.
         */
        Atom,

        /**
         * @brief Ionic species.
         *
         * @details
         * Represents charged atomic or molecular species.
         */
        Ion,

        /**
         * @brief Neutron-like species.
         *
         * @details
         * Represents neutral nuclear particles where a neutron model is needed.
         */
        Neutron,

        /**
         * @brief Solid material.
         *
         * @details
         * Represents solid-phase material models, typically used in continuum
         * or particle-solid interaction workflows.
         */
        Solid
    };
};

/**
 * @brief Stores the physical properties associated with a simulation material/species.
 *
 * @details
 * @ref MatrialProperties is a lightweight value object that aggregates the
 * physical parameters used to describe a material or species in Atlas.
 *
 * The record contains:
 * - a required material-type tag,
 * - a required mass,
 * - a collection of optional scalar and integer properties that may or may not
 *   be relevant depending on the simulation model.
 *
 * ## Typical usage
 * A material-property record may describe:
 * - a gas species used in particle emission,
 * - a collision-model participant,
 * - a fluid particle material definition,
 * - a solid-like material for continuum or hybrid models.
 *
 * ## Optional fields
 * Optional properties allow a single type to cover a wide range of modeling
 * needs without forcing all fields to be present. For example:
 * - collision-diameter-related fields are relevant to kinetic collisions,
 * - density, viscosity, and smoothing length are more relevant to SPH-like models,
 * - charge and electronic energy are relevant to ionized species.
 *
 * ## Builder support
 * A nested @ref Builder is provided for fluent staged construction and
 * implementation-defined validation.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for physical quantities.
 */
template <typename T>
class MatrialProperties final {
public:
    /**
     * @brief Fluent builder for configuring and constructing @ref MatrialProperties.
     *
     * @details
     * The builder stages required and optional physical parameters, validates them,
     * and constructs either:
     * - a material-property record by value, or
     * - a host-owned shared pointer to such a record.
     */
    class Builder;

public:
    /**
     * @brief Material classification tag.
     *
     * @details
     * Identifies the physical class of the material/species.
     */
    MaterialType::Value type {};

    /**
     * @brief Material or particle mass.
     *
     * @details
     * Stores the fundamental mass associated with the material/species.
     */
    T mass {};

    /**
     * @brief Optional statistical weight.
     *
     * @details
     * May be used to represent the number of physical particles represented by a
     * simulation particle or sample.
     */
    std::optional<T> statistical_weight;

    /**
     * @brief Optional translational energy.
     *
     * @details
     * May represent a translational-mode energy parameter relevant to the active model.
     */
    std::optional<T> translational_energy;

    /**
     * @brief Optional rotational energy.
     *
     * @details
     * May represent rotational-mode energy content for molecular species.
     */
    std::optional<T> rotational_energy;

    /**
     * @brief Optional vibrational energy.
     *
     * @details
     * May represent vibrational-mode energy content for molecular species.
     */
    std::optional<T> vibrational_energy;

    /**
     * @brief Optional species identifier.
     *
     * @details
     * Can be used as a stable application-level id for species lookup and indexing.
     */
    std::optional<int> species_id;

    /**
     * @brief Optional collision diameter.
     *
     * @details
     * Commonly used in kinetic-theory and collision cross-section models.
     */
    std::optional<T> collision_diameter;

    /**
     * @brief Optional viscosity index.
     *
     * @details
     * May be used in transport-property and variable-hard-sphere-like models.
     */
    std::optional<T> viscosity_index;

    /**
     * @brief Optional scattering parameter.
     *
     * @details
     * May parameterize scattering or collision-angle behavior in rarefied-gas models.
     */
    std::optional<T> scattering_parameter;

    /**
     * @brief Optional reference/rest density.
     *
     * @details
     * Often used in continuum or particle-fluid formulations such as SPH.
     */
    std::optional<T> rest_density;

    /**
     * @brief Optional pressure coefficient.
     *
     * @details
     * May be used in constitutive or equation-of-state style models.
     */
    std::optional<T> pressure_coefficient;

    /**
     * @brief Optional dynamic viscosity.
     *
     * @details
     * Represents the shear viscosity parameter when relevant to the active model.
     */
    std::optional<T> dynamic_viscosity;

    /**
     * @brief Optional smoothing length.
     *
     * @details
     * Commonly used in SPH-like particle methods as a kernel support radius.
     */
    std::optional<T> smoothing_length;

    /**
     * @brief Optional electronic energy.
     *
     * @details
     * May represent electron-level or excitation-related energy content.
     */
    std::optional<T> electronic_energy;

    /**
     * @brief Optional charge state.
     *
     * @details
     * Integer charge associated with ions or charged materials.
     */
    std::optional<int> charge;

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a material-property record with default-initialized required
     * fields and all optional fields disengaged.
     */
    MatrialProperties() = default;

    /**
     * @brief Builder entry point.
     *
     * @details
     * Returns a default-initialized @ref Builder for fluent staged construction.
     *
     * @return Fresh builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;
};

/**
 * @brief Fluent builder for @ref MatrialProperties.
 *
 * @details
 * The builder provides a controlled way to configure and construct a material
 * record while separating:
 * - required physical parameters,
 * - optional model-specific parameters,
 * - validation prior to finalization.
 *
 * ## Typical usage
 * @code
 * auto material = atlas::MatrialProperties<float>::builder()
 *     .with_type(atlas::MaterialType::Molecule)
 *     .with_mass(4.65e-26f)
 *     .with_species_id(0)
 *     .with_collision_diameter(3.7e-10f)
 *     .with_viscosity_index(0.81f)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - required fields such as type and mass are present,
 * - mass is physically admissible,
 * - optional physical quantities are finite and valid when supplied,
 * - cross-field consistency is maintained where required.
 *
 * The exact validation rules are implementation-defined in
 * `matrial_properties.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class MatrialProperties<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with no staged required fields and no optional physical
     * parameters set.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref MatrialProperties by value after validation.
     *
     * @return Constructed material-property record.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE MatrialProperties<T>
    build() const;

    /**
     * @brief Build a configured @ref MatrialProperties in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<MatrialProperties<T>>` owning the constructed record.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MatrialProperties<T>>
    make_host_shared() const;

    /**
     * @brief Set the material classification tag.
     *
     * @param t Material category to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_type(MaterialType::Value t);

    /**
     * @brief Set the material mass.
     *
     * @param m Mass value to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_mass(T m);

    /**
     * @brief Set the statistical weight.
     *
     * @param w Statistical weight to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_statistical_weight(T w);

    /**
     * @brief Set the translational energy.
     *
     * @param e Translational energy to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_translational_energy(T e);

    /**
     * @brief Set the rotational energy.
     *
     * @param e Rotational energy to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rotational_energy(T e);

    /**
     * @brief Set the vibrational energy.
     *
     * @param e Vibrational energy to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vibrational_energy(T e);

    /**
     * @brief Set the species identifier.
     *
     * @param id Species id to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_species_id(int id);

    /**
     * @brief Set the collision diameter.
     *
     * @param d_ref Collision diameter to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collision_diameter(T d_ref);

    /**
     * @brief Set the viscosity index.
     *
     * @param omega Viscosity index to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_viscosity_index(T omega);

    /**
     * @brief Set the scattering parameter.
     *
     * @param alpha Scattering parameter to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_scattering_parameter(T alpha);

    /**
     * @brief Set the reference/rest density.
     *
     * @param rho0 Rest density to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rest_density(T rho0);

    /**
     * @brief Set the pressure coefficient.
     *
     * @param k Pressure coefficient to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_pressure_coefficient(T k);

    /**
     * @brief Set the dynamic viscosity.
     *
     * @param mu Dynamic viscosity to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dynamic_viscosity(T mu);

    /**
     * @brief Set the smoothing length.
     *
     * @param h Smoothing length to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_smoothing_length(T h);

    /**
     * @brief Set the electronic energy.
     *
     * @param e Electronic energy to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_electronic_energy(T e);

    /**
     * @brief Set the integer charge state.
     *
     * @param q Charge value to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_charge(int q);

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on required and optional material parameters.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending material classification tag.
     */
    std::optional<MaterialType::Value> _type;

    /**
     * @brief Pending material mass.
     */
    std::optional<T> _mass;

    /**
     * @brief Pending statistical weight.
     */
    std::optional<T> _statistical_weight;

    /**
     * @brief Pending translational energy.
     */
    std::optional<T> _translational_energy;

    /**
     * @brief Pending rotational energy.
     */
    std::optional<T> _rotational_energy;

    /**
     * @brief Pending vibrational energy.
     */
    std::optional<T> _vibrational_energy;

    /**
     * @brief Pending species id.
     */
    std::optional<int> _species_id;

    /**
     * @brief Pending collision diameter.
     */
    std::optional<T> _collision_diameter;

    /**
     * @brief Pending viscosity index.
     */
    std::optional<T> _viscosity_index;

    /**
     * @brief Pending scattering parameter.
     */
    std::optional<T> _scattering_parameter;

    /**
     * @brief Pending rest density.
     */
    std::optional<T> _rest_density;

    /**
     * @brief Pending pressure coefficient.
     */
    std::optional<T> _pressure_coefficient;

    /**
     * @brief Pending dynamic viscosity.
     */
    std::optional<T> _dynamic_viscosity;

    /**
     * @brief Pending smoothing length.
     */
    std::optional<T> _smoothing_length;

    /**
     * @brief Pending electronic energy.
     */
    std::optional<T> _electronic_energy;

    /**
     * @brief Pending charge state.
     */
    std::optional<int> _charge;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::MaterialType.
 */
using MaterialType = system::MaterialType;

/**
 * @brief Convenience alias for @ref atlas::system::MatrialProperties.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using MatrialProperties = system::MatrialProperties<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to
 *        @ref atlas::system::MatrialProperties.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using MatrialPropertiesHostPtr = atlas::host_shared_ptr<system::MatrialProperties<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to
 *        @ref atlas::system::MatrialProperties.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using MatrialPropertiesDevicePtr = atlas::device_shared_ptr<system::MatrialProperties<T>>;

} // namespace atlas

#include <atlas/material/matrial_properties.hpp>