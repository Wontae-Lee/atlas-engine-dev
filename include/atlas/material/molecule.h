#pragma once

#include <atlas/core/macros.h>

namespace atlas {

/**
 * @brief Polyatomic-species leaf of the `Material` tagged union.
 *
 * `Molecule` is a self-contained, trivially copyable bundle of the physical
 * constants a single species contributes to the DSMC collision model. Because
 * it holds nothing but `float`s it satisfies `ConceptMaterial`, is safe to copy
 * into device memory, and can be captured by value inside a device lambda.
 *
 * All eight properties are *always present* — plain `float`s rather than
 * `std::optional<float>` — and are supplied through the constructor. The four
 * energy fields carry the molecule's per-mode internal energy; the four
 * remaining fields are the Variable Hard Sphere / Variable Soft Sphere (VHS/VSS)
 * cross-section parameters consumed by the DSMC kernels.
 *
 * @see ConceptMaterial, Material, MaterialType::molecule
 */
class Molecule final {
public:
    /**
     * @brief Constructs a molecule with every property zero-initialized.
     *
     * Required so `Molecule` can be the default-active member of the `Material`
     * union and so it can live in device buffers before being filled in.
     * `_viscosity_index` and `_scattering_parameter` retain their in-class
     * defaults (0.5f and 1.0f) rather than becoming 0.0f.
     */
    Molecule() = default;

    /**
     * @brief Constructs a molecule from its full set of physical constants.
     *
     * Host- and device-callable so a leaf can be assembled either on the host
     * (dictionary construction) or inside a device lambda. `noexcept`: it only
     * copies floats.
     *
     * @param mass Species mass in kilograms.
     * @param translational_energy Translational internal energy in joules.
     * @param rotational_energy Rotational internal energy in joules.
     * @param vibrational_energy Vibrational internal energy in joules.
     * @param reference_diameter VHS/VSS reference molecular diameter d_ref in
     *        metres; must be positive for the collision cross section to be
     *        finite.
     * @param reference_temperature VHS/VSS reference temperature T_ref in
     *        kelvin at which d_ref is defined; must be positive.
     * @param viscosity_index VHS viscosity–temperature exponent omega
     *        (dimensionless); 0.5 reproduces the hard-sphere model.
     * @param scattering_parameter VSS scattering exponent alpha
     *        (dimensionless); 1.0 gives isotropic (VHS) scattering.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Molecule(const float mass,
             const float translational_energy,
             const float rotational_energy,
             const float vibrational_energy,
             const float reference_diameter,
             const float reference_temperature,
             const float viscosity_index,
             const float scattering_parameter) noexcept
        : _mass(mass)
        , _translational_energy(translational_energy)
        , _rotational_energy(rotational_energy)
        , _vibrational_energy(vibrational_energy)
        , _reference_diameter(reference_diameter)
        , _reference_temperature(reference_temperature)
        , _viscosity_index(viscosity_index)
        , _scattering_parameter(scattering_parameter) {
    }

    /**
     * @brief Returns the species mass in kilograms.
     * @return The stored mass; host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    mass() const noexcept {
        return _mass;
    }

    /**
     * @brief Returns the translational internal energy in joules.
     * @return The stored translational energy; host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    translational_energy() const noexcept {
        return _translational_energy;
    }

    /**
     * @brief Returns the rotational internal energy in joules.
     * @return The stored rotational energy; host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    rotational_energy() const noexcept {
        return _rotational_energy;
    }

    /**
     * @brief Returns the vibrational internal energy in joules.
     * @return The stored vibrational energy; host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    vibrational_energy() const noexcept {
        return _vibrational_energy;
    }

    /**
     * @brief Returns the VHS/VSS reference diameter d_ref in metres.
     * @return The stored reference diameter; host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    reference_diameter() const noexcept {
        return _reference_diameter;
    }

    /**
     * @brief Returns the VHS/VSS reference temperature T_ref in kelvin.
     * @return The stored reference temperature; host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    reference_temperature() const noexcept {
        return _reference_temperature;
    }

    /**
     * @brief Returns the VHS viscosity–temperature exponent omega.
     * @return The stored viscosity index (dimensionless); host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    viscosity_index() const noexcept {
        return _viscosity_index;
    }

    /**
     * @brief Returns the VSS scattering exponent alpha.
     * @return The stored scattering parameter (dimensionless); host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    scattering_parameter() const noexcept {
        return _scattering_parameter;
    }

private:
    float _mass {}; ///< Species mass in kilograms.

    float _translational_energy {}; ///< Translational internal energy in joules.

    float _rotational_energy {}; ///< Rotational internal energy in joules.

    float _vibrational_energy {}; ///< Vibrational internal energy in joules.

    float _reference_diameter {}; ///< VHS/VSS reference diameter d_ref in metres.

    float _reference_temperature {}; ///< VHS/VSS reference temperature T_ref in kelvin.

    float _viscosity_index { 0.5f }; ///< VHS viscosity exponent omega; 0.5 = hard sphere.

    float _scattering_parameter { 1.0f }; ///< VSS scattering exponent alpha; 1.0 = isotropic.
};

}