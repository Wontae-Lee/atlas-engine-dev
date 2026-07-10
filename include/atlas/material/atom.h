#pragma once

#include <atlas/core/macros.h>

namespace atlas {

/**
 * @brief Monatomic-species leaf of the `Material` tagged union.
 *
 * `Atom` has the same layout and contract as `Molecule` (see `molecule.h`): a
 * trivially copyable bundle of eight `float`s that satisfies `ConceptMaterial`
 * and is safe to copy into device memory. It is kept as a distinct type so the
 * per-species physics can diverge later; a monatomic species carries no
 * rotational or vibrational modes physically, but the fields are still present
 * (typically left zero) to keep every leaf uniform.
 *
 * The final four fields are the Variable Hard Sphere / Variable Soft Sphere
 * (VHS/VSS) collision parameters consumed by the DSMC kernels.
 *
 * @see ConceptMaterial, Material, MaterialType::atom, Molecule
 */
class Atom final {
public:
    /**
     * @brief Constructs an atom with every property zero-initialized.
     *
     * `_viscosity_index` and `_scattering_parameter` keep their in-class
     * defaults (0.5f and 1.0f). Needed so `Atom` can live in device buffers
     * before being populated.
     */
    Atom() = default;

    /**
     * @brief Constructs an atom from its full set of physical constants.
     *
     * Host- and device-callable and `noexcept`; only copies floats.
     *
     * @param mass Species mass in kilograms.
     * @param translational_energy Translational internal energy in joules.
     * @param rotational_energy Rotational internal energy in joules (nominally 0 for an atom).
     * @param vibrational_energy Vibrational internal energy in joules (nominally 0 for an atom).
     * @param reference_diameter VHS/VSS reference diameter d_ref in metres; must be positive.
     * @param reference_temperature VHS/VSS reference temperature T_ref in kelvin; must be positive.
     * @param viscosity_index VHS viscosity exponent omega (dimensionless); 0.5 = hard sphere.
     * @param scattering_parameter VSS scattering exponent alpha (dimensionless); 1.0 = isotropic.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Atom(const float mass,
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