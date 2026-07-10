#pragma once

#include <atlas/core/macros.h>

namespace atlas {

/**
 * @brief Immobile boundary/wall leaf of the `Material` tagged union.
 *
 * `Solid` represents a static material such as a wall: it participates in the
 * species table but never flows, collides as a gas particle, or carries
 * internal energy. It therefore stores only its `mass` and stubs every other
 * `ConceptMaterial` property to a constant `1.0f`.
 *
 * The `1.0f` stub is a *deliberate* simplification, not an oversight. The
 * accessors must stay `noexcept` and `__host__ __device__` so `Solid` satisfies
 * `ConceptMaterial` and copies into device memory like the other leaves;
 * throwing or returning a sentinel is not an option on the device. `1.0f` is
 * chosen over `0.0f` specifically because the VHS/VSS collision math divides by
 * and takes powers of these values — a nonzero, finite placeholder keeps any
 * accidental use numerically well-defined rather than producing NaN/Inf. A
 * `Solid` is not expected to be selected as a DSMC collision partner, so these
 * values are never meant to feed real physics.
 *
 * @see ConceptMaterial, Material, MaterialType::solid, Molecule
 */
class Solid final {
public:
    /**
     * @brief Constructs a solid with zero mass.
     *
     * Needed so `Solid` can be default-constructed inside a `Material` union
     * and stored in device buffers before being populated.
     */
    Solid() = default;

    /**
     * @brief Constructs a solid from its mass.
     *
     * `explicit` to prevent an accidental `float`→`Solid` conversion. Host- and
     * device-callable and `noexcept`; only copies a float.
     *
     * @param mass Species mass in kilograms.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Solid(const float mass) noexcept
        : _mass(mass) {
    }

    /**
     * @brief Returns the species mass in kilograms.
     * @return The stored mass; host and device callable. This is the only
     *         property a `Solid` actually carries.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    mass() const noexcept {
        return _mass;
    }

    /**
     * @brief Stub translational energy for a solid.
     * @return Always 1.0f — a solid carries no internal energy; see the class
     *         note on why the stub is 1.0f rather than 0.0f.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    translational_energy() const noexcept {
        return 1.0f;
    }

    /**
     * @brief Stub rotational energy for a solid.
     * @return Always 1.0f; see the class note.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    rotational_energy() const noexcept {
        return 1.0f;
    }

    /**
     * @brief Stub vibrational energy for a solid.
     * @return Always 1.0f; see the class note.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    vibrational_energy() const noexcept {
        return 1.0f;
    }

    /**
     * @brief Stub VHS/VSS reference diameter for a solid.
     * @return Always 1.0f; a solid is not a real collision partner. See the class note.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    reference_diameter() const noexcept {
        return 1.0f;
    }

    /**
     * @brief Stub VHS/VSS reference temperature for a solid.
     * @return Always 1.0f; see the class note.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    reference_temperature() const noexcept {
        return 1.0f;
    }

    /**
     * @brief Stub VHS viscosity index for a solid.
     * @return Always 1.0f; see the class note.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    viscosity_index() const noexcept {
        return 1.0f;
    }

    /**
     * @brief Stub VSS scattering parameter for a solid.
     * @return Always 1.0f; see the class note.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    scattering_parameter() const noexcept {
        return 1.0f;
    }

private:
    float _mass {}; ///< Species mass in kilograms; the only real property of a Solid.
};

}