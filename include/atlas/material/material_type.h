#pragma once

namespace atlas {

/**
 * @brief Discriminant tag selecting which leaf a `Material` tagged union holds.
 *
 * One enumerator exists per `Material` leaf type. The value doubles as the
 * active-member index for the `DeviceVariant` machinery: `MaterialVariant` maps
 * each enumerator to the matching union member via a `DeviceVariantCase`, and
 * `Material::type` stores the currently active tag. The underlying type is
 * fixed to `int` so the tag is trivially copyable and stable across the
 * host/device boundary.
 *
 * @note `molecule` is the default tag: a default-constructed `Material`
 *       activates the `Molecule` leaf, and `DeviceVariant::normalize` folds any
 *       out-of-range tag back to it.
 */
enum class MaterialType : int {

    molecule, ///< Polyatomic species carrying translational/rotational/vibrational energy.

    atom, ///< Monatomic neutral species.

    ion, ///< Charged species (charge itself is not modelled here).

    neutron, ///< Neutral nuclear particle.

    solid ///< Immobile boundary/wall species; all non-mass properties are stubbed to 1.0f.
};

}