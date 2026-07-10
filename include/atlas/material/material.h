#pragma once

#include <atlas/core/device_variant.h>
#include <atlas/core/macros.h>
#include <atlas/material/atom.h>
#include <atlas/material/ion.h>
#include <atlas/material/material_type.h>
#include <atlas/material/molecule.h>
#include <atlas/material/neutron.h>
#include <atlas/material/solid.h>

#include <concepts>
#include <type_traits>

namespace atlas {

/**
 * @brief Constrains a type to the full material-property interface.
 *
 * A model must expose all eight scalar getters, each returning exactly `float`,
 * for `Material`'s visitor functors to dispatch to it uniformly. The static
 * asserts below enforce that every leaf conforms; extending `Material` with a
 * new property means adding a getter here and to every leaf.
 *
 * @tparam M The candidate leaf type (checked against a `const M`).
 */
template <typename M>
concept ConceptMaterial = requires(const M material) {
    { material.mass() } -> std::same_as<float>;
    { material.translational_energy() } -> std::same_as<float>;
    { material.rotational_energy() } -> std::same_as<float>;
    { material.vibrational_energy() } -> std::same_as<float>;
    { material.reference_diameter() } -> std::same_as<float>;
    { material.reference_temperature() } -> std::same_as<float>;
    { material.viscosity_index() } -> std::same_as<float>;
    { material.scattering_parameter() } -> std::same_as<float>;
};

static_assert(ConceptMaterial<Molecule>); ///< Molecule leaf satisfies the material interface.
static_assert(ConceptMaterial<Atom>);     ///< Atom leaf satisfies the material interface.
static_assert(ConceptMaterial<Ion>);      ///< Ion leaf satisfies the material interface.
static_assert(ConceptMaterial<Neutron>);  ///< Neutron leaf satisfies the material interface.
static_assert(ConceptMaterial<Solid>);    ///< Solid leaf satisfies the material interface.

/**
 * @brief Tagged-union umbrella over one material leaf per `MaterialType`.
 *
 * `Material` is the single value stored per species in `MaterialDictionary`'s
 * device table. It holds a `type` discriminant plus a `union` of the five
 * leaves and dispatches every property getter to the active leaf through
 * `MaterialVariant` (a `DeviceVariant`). Because every leaf is trivially
 * copyable, `Material` itself is trivially copyable, lives in a
 * `DeviceBuffer<Material>`, and can be captured by value inside a device lambda.
 *
 * The special members are `= default` and the class is trivially copyable *only
 * because* every leaf is; do not add a leaf that owns a resource. All the real
 * variant plumbing (placement-new by tag, copy/assign of the active member) is
 * done through `MaterialVariant`, invoked from the out-of-line constructors and
 * accessors defined at the bottom of this header.
 *
 * @see MaterialVariant, ConceptMaterial, MaterialDictionary
 */
class Material final {
public:
    MaterialType type = MaterialType::molecule; ///< Active-leaf discriminant; defaults to molecule.

    /**
     * @brief Storage for exactly one leaf; the active member is selected by `type`.
     *
     * Reading an inactive member is undefined; always go through the property
     * accessors, which honour `type` via `MaterialVariant::visit`.
     */
    union {

        Molecule molecule; ///< Active when `type == MaterialType::molecule`.

        Atom atom; ///< Active when `type == MaterialType::atom`.

        Ion ion; ///< Active when `type == MaterialType::ion`.

        Neutron neutron; ///< Active when `type == MaterialType::neutron`.

        Solid solid; ///< Active when `type == MaterialType::solid`.
    };

    /**
     * @brief Default-constructs a molecule leaf.
     *
     * Delegates to `MaterialVariant::construct` with the default tag so the
     * union has a well-defined active member. Host- and device-callable.
     */
    ATLAS_ALL_DEVICE
    Material() noexcept;

    /**
     * @brief Trivial copy constructor.
     *
     * Valid as a byte copy precisely because every leaf is trivially copyable;
     * the active `type` and its union bytes are copied verbatim.
     */
    ATLAS_ALL_DEVICE
    Material(const Material& other) noexcept = default;

    /**
     * @brief Trivial copy assignment (see the copy constructor).
     * @return Reference to this material.
     */
    ATLAS_ALL_DEVICE Material&
    operator=(const Material& other) noexcept = default;

    /**
     * @brief Trivial destructor; no leaf owns a resource, so nothing to release.
     */
    ATLAS_ALL_DEVICE
    ~Material() noexcept = default;

    /**
     * @brief Constructs a `Material` directly from one of its leaf values.
     *
     * Deduces the matching `MaterialType` from `Payload` and placement-news the
     * leaf into the union via `MaterialVariant::construct_payload`, e.g.
     * `Material m(Molecule(...))` or `Material m(Solid(mass))`.
     *
     * @tparam Payload The leaf type; the SFINAE guard excludes `Material` itself
     *         so this never shadows the copy constructor. Passing a type that is
     *         not a leaf triggers a `static_assert` inside `DeviceVariant`.
     * @param payload The leaf value to copy into the active union member.
     */
    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Material>, int> = 0>
    ATLAS_ALL_DEVICE explicit Material(const Payload& payload) noexcept;

    /**
     * @brief Returns the active leaf's mass in kilograms.
     * @return The species mass, or 0.0f if `type` matches no case (cannot
     *         happen for a normalized tag). Host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    mass() const noexcept;

    /**
     * @brief Returns the active leaf's translational internal energy in joules.
     * @return The translational energy, or 0.0f fallback. Host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    translational_energy() const noexcept;

    /**
     * @brief Returns the active leaf's rotational internal energy in joules.
     * @return The rotational energy, or 0.0f fallback. Host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    rotational_energy() const noexcept;

    /**
     * @brief Returns the active leaf's vibrational internal energy in joules.
     * @return The vibrational energy, or 0.0f fallback. Host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    vibrational_energy() const noexcept;

    /**
     * @brief Returns the active leaf's VHS/VSS reference diameter d_ref in metres.
     * @return The reference diameter, or 0.0f fallback. Host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    reference_diameter() const noexcept;

    /**
     * @brief Returns the active leaf's VHS/VSS reference temperature T_ref in kelvin.
     * @return The reference temperature, or 0.0f fallback. Host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    reference_temperature() const noexcept;

    /**
     * @brief Returns the active leaf's VHS viscosity–temperature exponent omega.
     * @return The viscosity index (dimensionless), or 0.0f fallback. Host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    viscosity_index() const noexcept;

    /**
     * @brief Returns the active leaf's VSS scattering exponent alpha.
     * @return The scattering parameter (dimensionless), or 0.0f fallback. Host and device callable.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    scattering_parameter() const noexcept;
};

/**
 * @brief `DeviceVariant` specialization that maps each `MaterialType` to its union member.
 *
 * Supplies the placement-new/visit machinery for `Material`: `molecule` is the
 * default tag, and each `DeviceVariantCase` binds a tag value to the member
 * pointer of the corresponding leaf. All of `Material`'s constructors and
 * accessors funnel through this alias.
 */
using MaterialVariant = DeviceVariant<
    Material,
    MaterialType,
    MaterialType::molecule,
    DeviceVariantCase<MaterialType::molecule, &Material::molecule>,
    DeviceVariantCase<MaterialType::atom, &Material::atom>,
    DeviceVariantCase<MaterialType::ion, &Material::ion>,
    DeviceVariantCase<MaterialType::neutron, &Material::neutron>,
    DeviceVariantCase<MaterialType::solid, &Material::solid>>;

/**
 * @brief Visitor functor returning `material.mass()` for any leaf type.
 *
 * Passed to `MaterialVariant::visit`, which calls it on whichever leaf is
 * active. A plain functor (not a lambda) so it is usable in device code without
 * hitting the nvcc restriction on extended lambdas in class scope.
 */
class MaterialMass {
public:
    /**
     * @brief Extracts the mass from any material leaf.
     * @tparam M The concrete leaf type deduced by `MaterialVariant::visit`.
     * @param material The active leaf.
     * @return `material.mass()` in kilograms.
     */
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const M& material) const noexcept { return material.mass(); }
};

/**
 * @brief Visitor functor returning `material.translational_energy()` for any leaf.
 * @see MaterialMass
 */
class MaterialTranslationalEnergy {
public:
    /**
     * @brief Extracts the translational internal energy from any material leaf.
     * @tparam M The concrete leaf type deduced by `MaterialVariant::visit`.
     * @param material The active leaf.
     * @return `material.translational_energy()` in joules.
     */
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const M& material) const noexcept { return material.translational_energy(); }
};

/**
 * @brief Visitor functor returning `material.rotational_energy()` for any leaf.
 * @see MaterialMass
 */
class MaterialRotationalEnergy {
public:
    /**
     * @brief Extracts the rotational internal energy from any material leaf.
     * @tparam M The concrete leaf type deduced by `MaterialVariant::visit`.
     * @param material The active leaf.
     * @return `material.rotational_energy()` in joules.
     */
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const M& material) const noexcept { return material.rotational_energy(); }
};

/**
 * @brief Visitor functor returning `material.vibrational_energy()` for any leaf.
 * @see MaterialMass
 */
class MaterialVibrationalEnergy {
public:
    /**
     * @brief Extracts the vibrational internal energy from any material leaf.
     * @tparam M The concrete leaf type deduced by `MaterialVariant::visit`.
     * @param material The active leaf.
     * @return `material.vibrational_energy()` in joules.
     */
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const M& material) const noexcept { return material.vibrational_energy(); }
};

/**
 * @brief Visitor functor returning `material.reference_diameter()` for any leaf.
 * @see MaterialMass
 */
class MaterialReferenceDiameter {
public:
    /**
     * @brief Extracts the VHS/VSS reference diameter from any material leaf.
     * @tparam M The concrete leaf type deduced by `MaterialVariant::visit`.
     * @param material The active leaf.
     * @return `material.reference_diameter()` in metres.
     */
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const M& material) const noexcept { return material.reference_diameter(); }
};

/**
 * @brief Visitor functor returning `material.reference_temperature()` for any leaf.
 * @see MaterialMass
 */
class MaterialReferenceTemperature {
public:
    /**
     * @brief Extracts the VHS/VSS reference temperature from any material leaf.
     * @tparam M The concrete leaf type deduced by `MaterialVariant::visit`.
     * @param material The active leaf.
     * @return `material.reference_temperature()` in kelvin.
     */
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const M& material) const noexcept { return material.reference_temperature(); }
};

/**
 * @brief Visitor functor returning `material.viscosity_index()` for any leaf.
 * @see MaterialMass
 */
class MaterialViscosityIndex {
public:
    /**
     * @brief Extracts the VHS viscosity exponent from any material leaf.
     * @tparam M The concrete leaf type deduced by `MaterialVariant::visit`.
     * @param material The active leaf.
     * @return `material.viscosity_index()` (dimensionless).
     */
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const M& material) const noexcept { return material.viscosity_index(); }
};

/**
 * @brief Visitor functor returning `material.scattering_parameter()` for any leaf.
 * @see MaterialMass
 */
class MaterialScatteringParameter {
public:
    /**
     * @brief Extracts the VSS scattering exponent from any material leaf.
     * @tparam M The concrete leaf type deduced by `MaterialVariant::visit`.
     * @param material The active leaf.
     * @return `material.scattering_parameter()` (dimensionless).
     */
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const M& material) const noexcept { return material.scattering_parameter(); }
};

// Out-of-line member definitions: they must follow MaterialVariant and the
// visitor functors, since each one instantiates the variant machinery.

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Material::Material() noexcept {
    // Activate the default (molecule) leaf so the union always has a live member.
    MaterialVariant::construct(*this, MaterialType::molecule);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Material>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Material::Material(const Payload& payload) noexcept {
    // Deduce the tag from Payload's type and placement-new the leaf in place.
    MaterialVariant::construct_payload(*this, payload);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Material::mass() const noexcept {
    // Dispatch on `type`; 0.0f is the fallback for an unmatched tag.
    return MaterialVariant::visit(*this, MaterialMass {}, 0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Material::translational_energy() const noexcept {
    return MaterialVariant::visit(*this, MaterialTranslationalEnergy {}, 0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Material::rotational_energy() const noexcept {
    return MaterialVariant::visit(*this, MaterialRotationalEnergy {}, 0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Material::vibrational_energy() const noexcept {
    return MaterialVariant::visit(*this, MaterialVibrationalEnergy {}, 0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Material::reference_diameter() const noexcept {
    return MaterialVariant::visit(*this, MaterialReferenceDiameter {}, 0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Material::reference_temperature() const noexcept {
    return MaterialVariant::visit(*this, MaterialReferenceTemperature {}, 0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Material::viscosity_index() const noexcept {
    return MaterialVariant::visit(*this, MaterialViscosityIndex {}, 0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Material::scattering_parameter() const noexcept {
    return MaterialVariant::visit(*this, MaterialScatteringParameter {}, 0.0f);
}

}