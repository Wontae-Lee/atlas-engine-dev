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
#include <stdexcept>
#include <type_traits>

namespace atlas {

template <typename M>
concept ConceptMaterial = requires(const M material) {
    { material.mass() } -> std::same_as<float>;
    { material.translational_energy() } -> std::same_as<float>;
    { material.rotational_energy() } -> std::same_as<float>;
    { material.vibrational_energy() } -> std::same_as<float>;
};

static_assert(ConceptMaterial<Molecule>);
static_assert(ConceptMaterial<Atom>);
static_assert(ConceptMaterial<Ion>);
static_assert(ConceptMaterial<Neutron>);
static_assert(ConceptMaterial<Solid>);

class Material final {
public:

    MaterialType type = MaterialType::molecule;

    union {

        Molecule molecule;

        Atom atom;

        Ion ion;

        Neutron neutron;

        Solid solid;
    };

    ATLAS_ALL_DEVICE
    Material() noexcept;

    ATLAS_ALL_DEVICE
    Material(const Material& other) noexcept = default;

    ATLAS_ALL_DEVICE Material&
    operator=(const Material& other) noexcept = default;

    ATLAS_ALL_DEVICE
    ~Material() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Material>, int> = 0>
    ATLAS_ALL_DEVICE explicit Material(const Payload& payload) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    mass() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE float
    translational_energy() const;

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE float
    rotational_energy() const;

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE float
    vibrational_energy() const;
};

using MaterialVariant = DeviceVariant<
    Material,
    MaterialType,
    MaterialType::molecule,
    DeviceVariantCase<MaterialType::molecule, &Material::molecule>,
    DeviceVariantCase<MaterialType::atom, &Material::atom>,
    DeviceVariantCase<MaterialType::ion, &Material::ion>,
    DeviceVariantCase<MaterialType::neutron, &Material::neutron>,
    DeviceVariantCase<MaterialType::solid, &Material::solid>>;

class MaterialMass {
public:
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const M& material) const noexcept { return material.mass(); }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Material::Material() noexcept {
    MaterialVariant::construct(*this, MaterialType::molecule);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Material>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Material::Material(const Payload& payload) noexcept {
    MaterialVariant::construct_payload(*this, payload);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
Material::mass() const noexcept {
    return MaterialVariant::visit(*this, MaterialMass {}, 0.0f);
}

ATLAS_HOST ATLAS_FORCE_INLINE float
Material::translational_energy() const {
    switch (type) {
        case MaterialType::molecule: return molecule.translational_energy();
        case MaterialType::atom: return atom.translational_energy();
        case MaterialType::ion: return ion.translational_energy();
        case MaterialType::neutron: return neutron.translational_energy();
        case MaterialType::solid: return solid.translational_energy();
    }

    throw std::runtime_error("Material holds an unknown material type.");
}

ATLAS_HOST ATLAS_FORCE_INLINE float
Material::rotational_energy() const {
    switch (type) {
        case MaterialType::molecule: return molecule.rotational_energy();
        case MaterialType::atom: return atom.rotational_energy();
        case MaterialType::ion: return ion.rotational_energy();
        case MaterialType::neutron: return neutron.rotational_energy();
        case MaterialType::solid: return solid.rotational_energy();
    }

    throw std::runtime_error("Material holds an unknown material type.");
}

ATLAS_HOST ATLAS_FORCE_INLINE float
Material::vibrational_energy() const {
    switch (type) {
        case MaterialType::molecule: return molecule.vibrational_energy();
        case MaterialType::atom: return atom.vibrational_energy();
        case MaterialType::ion: return ion.vibrational_energy();
        case MaterialType::neutron: return neutron.vibrational_energy();
        case MaterialType::solid: return solid.vibrational_energy();
    }

    throw std::runtime_error("Material holds an unknown material type.");
}

}
