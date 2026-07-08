#pragma once

#include <atlas/core/device_variant.h>
#include <atlas/core/macros.h>
#include <atlas/material/material_type.h>

#include <optional>
#include <type_traits>

namespace atlas {

// The per-type material leaves. They currently share the same fields; distinct
// types leave room for per-type physics to diverge later (e.g. atoms dropping
// rotational/vibrational energy).
class MoleculeMaterial final {
public:
    float mass {};
    std::optional<float> translational_energy;
    std::optional<float> rotational_energy;
    std::optional<float> vibrational_energy;
};

class AtomMaterial final {
public:
    float mass {};
    std::optional<float> translational_energy;
    std::optional<float> rotational_energy;
    std::optional<float> vibrational_energy;
};

class IonMaterial final {
public:
    float mass {};
    std::optional<float> translational_energy;
    std::optional<float> rotational_energy;
    std::optional<float> vibrational_energy;
};

class NeutronMaterial final {
public:
    float mass {};
    std::optional<float> translational_energy;
    std::optional<float> rotational_energy;
    std::optional<float> vibrational_energy;
};

class SolidMaterial final {
public:
    float mass {};
    std::optional<float> translational_energy;
    std::optional<float> rotational_energy;
    std::optional<float> vibrational_energy;
};

class Material final {
public:

    MaterialType type = MaterialType::molecule;

    union {

        MoleculeMaterial molecule;

        AtomMaterial atom;

        IonMaterial ion;

        NeutronMaterial neutron;

        SolidMaterial solid;
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

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::optional<float>
    translational_energy() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::optional<float>
    rotational_energy() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::optional<float>
    vibrational_energy() const noexcept;
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
    operator()(const M& material) const noexcept { return material.mass; }
};
class MaterialTranslationalEnergy {
public:
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::optional<float>
    operator()(const M& material) const noexcept { return material.translational_energy; }
};
class MaterialRotationalEnergy {
public:
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::optional<float>
    operator()(const M& material) const noexcept { return material.rotational_energy; }
};
class MaterialVibrationalEnergy {
public:
    template <typename M>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::optional<float>
    operator()(const M& material) const noexcept { return material.vibrational_energy; }
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

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::optional<float>
Material::translational_energy() const noexcept {
    return MaterialVariant::visit(*this, MaterialTranslationalEnergy {}, std::optional<float> {});
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::optional<float>
Material::rotational_energy() const noexcept {
    return MaterialVariant::visit(*this, MaterialRotationalEnergy {}, std::optional<float> {});
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::optional<float>
Material::vibrational_energy() const noexcept {
    return MaterialVariant::visit(*this, MaterialVibrationalEnergy {}, std::optional<float> {});
}

}
