#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sampling/sampling.h>

namespace atlas {

enum class DiffuseSampling {

    cosine_weighted,

    uniform
};

class IsothermalSurfaceInteraction final {
public:
    class Builder;

public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    IsothermalSurfaceInteraction() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~IsothermalSurfaceInteraction() noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    set_diffuse_sampling(DiffuseSampling mode) noexcept;

    ATLAS_HOST void
    set_restitution(float restitution_coeff) noexcept;

    ATLAS_HOST void
    set_momentum_acc(float momentum_acc) noexcept;

    ATLAS_HOST void
    set_temperature(float temperature) noexcept;

    ATLAS_NODISCARD ATLAS_HOST DiffuseSampling
    diffuse_sampling() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    restitution() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    momentum_acc() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    temperature() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const Float3& incident, const Float3& normal) const noexcept {
        const float incident_speed = incident.length();

        if (incident_speed <= atlas::tol) {
            return Float3(0.0f, 0.0f, 0.0f);
        }

        const Float3 specular_dir  = atlas::reflected(incident, normal);
        const Float3 specular_unit = specular_dir.normalized();

        if (_momentum_acc <= 0.0f) {
            return specular_unit * (incident_speed * _restitution_coeff);
        }

        const float u1 = atlas::sample_hashed_unit_interval(
            incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U1);

        const float u2 = atlas::sample_hashed_unit_interval(
            normal + incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U2);

        Float3 diffuse_dir {};

        if (_diffuse_sampling == DiffuseSampling::cosine_weighted) {
            diffuse_dir = atlas::sample_cosine_hemisphere(normal, u1, u2);
        } else {
            diffuse_dir = atlas::sample_uniform_hemisphere(normal, u1, u2);
        }

        const float mix = atlas::sample_hashed_unit_interval(
            incident + normal * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX,
            atlas::RANDOM_HASH_SALT_MIX);

        const Float3 out_unit = (mix < _momentum_acc)
            ? diffuse_dir.normalized()
            : specular_unit;

        return out_unit * (incident_speed * _restitution_coeff);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
    internal_energy(const FluidInternalEnergy& incident_energy,
                    const Float3& incident_velocity,
                    const Float3& normal,
                    const MaterialProperties& material) const noexcept {
        static_cast<void>(incident_velocity);
        static_cast<void>(normal);
        static_cast<void>(material);
        return incident_energy;
    }

private:
    float _restitution_coeff { 1.0f };

    float _momentum_acc { 1.0f };

    float _temperature { 273.15f };

    DiffuseSampling _diffuse_sampling { DiffuseSampling::uniform };
};

class IsothermalSurfaceInteraction::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_diffuse_sampling(DiffuseSampling mode) noexcept;

    ATLAS_HOST Builder&
    with_restitution(float restitution) noexcept;

    ATLAS_HOST Builder&
    with_momentum_acc(float momentum_acc) noexcept;

    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

    ATLAS_NODISCARD ATLAS_HOST IsothermalSurfaceInteraction
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<IsothermalSurfaceInteraction>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    DiffuseSampling _diffuse_sampling { DiffuseSampling::uniform };

    float _restitution { 1.0f };

    float _momentum_acc { 1.0f };

    float _temperature { 273.15f };
};

using IsothermalSurfaceInteractionHostPtr = atlas::host_shared_ptr<IsothermalSurfaceInteraction>;

using IsothermalSurfaceInteractionDevicePtr = atlas::device_shared_ptr<IsothermalSurfaceInteraction>;

}
