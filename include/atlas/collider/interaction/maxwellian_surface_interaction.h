#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sampling/sampling.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace atlas {

enum struct MaxwellianInternalEnergyStyle : int {

    none,

    smooth,

    discrete
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
maxwellian_unit_sample(const float u) noexcept {

    return u > atlas::eps ? u : atlas::eps;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
maxwellian_tangent_seed(const Float3& normal, const Float3& seed) noexcept {
    return atlas::orthogonal_unit_vector(normal, seed, atlas::tol);
}

class MaxwellianSurfaceInteraction final {
public:
    class Builder;

public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MaxwellianSurfaceInteraction() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~MaxwellianSurfaceInteraction() noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    set_temperature(float temperature) noexcept;

    ATLAS_HOST void
    set_molecular_mass(float molecular_mass) noexcept;

    ATLAS_HOST void
    set_momentum_acc(float momentum_acc) noexcept;

    ATLAS_HOST void
    set_trans_acc(float trans_acc) noexcept;

    ATLAS_HOST void
    set_rot_acc(float rot_acc) noexcept;

    ATLAS_HOST void
    set_vib_acc(float vib_acc) noexcept;

    ATLAS_HOST void
    set_rot_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST void
    set_vib_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST void
    set_accommodation(float momentum_acc, float trans_acc, float rot_acc, float vib_acc) noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    temperature() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    molecular_mass() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    momentum_acc() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    trans_acc() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    rot_acc() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    vib_acc() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST MaxwellianInternalEnergyStyle
    rot_style() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST MaxwellianInternalEnergyStyle
    vib_style() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    most_probable_speed() const noexcept {
        return atlas::sqrt_nonnegative(
            2.0f * atlas::boltzmann_constant * _temperature / _molecular_mass);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const Float3& incident, const Float3& normal) const noexcept {
        const float branch_sample = atlas::sample_hashed_unit_interval(
            incident + normal * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX,
            atlas::RANDOM_HASH_SALT_MIX);

        const float perpendicular_sample = atlas::sample_hashed_unit_interval(
            incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U1);

        const float theta_sample = atlas::sample_hashed_unit_interval(
            normal + incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U2);

        const float tangent_sample = atlas::sample_hashed_unit_interval(
            incident + normal * 3.0f,
            4.19f);

        const Float3 tangent_seed(
            atlas::sample_hashed_unit_interval(incident, 5.11f),
            atlas::sample_hashed_unit_interval(normal, 6.23f),
            atlas::sample_hashed_unit_interval(incident + normal, 7.37f));

        return sample(
            incident,
            normal,
            branch_sample,
            perpendicular_sample,
            theta_sample,
            tangent_sample,
            tangent_seed);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    sample(const Float3& incident,
           const Float3& normal,
           const float branch_sample,
           const float perpendicular_sample,
           const float theta_sample,
           const float tangent_sample,
           const Float3& tangent_seed) const noexcept {
        if (branch_sample > _momentum_acc) {
            return atlas::reflected(incident, normal);
        }

        const float vrm      = most_probable_speed();
        const float vperp    = vrm * atlas::sqrt_nonnegative(-std::log(maxwellian_unit_sample(perpendicular_sample)));
        const float theta    = 2.0f * atlas::pi * theta_sample;
        const float vtangent = vrm * atlas::sqrt_nonnegative(-std::log(maxwellian_unit_sample(tangent_sample)));
        const float vtan1    = vtangent * std::sin(theta);
        const float vtan2    = vtangent * std::cos(theta);

        Float3 tangent1 = atlas::reject(incident, normal);

        if (tangent1.length_squared() == 0.0f) {
            tangent1 = maxwellian_tangent_seed(normal, tangent_seed);
        } else {
            tangent1 = tangent1.normalized();
        }

        const Float3 tangent2 = atlas::cross(normal, tangent1);

        return normal * vperp + tangent1 * vtan1 + tangent2 * vtan2;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
    internal_energy(const FluidInternalEnergy& incident) const noexcept {
        const Float3 seed(incident.translational, incident.rotational, incident.vibrational);

        return sample_internal_energy(
            incident,
            atlas::sample_hashed_unit_interval(seed, 8.11f),
            atlas::sample_hashed_unit_interval(seed, 9.23f),
            atlas::sample_hashed_unit_interval(seed, 10.37f),
            atlas::sample_hashed_unit_interval(seed, 11.41f),
            atlas::sample_hashed_unit_interval(seed, 12.53f),
            atlas::sample_hashed_unit_interval(seed, 13.67f));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
    internal_energy(const FluidInternalEnergy& incident_energy,
                    const Float3& incident_velocity,
                    const Float3& normal) const noexcept {
        const float branch_sample = atlas::sample_hashed_unit_interval(
            incident_velocity + normal * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX,
            atlas::RANDOM_HASH_SALT_MIX);

        return branch_sample > _momentum_acc ? incident_energy : internal_energy(incident_energy);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
    internal_energy(const FluidInternalEnergy& incident_energy,
                    const Float3& incident_velocity,
                    const Float3& normal,
                    const MaterialProperties& material) const noexcept {
        const float branch_sample = atlas::sample_hashed_unit_interval(
            incident_velocity + normal * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX,
            atlas::RANDOM_HASH_SALT_MIX);

        if (branch_sample > _momentum_acc) {
            return incident_energy;
        }

        const Float3 seed(
            incident_energy.translational + incident_velocity.x,
            incident_energy.rotational + incident_velocity.y,
            incident_energy.vibrational + incident_velocity.z);

        return FluidInternalEnergy {
            incident_energy.translational,
            sample_diffuse_rotational_energy(material, seed),
            sample_diffuse_vibrational_energy(material, seed)
        };
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
    sample_internal_energy(const FluidInternalEnergy& incident,
                           const float trans_sample,
                           const float trans_theta_sample,
                           const float rot_sample,
                           const float rot_theta_sample,
                           const float vib_sample,
                           const float vib_theta_sample) const noexcept {
        return FluidInternalEnergy {
            sample_internal_energy_mode(incident.translational, _trans_acc, trans_sample, trans_theta_sample),
            sample_internal_energy_mode(incident.rotational, _rot_acc, rot_sample, rot_theta_sample),
            sample_internal_energy_mode(incident.vibrational, _vib_acc, vib_sample, vib_theta_sample)
        };
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    sample_internal_energy_mode(const float incident,
                                const float acc,
                                const float sample,
                                const float theta_sample) const noexcept {
        const float wall_energy = atlas::boltzmann_constant * _temperature;
        const float safe_wall   = wall_energy > 0.0f ? wall_energy : std::numeric_limits<float>::min();
        const float safe_energy = std::max(incident, 0.0f);
        const float magnitude   = atlas::sqrt_nonnegative(safe_energy * (1.0f - acc) / safe_wall);
        const float radius      = atlas::sqrt_nonnegative(-acc * std::log(maxwellian_unit_sample(sample)));
        const float phase       = std::cos(2.0f * atlas::pi * theta_sample);

        return safe_wall * (radius * radius + magnitude * magnitude + 2.0f * radius * magnitude * phase);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    sample_diffuse_rotational_energy(const MaterialProperties& material,
                                     const Float3& seed) const noexcept {
        const int dof = material.rotational_dof.has_value() ? *material.rotational_dof : 0;
        if (_rot_style == MaxwellianInternalEnergyStyle::none || dof < 2) {
            return 0.0f;
        }

        if (_rot_style == MaxwellianInternalEnergyStyle::discrete && dof == 2) {
            const float rot_temperature = material.rotational_temperature.has_value()
                ? *material.rotational_temperature
                : _temperature;
            const float quantum         = rot_temperature > atlas::eps ? rot_temperature : atlas::eps;
            const float sample          = maxwellian_unit_sample(
                atlas::sample_hashed_unit_interval(seed, 14.11f));
            const int level = static_cast<int>(-std::log(sample) * _temperature / quantum);
            return static_cast<float>(level) * atlas::boltzmann_constant * quantum;
        }

        return sample_diffuse_smooth_energy(dof, seed, 15.23f);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    sample_diffuse_vibrational_energy(const MaterialProperties& material,
                                      const Float3& seed) const noexcept {
        const int dof = material.vibrational_dof.has_value() ? *material.vibrational_dof : 0;
        if (_vib_style == MaxwellianInternalEnergyStyle::none || dof < 2) {
            return 0.0f;
        }

        if (_vib_style == MaxwellianInternalEnergyStyle::discrete && dof == 2) {
            const float vib_temperature = material.characteristic_vibrational_temperature.has_value()
                ? *material.characteristic_vibrational_temperature
                : _temperature;
            const float quantum         = vib_temperature > atlas::eps ? vib_temperature : atlas::eps;
            const float sample          = maxwellian_unit_sample(
                atlas::sample_hashed_unit_interval(seed, 16.37f));
            const int level = static_cast<int>(-std::log(sample) * _temperature / quantum);
            return static_cast<float>(level) * atlas::boltzmann_constant * quantum;
        }

        return sample_diffuse_smooth_energy(dof, seed, 17.41f);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    sample_diffuse_smooth_energy(const int dof,
                                 const Float3& seed,
                                 const float salt) const noexcept {
        if (dof == 2) {
            const float sample = maxwellian_unit_sample(
                atlas::sample_hashed_unit_interval(seed, salt));
            return -std::log(sample) * atlas::boltzmann_constant * _temperature;
        }

        const float a = 0.5f * static_cast<float>(dof) - 1.0f;
        if (!(a > 0.0f)) {
            return 0.0f;
        }

        for (int i = 0;; ++i) {
            const float energy_sample = atlas::sample_hashed_unit_interval(seed, salt + static_cast<float>(i) * 0.37f);
            const float accept_sample = atlas::sample_hashed_unit_interval(seed, salt + static_cast<float>(i) * 0.37f + 0.19f);
            const float erm           = 10.0f * energy_sample;
            const float b             = std::pow(erm / a, a) * std::exp(a - erm);
            if (b > accept_sample) {
                return erm * atlas::boltzmann_constant * _temperature;
            }
        }

        return 0.0f;
    }

private:
    float _temperature { 273.15f };
    float _molecular_mass { 1.0f };
    float _momentum_acc { 1.0f };
    float _trans_acc { 1.0f };
    float _rot_acc { 1.0f };
    float _vib_acc { 1.0f };
    MaxwellianInternalEnergyStyle _rot_style { MaxwellianInternalEnergyStyle::smooth };
    MaxwellianInternalEnergyStyle _vib_style { MaxwellianInternalEnergyStyle::smooth };
};

class MaxwellianSurfaceInteraction::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

    ATLAS_HOST Builder&
    with_molecular_mass(float molecular_mass) noexcept;

    ATLAS_HOST Builder&
    with_momentum_acc(float momentum_acc) noexcept;

    ATLAS_HOST Builder&
    with_trans_acc(float trans_acc) noexcept;

    ATLAS_HOST Builder&
    with_rot_acc(float rot_acc) noexcept;

    ATLAS_HOST Builder&
    with_vib_acc(float vib_acc) noexcept;

    ATLAS_HOST Builder&
    with_rot_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST Builder&
    with_vib_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST Builder&
    with_accommodation(float momentum_acc, float trans_acc, float rot_acc, float vib_acc) noexcept;

    ATLAS_NODISCARD ATLAS_HOST MaxwellianSurfaceInteraction
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<MaxwellianSurfaceInteraction>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    float _temperature { 273.15f };
    float _molecular_mass { 1.0f };
    float _momentum_acc { 1.0f };
    float _trans_acc { 1.0f };
    float _rot_acc { 1.0f };
    float _vib_acc { 1.0f };
    MaxwellianInternalEnergyStyle _rot_style { MaxwellianInternalEnergyStyle::smooth };
    MaxwellianInternalEnergyStyle _vib_style { MaxwellianInternalEnergyStyle::smooth };
};

using MaxwellianSurfaceInteractionHostPtr = atlas::host_shared_ptr<MaxwellianSurfaceInteraction>;

using MaxwellianSurfaceInteractionDevicePtr = atlas::device_shared_ptr<MaxwellianSurfaceInteraction>;

}
