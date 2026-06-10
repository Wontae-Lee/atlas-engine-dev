#pragma once

#include <atlas/logging/logging.h>
#include <atlas/math/constants.h>
#include <atlas/sampling/sampling.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace atlas::system {

namespace detail {

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
maxwellian_unit_sample(const T u) noexcept {
    return std::max(u, static_cast<T>(atlas::eps));
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
maxwellian_tangent_seed(const Vector3<T>& normal, const Vector3<T>& seed) noexcept {
    Vector3<T> tangent = atlas::math::cross(normal, seed);

    if (tangent.length_squared() <= T(atlas::tol)) {
        const Vector3<T> axis = (std::abs(normal.x) < T(0.9))
            ? Vector3<T>(T(1), T(0), T(0))
            : Vector3<T>(T(0), T(1), T(0));
        tangent = atlas::math::cross(normal, axis);
    }

    return atlas::math::normalize(tangent);
}

} // namespace detail

template <typename T>
typename MaxwellianSurfaceInteraction<T>::Builder
MaxwellianSurfaceInteraction<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
MaxwellianSurfaceInteraction<T>::set_temperature(const T temperature) noexcept {
    _temperature = temperature;
}

template <typename T>
void
MaxwellianSurfaceInteraction<T>::set_molecular_mass(const T molecular_mass) noexcept {
    _molecular_mass = molecular_mass;
}

template <typename T>
void
MaxwellianSurfaceInteraction<T>::set_momentum_acc(const T momentum_acc) noexcept {
    _momentum_acc = momentum_acc;
}

template <typename T>
void
MaxwellianSurfaceInteraction<T>::set_trans_acc(const T trans_acc) noexcept {
    _trans_acc = trans_acc;
}

template <typename T>
void
MaxwellianSurfaceInteraction<T>::set_rot_acc(const T rot_acc) noexcept {
    _rot_acc = rot_acc;
}

template <typename T>
void
MaxwellianSurfaceInteraction<T>::set_vib_acc(const T vib_acc) noexcept {
    _vib_acc = vib_acc;
}

template <typename T>
void
MaxwellianSurfaceInteraction<T>::set_rot_style(const MaxwellianInternalEnergyStyle style) noexcept {
    _rot_style = style;
}

template <typename T>
void
MaxwellianSurfaceInteraction<T>::set_vib_style(const MaxwellianInternalEnergyStyle style) noexcept {
    _vib_style = style;
}

template <typename T>
void
MaxwellianSurfaceInteraction<T>::set_accommodation(const T momentum_acc,
                                                   const T trans_acc,
                                                   const T rot_acc,
                                                   const T vib_acc) noexcept {
    _momentum_acc = momentum_acc;
    _trans_acc    = trans_acc;
    _rot_acc      = rot_acc;
    _vib_acc      = vib_acc;
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::temperature() const noexcept {
    return _temperature;
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::molecular_mass() const noexcept {
    return _molecular_mass;
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::momentum_acc() const noexcept {
    return _momentum_acc;
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::trans_acc() const noexcept {
    return _trans_acc;
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::rot_acc() const noexcept {
    return _rot_acc;
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::vib_acc() const noexcept {
    return _vib_acc;
}

template <typename T>
MaxwellianInternalEnergyStyle
MaxwellianSurfaceInteraction<T>::rot_style() const noexcept {
    return _rot_style;
}

template <typename T>
MaxwellianInternalEnergyStyle
MaxwellianSurfaceInteraction<T>::vib_style() const noexcept {
    return _vib_style;
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::most_probable_speed() const noexcept {
    return std::sqrt(T(2) * static_cast<T>(atlas::boltzmann_constant) * _temperature / _molecular_mass);
}

template <typename T>
Vector3<T>
MaxwellianSurfaceInteraction<T>::operator()(const Vector3<T>& incident,
                                            const Vector3<T>& normal) const noexcept {
    const T branch_sample = atlas::sampling::sample_hashed_unit_interval(
        incident + normal * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
        T(atlas::seed::RANDOM_HASH_SALT_MIX));

    const T perpendicular_sample = atlas::sampling::sample_hashed_unit_interval(
        incident,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1));

    const T theta_sample = atlas::sampling::sample_hashed_unit_interval(
        normal + incident,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));

    const T tangent_sample = atlas::sampling::sample_hashed_unit_interval(
        incident + normal * T(3.0),
        T(4.19));

    const Vector3<T> tangent_seed(
        atlas::sampling::sample_hashed_unit_interval(incident, T(5.11)),
        atlas::sampling::sample_hashed_unit_interval(normal, T(6.23)),
        atlas::sampling::sample_hashed_unit_interval(incident + normal, T(7.37)));

    return sample(
        incident,
        normal,
        branch_sample,
        perpendicular_sample,
        theta_sample,
        tangent_sample,
        tangent_seed);
}

template <typename T>
Vector3<T>
MaxwellianSurfaceInteraction<T>::sample(const Vector3<T>& incident,
                                        const Vector3<T>& normal,
                                        const T branch_sample,
                                        const T perpendicular_sample,
                                        const T theta_sample,
                                        const T tangent_sample,
                                        const Vector3<T>& tangent_seed) const noexcept {
    if (branch_sample > _momentum_acc) {
        return atlas::math::reflected(incident, normal);
    }

    const T vrm      = most_probable_speed();
    const T vperp    = vrm * std::sqrt(-std::log(detail::maxwellian_unit_sample(perpendicular_sample)));
    const T theta    = T(2) * static_cast<T>(atlas::pi) * theta_sample;
    const T vtangent = vrm * std::sqrt(-std::log(detail::maxwellian_unit_sample(tangent_sample)));
    const T vtan1    = vtangent * std::sin(theta);
    const T vtan2    = vtangent * std::cos(theta);

    const T dot = atlas::math::dot(incident, normal);
    Vector3<T> tangent1(
        incident.x - dot * normal.x,
        incident.y - dot * normal.y,
        incident.z - dot * normal.z);

    if (tangent1.length_squared() == T(0)) {
        tangent1 = detail::maxwellian_tangent_seed(normal, tangent_seed);
    } else {
        tangent1 = atlas::math::normalize(tangent1);
    }

    const Vector3<T> tangent2 = atlas::math::cross(normal, tangent1);

    return normal * vperp + tangent1 * vtan1 + tangent2 * vtan2;
}

template <typename T>
fluid::FluidInternalEnergy<T>
MaxwellianSurfaceInteraction<T>::internal_energy(
    const fluid::FluidInternalEnergy<T>& incident) const noexcept {
    const Vector3<T> seed(incident.translational, incident.rotational, incident.vibrational);

    return sample_internal_energy(
        incident,
        atlas::sampling::sample_hashed_unit_interval(seed, T(8.11)),
        atlas::sampling::sample_hashed_unit_interval(seed, T(9.23)),
        atlas::sampling::sample_hashed_unit_interval(seed, T(10.37)),
        atlas::sampling::sample_hashed_unit_interval(seed, T(11.41)),
        atlas::sampling::sample_hashed_unit_interval(seed, T(12.53)),
        atlas::sampling::sample_hashed_unit_interval(seed, T(13.67)));
}

template <typename T>
fluid::FluidInternalEnergy<T>
MaxwellianSurfaceInteraction<T>::internal_energy(
    const fluid::FluidInternalEnergy<T>& incident_energy,
    const Vector3<T>& incident_velocity,
    const Vector3<T>& normal) const noexcept {
    const T branch_sample = atlas::sampling::sample_hashed_unit_interval(
        incident_velocity + normal * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
        T(atlas::seed::RANDOM_HASH_SALT_MIX));

    return branch_sample > _momentum_acc ? incident_energy : internal_energy(incident_energy);
}

template <typename T>
fluid::FluidInternalEnergy<T>
MaxwellianSurfaceInteraction<T>::internal_energy(
    const fluid::FluidInternalEnergy<T>& incident_energy,
    const Vector3<T>& incident_velocity,
    const Vector3<T>& normal,
    const MaterialProperties<T>& material) const noexcept {
    const T branch_sample = atlas::sampling::sample_hashed_unit_interval(
        incident_velocity + normal * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
        T(atlas::seed::RANDOM_HASH_SALT_MIX));

    if (branch_sample > _momentum_acc) {
        return incident_energy;
    }

    const Vector3<T> seed(
        incident_energy.translational + incident_velocity.x,
        incident_energy.rotational + incident_velocity.y,
        incident_energy.vibrational + incident_velocity.z);

    return fluid::FluidInternalEnergy<T> {
        incident_energy.translational,
        sample_diffuse_rotational_energy(material, seed),
        sample_diffuse_vibrational_energy(material, seed)
    };
}

template <typename T>
fluid::FluidInternalEnergy<T>
MaxwellianSurfaceInteraction<T>::sample_internal_energy(
    const fluid::FluidInternalEnergy<T>& incident,
    const T trans_sample,
    const T trans_theta_sample,
    const T rot_sample,
    const T rot_theta_sample,
    const T vib_sample,
    const T vib_theta_sample) const noexcept {
    return fluid::FluidInternalEnergy<T> {
        sample_internal_energy_mode(incident.translational, _trans_acc, trans_sample, trans_theta_sample),
        sample_internal_energy_mode(incident.rotational, _rot_acc, rot_sample, rot_theta_sample),
        sample_internal_energy_mode(incident.vibrational, _vib_acc, vib_sample, vib_theta_sample)
    };
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::sample_internal_energy_mode(const T incident,
                                                             const T acc,
                                                             const T sample,
                                                             const T theta_sample) const noexcept {
    const T wall_energy = static_cast<T>(atlas::boltzmann_constant) * _temperature;
    const T safe_wall   = std::max(wall_energy, T(atlas::eps));
    const T safe_energy = std::max(incident, T(0));
    const T magnitude   = std::sqrt(safe_energy * (T(1) - acc) / safe_wall);
    const T radius      = std::sqrt(-acc * std::log(detail::maxwellian_unit_sample(sample)));
    const T phase       = std::cos(T(2) * static_cast<T>(atlas::pi) * theta_sample);

    return safe_wall * (radius * radius + magnitude * magnitude + T(2) * radius * magnitude * phase);
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::sample_diffuse_rotational_energy(
    const MaterialProperties<T>& material,
    const Vector3<T>& seed) const noexcept {
    const int dof = material.rotational_dof.has_value() ? *material.rotational_dof : 0;
    if (_rot_style == MaxwellianInternalEnergyStyle::none || dof < 2) {
        return T(0);
    }

    if (_rot_style == MaxwellianInternalEnergyStyle::discrete && dof == 2) {
        const T rot_temperature = material.rotational_temperature.has_value()
            ? *material.rotational_temperature
            : _temperature;
        const T quantum = std::max(rot_temperature, T(atlas::eps));
        const T sample = detail::maxwellian_unit_sample(
            atlas::sampling::sample_hashed_unit_interval(seed, T(14.11)));
        const int level = static_cast<int>(-std::log(sample) * _temperature / quantum);
        return static_cast<T>(level) * static_cast<T>(atlas::boltzmann_constant) * quantum;
    }

    return sample_diffuse_smooth_energy(dof, seed, T(15.23));
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::sample_diffuse_vibrational_energy(
    const MaterialProperties<T>& material,
    const Vector3<T>& seed) const noexcept {
    const int dof = material.vibrational_dof.has_value() ? *material.vibrational_dof : 0;
    if (_vib_style == MaxwellianInternalEnergyStyle::none || dof < 2) {
        return T(0);
    }

    if (_vib_style == MaxwellianInternalEnergyStyle::discrete && dof == 2) {
        const T vib_temperature = material.characteristic_vibrational_temperature.has_value()
            ? *material.characteristic_vibrational_temperature
            : _temperature;
        const T quantum = std::max(vib_temperature, T(atlas::eps));
        const T sample = detail::maxwellian_unit_sample(
            atlas::sampling::sample_hashed_unit_interval(seed, T(16.37)));
        const int level = static_cast<int>(-std::log(sample) * _temperature / quantum);
        return static_cast<T>(level) * static_cast<T>(atlas::boltzmann_constant) * quantum;
    }

    return sample_diffuse_smooth_energy(dof, seed, T(17.41));
}

template <typename T>
T
MaxwellianSurfaceInteraction<T>::sample_diffuse_smooth_energy(const int dof,
                                                              const Vector3<T>& seed,
                                                              const T salt) const noexcept {
    if (dof == 2) {
        const T sample = detail::maxwellian_unit_sample(
            atlas::sampling::sample_hashed_unit_interval(seed, salt));
        return -std::log(sample) * static_cast<T>(atlas::boltzmann_constant) * _temperature;
    }

    const T a = T(0.5) * static_cast<T>(dof) - T(1);
    if (!(a > T(0))) {
        return T(0);
    }

    for (int i = 0;; ++i) {
        const T energy_sample = atlas::sampling::sample_hashed_unit_interval(seed, salt + T(i) * T(0.37));
        const T accept_sample = atlas::sampling::sample_hashed_unit_interval(seed, salt + T(i) * T(0.37) + T(0.19));
        const T erm = T(10) * energy_sample;
        const T b = std::pow(erm / a, a) * std::exp(a - erm);
        if (b > accept_sample) {
            return erm * static_cast<T>(atlas::boltzmann_constant) * _temperature;
        }
    }

    return T(0);
}

template <typename T>
typename MaxwellianSurfaceInteraction<T>::Builder&
MaxwellianSurfaceInteraction<T>::Builder::with_temperature(const T temperature) noexcept {
    _temperature = temperature;
    return *this;
}

template <typename T>
typename MaxwellianSurfaceInteraction<T>::Builder&
MaxwellianSurfaceInteraction<T>::Builder::with_molecular_mass(const T molecular_mass) noexcept {
    _molecular_mass = molecular_mass;
    return *this;
}

template <typename T>
typename MaxwellianSurfaceInteraction<T>::Builder&
MaxwellianSurfaceInteraction<T>::Builder::with_momentum_acc(const T momentum_acc) noexcept {
    _momentum_acc = momentum_acc;
    return *this;
}

template <typename T>
typename MaxwellianSurfaceInteraction<T>::Builder&
MaxwellianSurfaceInteraction<T>::Builder::with_trans_acc(const T trans_acc) noexcept {
    _trans_acc = trans_acc;
    return *this;
}

template <typename T>
typename MaxwellianSurfaceInteraction<T>::Builder&
MaxwellianSurfaceInteraction<T>::Builder::with_rot_acc(const T rot_acc) noexcept {
    _rot_acc = rot_acc;
    return *this;
}

template <typename T>
typename MaxwellianSurfaceInteraction<T>::Builder&
MaxwellianSurfaceInteraction<T>::Builder::with_vib_acc(const T vib_acc) noexcept {
    _vib_acc = vib_acc;
    return *this;
}

template <typename T>
typename MaxwellianSurfaceInteraction<T>::Builder&
MaxwellianSurfaceInteraction<T>::Builder::with_rot_style(const MaxwellianInternalEnergyStyle style) noexcept {
    _rot_style = style;
    return *this;
}

template <typename T>
typename MaxwellianSurfaceInteraction<T>::Builder&
MaxwellianSurfaceInteraction<T>::Builder::with_vib_style(const MaxwellianInternalEnergyStyle style) noexcept {
    _vib_style = style;
    return *this;
}

template <typename T>
typename MaxwellianSurfaceInteraction<T>::Builder&
MaxwellianSurfaceInteraction<T>::Builder::with_accommodation(const T momentum_acc,
                                                             const T trans_acc,
                                                             const T rot_acc,
                                                             const T vib_acc) noexcept {
    _momentum_acc = momentum_acc;
    _trans_acc    = trans_acc;
    _rot_acc      = rot_acc;
    _vib_acc      = vib_acc;
    return *this;
}

template <typename T>
MaxwellianSurfaceInteraction<T>
MaxwellianSurfaceInteraction<T>::Builder::build() const {
    validate();

    MaxwellianSurfaceInteraction<T> interaction {};
    interaction.set_temperature(_temperature);
    interaction.set_molecular_mass(_molecular_mass);
    interaction.set_momentum_acc(_momentum_acc);
    interaction.set_trans_acc(_trans_acc);
    interaction.set_rot_acc(_rot_acc);
    interaction.set_vib_acc(_vib_acc);
    interaction.set_rot_style(_rot_style);
    interaction.set_vib_style(_vib_style);
    return interaction;
}

template <typename T>
atlas::host_shared_ptr<MaxwellianSurfaceInteraction<T>>
MaxwellianSurfaceInteraction<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<MaxwellianSurfaceInteraction<T>>(build());
}

template <typename T>
void
MaxwellianSurfaceInteraction<T>::Builder::validate() const {
    if (!std::isfinite(_temperature) || _temperature <= T(0)) {
        throw std::runtime_error(
            "MaxwellianSurfaceInteraction::Builder: temperature must be finite and positive.");
    }

    if (!std::isfinite(_molecular_mass) || _molecular_mass <= T(0)) {
        throw std::runtime_error(
            "MaxwellianSurfaceInteraction::Builder: molecular_mass must be finite and positive.");
    }

    const bool invalid_accommodation =
        !std::isfinite(_momentum_acc) || _momentum_acc < T(0) || _momentum_acc > T(1)
        || !std::isfinite(_trans_acc) || _trans_acc < T(0) || _trans_acc > T(1)
        || !std::isfinite(_rot_acc) || _rot_acc < T(0) || _rot_acc > T(1)
        || !std::isfinite(_vib_acc) || _vib_acc < T(0) || _vib_acc > T(1);

    if (invalid_accommodation) {
        throw std::runtime_error(
            "MaxwellianSurfaceInteraction::Builder: accommodation coefficients must be finite and within [0, 1].");
    }
}

} // namespace atlas::system
