#pragma once

#include <atlas/logging/logging.h>
#include <atlas/math/constants.h>
#include <atlas/sampling/sampling.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace atlas::system {

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder
PiclasSurfaceInteraction<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_momentum_accommodation(const T accommodation) noexcept {
    _momentum_accommodation = accommodation;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_translational_accommodation(const T accommodation) noexcept {
    _translational_accommodation = accommodation;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_vibrational_accommodation(const T accommodation) noexcept {
    _vibrational_accommodation = accommodation;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_rotational_accommodation(const T accommodation) noexcept {
    _rotational_accommodation = accommodation;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_electronic_accommodation(const T accommodation) noexcept {
    _electronic_accommodation = accommodation;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_only_specular(const bool only_specular) noexcept {
    _only_specular = only_specular;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_only_diffuse(const bool only_diffuse) noexcept {
    _only_diffuse = only_diffuse;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_temperature(const T temperature) noexcept {
    _temperature = temperature;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_molecular_mass(const T molecular_mass) noexcept {
    _molecular_mass = molecular_mass;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_particle_properties(const MaterialProperties<T>& properties) noexcept {
    _particle_properties = properties;
    if (properties.molecular_mass > T(0)) {
        _molecular_mass = properties.molecular_mass;
    }
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_wall_properties(const MaterialProperties<T>& properties) noexcept {
    _wall_properties = properties;
    if (properties.reference_temperature.has_value()) {
        _temperature = properties.reference_temperature.value();
    }
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_wall_velocity(const Vector3<T>& velocity) noexcept {
    _wall_velocity = velocity;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_wall_angular_velocity(const Vector3<T>& angular_velocity) noexcept {
    _wall_angular_velocity = angular_velocity;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_wall_rotation_origin(const Vector3<T>& origin) noexcept {
    _wall_rotation_origin = origin;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_normal_points_out_of_domain(const bool points_out_of_domain) noexcept {
    _normal_points_out_of_domain = points_out_of_domain;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_use_dsmc(const bool use_dsmc) noexcept {
    _use_dsmc = use_dsmc;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_collision_mode(const int collision_mode) noexcept {
    _collision_mode = collision_mode;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_vibrational_relaxation_enabled(const bool enabled) noexcept {
    _vibrational_relaxation_enabled = enabled;
}

template <typename T>
void
PiclasSurfaceInteraction<T>::set_electronic_relaxation_enabled(const bool enabled) noexcept {
    _electronic_relaxation_enabled = enabled;
}

template <typename T>
T
PiclasSurfaceInteraction<T>::momentum_accommodation() const noexcept {
    return _momentum_accommodation;
}

template <typename T>
T
PiclasSurfaceInteraction<T>::translational_accommodation() const noexcept {
    return _translational_accommodation;
}

template <typename T>
T
PiclasSurfaceInteraction<T>::vibrational_accommodation() const noexcept {
    return _vibrational_accommodation;
}

template <typename T>
T
PiclasSurfaceInteraction<T>::rotational_accommodation() const noexcept {
    return _rotational_accommodation;
}

template <typename T>
T
PiclasSurfaceInteraction<T>::electronic_accommodation() const noexcept {
    return _electronic_accommodation;
}

template <typename T>
bool
PiclasSurfaceInteraction<T>::only_specular() const noexcept {
    return _only_specular;
}

template <typename T>
bool
PiclasSurfaceInteraction<T>::only_diffuse() const noexcept {
    return _only_diffuse;
}

template <typename T>
T
PiclasSurfaceInteraction<T>::temperature() const noexcept {
    return _temperature;
}

template <typename T>
T
PiclasSurfaceInteraction<T>::molecular_mass() const noexcept {
    return _molecular_mass;
}

template <typename T>
const MaterialProperties<T>&
PiclasSurfaceInteraction<T>::particle_properties() const noexcept {
    return _particle_properties;
}

template <typename T>
const MaterialProperties<T>&
PiclasSurfaceInteraction<T>::wall_properties() const noexcept {
    return _wall_properties;
}

template <typename T>
const Vector3<T>&
PiclasSurfaceInteraction<T>::wall_velocity() const noexcept {
    return _wall_velocity;
}

template <typename T>
const Vector3<T>&
PiclasSurfaceInteraction<T>::wall_angular_velocity() const noexcept {
    return _wall_angular_velocity;
}

template <typename T>
const Vector3<T>&
PiclasSurfaceInteraction<T>::wall_rotation_origin() const noexcept {
    return _wall_rotation_origin;
}

template <typename T>
bool
PiclasSurfaceInteraction<T>::normal_points_out_of_domain() const noexcept {
    return _normal_points_out_of_domain;
}

template <typename T>
bool
PiclasSurfaceInteraction<T>::use_dsmc() const noexcept {
    return _use_dsmc;
}

template <typename T>
int
PiclasSurfaceInteraction<T>::collision_mode() const noexcept {
    return _collision_mode;
}

template <typename T>
bool
PiclasSurfaceInteraction<T>::vibrational_relaxation_enabled() const noexcept {
    return _vibrational_relaxation_enabled;
}

template <typename T>
bool
PiclasSurfaceInteraction<T>::electronic_relaxation_enabled() const noexcept {
    return _electronic_relaxation_enabled;
}

template <typename T>
Vector3<T>
PiclasSurfaceInteraction<T>::operator()(const Vector3<T>& incident,
                                        const Vector3<T>& normal) const noexcept {
    const T incident_speed_squared = incident.length_squared();
    if (!(incident_speed_squared > T(atlas::tol))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    const Vector3<T> specular = atlas::math::reflected(incident, normal);
    if (_only_specular || _momentum_accommodation <= T(0)) {
        return specular;
    }

    if (_only_diffuse) {
        return diffuse_reflection(incident, normal);
    }

    const T mix = atlas::sampling::sample_hashed_unit_interval(
        incident + normal * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
        T(atlas::seed::RANDOM_HASH_SALT_MIX));
    if (mix >= _momentum_accommodation) {
        return specular;
    }

    return diffuse_reflection(incident, normal);
}

template <typename T>
typename PiclasSurfaceInteraction<T>::ParticleState
PiclasSurfaceInteraction<T>::collide(const ParticleState& state,
                                     const Vector3<T>& point_of_intersection,
                                     const Vector3<T>& normal,
                                     const T remaining_time) const noexcept {
    ParticleState result = state;

    const Vector3<T> local_wall_velocity = wall_velocity_at(point_of_intersection);
    const Vector3<T> relative_incident = state.velocity - local_wall_velocity;
    const Vector3<T> reflected_relative = (*this)(relative_incident, normal);
    const Vector3<T> reflected_velocity = reflected_relative + local_wall_velocity;

    result.last_position = point_of_intersection;
    result.velocity = reflected_velocity;
    result.position = point_of_intersection + reflected_velocity * remaining_time;
    result.trajectory = result.position - result.last_position;
    result.trajectory_length = result.trajectory.length();
    if (result.trajectory_length > T(atlas::tol)) {
        result.trajectory = result.trajectory / result.trajectory_length;
    } else {
        result.trajectory = Vector3<T>(T(0), T(0), T(0));
        result.trajectory_length = T(0);
    }
    result.internal_energy = accommodate_internal_energy(
        state.internal_energy,
        point_of_intersection + reflected_velocity);

    return result;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::InternalEnergyState
PiclasSurfaceInteraction<T>::accommodate_internal_energy(const InternalEnergyState& old_energy,
                                                        const Vector3<T>& seed) const noexcept {
    InternalEnergyState energy = old_energy;
    if (!_use_dsmc || _collision_mode <= 1) {
        return energy;
    }

    const int interaction_id = _particle_properties.interaction_id.value_or(2);
    const bool fully_ionized = _particle_properties.fully_ionized.value_or(false);

    if (should_accommodate_rotational_vibrational(interaction_id)) {
        const T rotational_sample = atlas::sampling::sample_hashed_unit_interval(
            seed,
            T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1));
        if (rotational_sample < _rotational_accommodation) {
            energy.rotational = _wall_properties.rotational_energy.value_or(
                _particle_properties.rotational_energy.value_or(energy.rotational));
        }

        if (_vibrational_relaxation_enabled
            && _particle_properties.characteristic_vibrational_temperature.has_value()) {
            const T vibrational_sample = atlas::sampling::sample_hashed_unit_interval(
                seed + Vector3<T>(T(1), T(0), T(0)),
                T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));
            if (vibrational_sample < _vibrational_accommodation) {
                const T quantum_sample = atlas::sampling::sample_hashed_unit_interval(
                    seed + Vector3<T>(T(0), T(1), T(0)),
                    T(atlas::seed::RANDOM_HASH_SALT_MIX));
                const int quantum = sample_vibrational_quantum(
                    quantum_sample,
                    _particle_properties.characteristic_vibrational_temperature.value(),
                    _particle_properties.max_vibrational_quantum.value_or(0));
                energy.vibrational = (static_cast<T>(quantum) + _particle_properties.gamma_quant.value_or(T(0)))
                    * static_cast<T>(atlas::boltzmann_constant)
                    * _particle_properties.characteristic_vibrational_temperature.value();
            }
        }
    }

    if (_electronic_relaxation_enabled
        && should_accommodate_electronic(interaction_id, fully_ionized)) {
        const T electronic_sample = atlas::sampling::sample_hashed_unit_interval(
            seed + Vector3<T>(T(0), T(0), T(1)),
            T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1 + atlas::seed::RANDOM_HASH_SALT_MIX));
        if (electronic_sample < _electronic_accommodation) {
            energy.electronic = _wall_properties.electronic_energy.value_or(
                _particle_properties.electronic_energy.value_or(energy.electronic));
        }
    }

    return energy;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::InternalEnergyState
PiclasSurfaceInteraction<T>::accommodate_internal_energy(const InternalEnergyState& old_energy,
                                                        const InternalEnergyParameters& parameters,
                                                        const Vector3<T>& seed) const noexcept {
    InternalEnergyState energy = old_energy;
    if (!parameters.use_dsmc || parameters.collision_mode <= 1) {
        return energy;
    }

    if (should_accommodate_rotational_vibrational(parameters.interaction_id)) {
        const T rotational_sample = atlas::sampling::sample_hashed_unit_interval(
            seed,
            T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1));
        if (rotational_sample < _rotational_accommodation) {
            energy.rotational = parameters.rotational_wall_energy;
        }

        if (parameters.enable_vibrational_relaxation) {
            const T vibrational_sample = atlas::sampling::sample_hashed_unit_interval(
                seed + Vector3<T>(T(1), T(0), T(0)),
                T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));
            if (vibrational_sample < _vibrational_accommodation) {
                const T quantum_sample = atlas::sampling::sample_hashed_unit_interval(
                    seed + Vector3<T>(T(0), T(1), T(0)),
                    T(atlas::seed::RANDOM_HASH_SALT_MIX));
                const int quantum = sample_vibrational_quantum(
                    quantum_sample,
                    parameters.characteristic_vibrational_temperature,
                    parameters.max_vibrational_quantum);
                energy.vibrational = (static_cast<T>(quantum) + parameters.gamma_quant)
                    * static_cast<T>(atlas::boltzmann_constant)
                    * parameters.characteristic_vibrational_temperature;
            }
        }
    }

    if (parameters.enable_electronic_relaxation
        && should_accommodate_electronic(parameters.interaction_id, parameters.fully_ionized)) {
        const T electronic_sample = atlas::sampling::sample_hashed_unit_interval(
            seed + Vector3<T>(T(0), T(0), T(1)),
            T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1 + atlas::seed::RANDOM_HASH_SALT_MIX));
        if (electronic_sample < _electronic_accommodation) {
            energy.electronic = parameters.electronic_wall_energy;
        }
    }

    return energy;
}

template <typename T>
T
PiclasSurfaceInteraction<T>::effective_molecular_mass() const noexcept {
    return _particle_properties.molecular_mass > T(0)
        ? _particle_properties.molecular_mass
        : _molecular_mass;
}

template <typename T>
bool
PiclasSurfaceInteraction<T>::should_accommodate_rotational_vibrational(const int interaction_id) const noexcept {
    return interaction_id == 2 || interaction_id == 20;
}

template <typename T>
bool
PiclasSurfaceInteraction<T>::should_accommodate_electronic(const int interaction_id,
                                                           const bool fully_ionized) const noexcept {
    return !fully_ionized && interaction_id != 4 && interaction_id != 100;
}

template <typename T>
Vector3<T>
PiclasSurfaceInteraction<T>::wall_velocity_at(const Vector3<T>& point) const noexcept {
    return _wall_velocity + atlas::math::cross(
        _wall_angular_velocity,
        point - _wall_rotation_origin);
}

template <typename T>
Vector3<T>
PiclasSurfaceInteraction<T>::diffuse_normal(const Vector3<T>& normal) const noexcept {
    return _normal_points_out_of_domain ? -normal : normal;
}

template <typename T>
Vector3<T>
PiclasSurfaceInteraction<T>::diffuse_reflection(const Vector3<T>& incident,
                                                const Vector3<T>& normal) const noexcept {
    const T incident_speed_squared = incident.length_squared();
    const Vector3<T> specular = atlas::math::reflected(incident, normal);
    const Vector3<T> unit_normal = atlas::math::normalize(diffuse_normal(normal));
    const auto tangents = unit_normal.tangential();
    const Vector3<T> tang1 = std::get<0>(tangents);
    const Vector3<T> tang2 = std::get<1>(tangents);

    const T u_radial = atlas::sampling::sample_hashed_unit_interval(
        incident,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1));
    const T u_normal = atlas::sampling::sample_hashed_unit_interval(
        incident + unit_normal,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));
    const T u_energy = atlas::sampling::sample_hashed_unit_interval(
        incident + tang1,
        T(atlas::seed::RANDOM_HASH_SALT_MIX));
    const T u_phi = atlas::sampling::sample_hashed_unit_interval(
        incident + tang2,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1 + atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));

    const T radial_sample = std::max(u_radial, static_cast<T>(atlas::eps));
    const T normal_sample = std::max(u_normal, static_cast<T>(atlas::eps));
    const T radial_speed = static_cast<T>(std::sqrt(static_cast<double>(-std::log(radial_sample))));
    const T normal_speed = static_cast<T>(std::sqrt(static_cast<double>(-std::log(normal_sample))));
    const T speed_factor = radial_speed * radial_speed + normal_speed * normal_speed;
    if (!(speed_factor > T(0))) {
        return specular;
    }

    const T molecular_mass = effective_molecular_mass();
    T translational_energy = T(0.5) * molecular_mass * incident_speed_squared;
    if (u_energy < _translational_accommodation) {
        translational_energy = static_cast<T>(atlas::boltzmann_constant) * _temperature * speed_factor;
    }

    const T scale = static_cast<T>(std::sqrt(static_cast<double>(
        T(2) * translational_energy / (molecular_mass * speed_factor))));
    const T phi = T(2) * static_cast<T>(std::numbers::pi_v<double>) * u_phi;
    const T cos_phi = static_cast<T>(std::cos(static_cast<double>(phi)));
    const T sin_phi = static_cast<T>(std::sin(static_cast<double>(phi)));

    return tang1 * (scale * radial_speed * cos_phi)
        + tang2 * (scale * radial_speed * sin_phi)
        + unit_normal * (scale * normal_speed);
}

template <typename T>
int
PiclasSurfaceInteraction<T>::sample_vibrational_quantum(const T sample,
                                                        const T characteristic_vibrational_temperature,
                                                        const int max_vibrational_quantum) const noexcept {
    if (!(characteristic_vibrational_temperature > T(0))) {
        return 0;
    }

    const T bounded_sample = std::max(sample, static_cast<T>(atlas::eps));
    int quantum = static_cast<int>(-std::log(bounded_sample) * _temperature / characteristic_vibrational_temperature);
    if (max_vibrational_quantum > 0 && quantum >= max_vibrational_quantum) {
        quantum = max_vibrational_quantum - 1;
    }
    return quantum > 0 ? quantum : 0;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_momentum_accommodation(const T accommodation) noexcept {
    _momentum_accommodation = accommodation;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_translational_accommodation(const T accommodation) noexcept {
    _translational_accommodation = accommodation;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_vibrational_accommodation(const T accommodation) noexcept {
    _vibrational_accommodation = accommodation;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_rotational_accommodation(const T accommodation) noexcept {
    _rotational_accommodation = accommodation;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_electronic_accommodation(const T accommodation) noexcept {
    _electronic_accommodation = accommodation;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_only_specular(const bool only_specular) noexcept {
    _only_specular = only_specular;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_only_diffuse(const bool only_diffuse) noexcept {
    _only_diffuse = only_diffuse;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_temperature(const T temperature) noexcept {
    _temperature = temperature;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_molecular_mass(const T molecular_mass) noexcept {
    _molecular_mass = molecular_mass;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_particle_properties(const MaterialProperties<T>& properties) noexcept {
    _particle_properties = properties;
    if (properties.molecular_mass > T(0)) {
        _molecular_mass = properties.molecular_mass;
    }
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_wall_properties(const MaterialProperties<T>& properties) noexcept {
    _wall_properties = properties;
    if (properties.reference_temperature.has_value()) {
        _temperature = properties.reference_temperature.value();
    }
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_wall_velocity(const Vector3<T>& velocity) noexcept {
    _wall_velocity = velocity;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_wall_angular_velocity(const Vector3<T>& angular_velocity) noexcept {
    _wall_angular_velocity = angular_velocity;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_wall_rotation_origin(const Vector3<T>& origin) noexcept {
    _wall_rotation_origin = origin;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_normal_points_out_of_domain(const bool points_out_of_domain) noexcept {
    _normal_points_out_of_domain = points_out_of_domain;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_use_dsmc(const bool use_dsmc) noexcept {
    _use_dsmc = use_dsmc;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_collision_mode(const int collision_mode) noexcept {
    _collision_mode = collision_mode;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_vibrational_relaxation_enabled(const bool enabled) noexcept {
    _vibrational_relaxation_enabled = enabled;
    return *this;
}

template <typename T>
typename PiclasSurfaceInteraction<T>::Builder&
PiclasSurfaceInteraction<T>::Builder::with_electronic_relaxation_enabled(const bool enabled) noexcept {
    _electronic_relaxation_enabled = enabled;
    return *this;
}

template <typename T>
PiclasSurfaceInteraction<T>
PiclasSurfaceInteraction<T>::Builder::build() const {
    validate();

    PiclasSurfaceInteraction<T> interaction {};
    interaction.set_momentum_accommodation(_momentum_accommodation);
    interaction.set_translational_accommodation(_translational_accommodation);
    interaction.set_vibrational_accommodation(_vibrational_accommodation);
    interaction.set_rotational_accommodation(_rotational_accommodation);
    interaction.set_electronic_accommodation(_electronic_accommodation);
    interaction.set_only_specular(_only_specular);
    interaction.set_only_diffuse(_only_diffuse);
    interaction.set_temperature(_temperature);
    interaction.set_molecular_mass(_molecular_mass);
    interaction.set_particle_properties(_particle_properties);
    interaction.set_wall_properties(_wall_properties);
    interaction.set_wall_velocity(_wall_velocity);
    interaction.set_wall_angular_velocity(_wall_angular_velocity);
    interaction.set_wall_rotation_origin(_wall_rotation_origin);
    interaction.set_normal_points_out_of_domain(_normal_points_out_of_domain);
    interaction.set_use_dsmc(_use_dsmc);
    interaction.set_collision_mode(_collision_mode);
    interaction.set_vibrational_relaxation_enabled(_vibrational_relaxation_enabled);
    interaction.set_electronic_relaxation_enabled(_electronic_relaxation_enabled);
    return interaction;
}

template <typename T>
atlas::host_shared_ptr<PiclasSurfaceInteraction<T>>
PiclasSurfaceInteraction<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<PiclasSurfaceInteraction<T>>(build());
}

template <typename T>
void
PiclasSurfaceInteraction<T>::Builder::validate() const {
    if (!std::isfinite(_momentum_accommodation)
        || _momentum_accommodation < T(0) || _momentum_accommodation > T(1)) {
        throw std::runtime_error(
            "PiclasSurfaceInteraction::Builder: momentum accommodation must be finite and within [0, 1].");
    }

    if (!std::isfinite(_translational_accommodation)
        || _translational_accommodation < T(0) || _translational_accommodation > T(1)) {
        throw std::runtime_error(
            "PiclasSurfaceInteraction::Builder: translational accommodation must be finite and within [0, 1].");
    }

    if (!std::isfinite(_vibrational_accommodation)
        || _vibrational_accommodation < T(0) || _vibrational_accommodation > T(1)) {
        throw std::runtime_error(
            "PiclasSurfaceInteraction::Builder: vibrational accommodation must be finite and within [0, 1].");
    }

    if (!std::isfinite(_rotational_accommodation)
        || _rotational_accommodation < T(0) || _rotational_accommodation > T(1)) {
        throw std::runtime_error(
            "PiclasSurfaceInteraction::Builder: rotational accommodation must be finite and within [0, 1].");
    }

    if (!std::isfinite(_electronic_accommodation)
        || _electronic_accommodation < T(0) || _electronic_accommodation > T(1)) {
        throw std::runtime_error(
            "PiclasSurfaceInteraction::Builder: electronic accommodation must be finite and within [0, 1].");
    }

    if (_only_specular && _only_diffuse) {
        throw std::runtime_error(
            "PiclasSurfaceInteraction::Builder: only specular and only diffuse modes are mutually exclusive.");
    }

    if (!std::isfinite(_temperature) || _temperature < T(0)) {
        throw std::runtime_error(
            "PiclasSurfaceInteraction::Builder: temperature must be finite and non-negative.");
    }

    if (!std::isfinite(_molecular_mass) || !(_molecular_mass > T(0))) {
        throw std::runtime_error(
            "PiclasSurfaceInteraction::Builder: molecular mass must be finite and positive.");
    }

    if (_collision_mode < 0) {
        throw std::runtime_error(
            "PiclasSurfaceInteraction::Builder: collision mode must be non-negative.");
    }
}

} // namespace atlas::system
