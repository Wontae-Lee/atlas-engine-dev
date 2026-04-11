#pragma once

#include <atlas/sampling/sampling.h>

#include <cmath>

namespace atlas::system {

template <typename T>
HardSphereDsmcOperator<T>::HardSphereDsmcOperator(const T probability_scale,
                                                  const unsigned int seed) noexcept
    : probability_scale(probability_scale)
    , seed(seed) { }

template <typename T>
T
HardSphereDsmcOperator<T>::collision_kernel(const T effective_collision_diameter,
                                            const T relative_speed) const noexcept {
    if (!(effective_collision_diameter > T(0)) || !(relative_speed > T(0))) {
        return T(0);
    }
    return probability_scale * static_cast<T>(atlas::pi) * effective_collision_diameter * effective_collision_diameter * relative_speed;
}

template <typename T>
VhsDsmcOperator<T>::VhsDsmcOperator(const T reference_temperature,
                                    const T probability_scale,
                                    const unsigned int seed) noexcept
    : reference_temperature(reference_temperature)
    , probability_scale(probability_scale)
    , seed(seed) { }

template <typename T>
T
VhsDsmcOperator<T>::collision_kernel(const T effective_collision_diameter,
                                     const T reduced_mass,
                                     const T effective_viscosity_index,
                                     const T local_temperature,
                                     const T relative_speed) const noexcept {
    const T temperature = (local_temperature > T(0)) ? local_temperature : reference_temperature;
    if (!(effective_collision_diameter > T(0)) || !(relative_speed > T(0)) || !(reduced_mass > T(0)) || !(temperature > T(0))) {
        return T(0);
    }

    const T g_ref        = std::sqrt(T(2) * static_cast<T>(atlas::boltzmann_constant) * temperature / reduced_mass);
    const T exponent     = T(2) * (effective_viscosity_index - T(0.5));
    const T safe_speed   = (relative_speed > T(1e-12)) ? relative_speed : T(1e-12);
    const T speed_factor = std::pow(g_ref / safe_speed, exponent);
    return probability_scale * static_cast<T>(atlas::pi) * effective_collision_diameter * effective_collision_diameter * relative_speed * speed_factor;
}

template <typename T>
VssDsmcOperator<T>::VssDsmcOperator(const T reference_temperature,
                                    const T probability_scale,
                                    const unsigned int seed) noexcept
    : reference_temperature(reference_temperature)
    , probability_scale(probability_scale)
    , seed(seed) { }

template <typename T>
T
VssDsmcOperator<T>::collision_kernel(const T effective_collision_diameter,
                                     const T reduced_mass,
                                     const T effective_viscosity_index,
                                     const T effective_scattering_parameter,
                                     const T local_temperature,
                                     const T relative_speed) const noexcept {
    if (!(effective_scattering_parameter > T(0))) {
        return T(0);
    }

    const T temperature = (local_temperature > T(0)) ? local_temperature : reference_temperature;
    if (!(effective_collision_diameter > T(0)) || !(relative_speed > T(0)) || !(reduced_mass > T(0)) || !(temperature > T(0))) {
        return T(0);
    }

    const T g_ref        = std::sqrt(T(2) * static_cast<T>(atlas::boltzmann_constant) * temperature / reduced_mass);
    const T exponent     = T(2) * (effective_viscosity_index - T(0.5));
    const T safe_speed   = (relative_speed > T(1e-12)) ? relative_speed : T(1e-12);
    const T speed_factor = std::pow(g_ref / safe_speed, exponent);
    return probability_scale * static_cast<T>(atlas::pi) * effective_collision_diameter * effective_collision_diameter * relative_speed * speed_factor / effective_scattering_parameter;
}

template <typename T>
DsmcOperator<T>::DsmcOperator() noexcept
    : type(DsmcModelType::hard_sphere) {
    new (&hard_sphere) HardSphereDsmcOperator<T> {};
}

template <typename T>
DsmcOperator<T>::DsmcOperator(const DsmcModelType type,
                              const T param0,
                              const T probability,
                              const unsigned int seed) noexcept
    : type(type) {
    switch (type) {
    case DsmcModelType::hard_sphere:
        new (&hard_sphere) HardSphereDsmcOperator<T>(probability, seed);
        return;
    case DsmcModelType::vhs:
        new (&vhs) VhsDsmcOperator<T>(param0, probability, seed);
        return;
    case DsmcModelType::vss:
        new (&vss) VssDsmcOperator<T>(param0, probability, seed);
        return;
    default:
        this->type = DsmcModelType::hard_sphere;
        new (&hard_sphere) HardSphereDsmcOperator<T>(probability, seed);
        return;
    }
}

template <typename T>
DsmcOperator<T>::DsmcOperator(const DsmcOperator& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
DsmcOperator<T>&
DsmcOperator<T>::operator=(const DsmcOperator& other) noexcept {
    if (this == &other) return *this;
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
DsmcOperator<T>::~DsmcOperator() noexcept {
    destroy_active();
}

template <typename T>
DsmcOperator<T>::DsmcOperator(const HardSphereDsmcOperator<T>& op)
    : type(DsmcModelType::hard_sphere) {
    new (&hard_sphere) HardSphereDsmcOperator<T>(op);
}

template <typename T>
DsmcOperator<T>::DsmcOperator(const VhsDsmcOperator<T>& op)
    : type(DsmcModelType::vhs) {
    new (&vhs) VhsDsmcOperator<T>(op);
}

template <typename T>
DsmcOperator<T>::DsmcOperator(const VssDsmcOperator<T>& op)
    : type(DsmcModelType::vss) {
    new (&vss) VssDsmcOperator<T>(op);
}

template <typename T>
T
DsmcOperator<T>::collision_kernel(const T effective_collision_diameter,
                                  const T reduced_mass,
                                  const T effective_viscosity_index,
                                  const T effective_scattering_parameter,
                                  const T local_temperature,
                                  const T relative_speed) const noexcept {
    switch (type) {
    case DsmcModelType::hard_sphere:
        return hard_sphere.collision_kernel(effective_collision_diameter, relative_speed);
    case DsmcModelType::vhs:
        return vhs.collision_kernel(
            effective_collision_diameter,
            reduced_mass,
            effective_viscosity_index,
            local_temperature,
            relative_speed);
    case DsmcModelType::vss:
        return vss.collision_kernel(
            effective_collision_diameter,
            reduced_mass,
            effective_viscosity_index,
            effective_scattering_parameter,
            local_temperature,
            relative_speed);
    default:
        return T(0);
    }
}

template <typename T>
unsigned int
DsmcOperator<T>::base_seed() const noexcept {
    switch (type) {
    case DsmcModelType::hard_sphere:
        return hard_sphere.seed;
    case DsmcModelType::vhs:
        return vhs.seed;
    case DsmcModelType::vss:
        return vss.seed;
    default:
        return 0u;
    }
}

template <typename T>
bool
DsmcOperator<T>::should_collide(const T effective_collision_diameter,
                                const T reduced_mass,
                                const T effective_viscosity_index,
                                const T effective_scattering_parameter,
                                const T local_temperature,
                                const Vector3<T>& relative_velocity,
                                const std::uint64_t pair_id) const noexcept {
    const T relative_speed = relative_velocity.length();
    const T kernel         = collision_kernel(
        effective_collision_diameter,
        reduced_mass,
        effective_viscosity_index,
        effective_scattering_parameter,
        local_temperature,
        relative_speed);
    if (!(kernel > T(0))) {
        return false;
    }

    atlas::uniform_real_distribution<T> dist(T(0), T(1));
    auto engine = atlas::default_random_engine<T>(
        static_cast<unsigned int>(base_seed() + 1664525u * static_cast<unsigned int>(pair_id) + 1013904223u));
    return dist(engine) < std::min(T(1), kernel);
}

template <typename T>
Vector3<T>
DsmcOperator<T>::scatter_relative_velocity(const T effective_scattering_parameter,
                                           const Vector3<T>& relative_velocity,
                                           const std::uint64_t pair_id) const noexcept {
    const T relative_speed = relative_velocity.length();
    if (!(relative_speed > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    auto engine = atlas::default_random_engine<T>(
        static_cast<unsigned int>((base_seed() ^ 0x9e3779b9u) + 1664525u * static_cast<unsigned int>(pair_id) + 1013904223u));

    Vector3<T> direction = relative_velocity / relative_speed;
    if (type == DsmcModelType::vss) {
        direction = atlas::sampling::sample_directional_unit_vector(direction, effective_scattering_parameter, engine);
    } else {
        direction = atlas::sampling::sample_random_unit_vector<T>(engine);
    }

    return direction * relative_speed;
}

template <typename T>
void
DsmcOperator<T>::destroy_active() noexcept {
    switch (type) {
    case DsmcModelType::hard_sphere:
        hard_sphere.~HardSphereDsmcOperator<T>();
        return;
    case DsmcModelType::vhs:
        vhs.~VhsDsmcOperator<T>();
        return;
    case DsmcModelType::vss:
        vss.~VssDsmcOperator<T>();
        return;
    default:
        hard_sphere.~HardSphereDsmcOperator<T>();
        return;
    }
}

template <typename T>
void
DsmcOperator<T>::copy_from(const DsmcOperator& other) noexcept {
    switch (type) {
    case DsmcModelType::hard_sphere:
        new (&hard_sphere) HardSphereDsmcOperator<T>(other.hard_sphere);
        return;
    case DsmcModelType::vhs:
        new (&vhs) VhsDsmcOperator<T>(other.vhs);
        return;
    case DsmcModelType::vss:
        new (&vss) VssDsmcOperator<T>(other.vss);
        return;
    default:
        type = DsmcModelType::hard_sphere;
        new (&hard_sphere) HardSphereDsmcOperator<T>(other.hard_sphere);
        return;
    }
}

}
