#pragma once

namespace atlas::system {

template <typename T>
Vector3<T>
FastColliderKernel<T>::surface_velocity(const Unit<T>& unit,
                                        const Vector3<T>& surface_point) noexcept {
    Vector3<T> velocity(T(0), T(0), T(0));

    if (unit.velocity().has_value()) {
        velocity += *unit.velocity();
    }

    if (unit.angular_velocity().has_value()) {
        const Vector3<T> radius = surface_point - unit.sync_operator().translation;
        velocity += atlas::math::cross(*unit.angular_velocity(), radius);
    }

    return velocity;
}

template <typename T>
void
FastColliderKernel<T>::sweep_motion(const Unit<T>&,
                                    const Vector3<T>&,
                                    const Vector3<T>& incident,
                                    const T incident_speed,
                                    const T dt,
                                    Vector3<T>& sweep_direction,
                                    T& sweep_speed,
                                    T& sweep_length) const noexcept {
    sweep_direction = incident * dt;
    sweep_speed     = incident_speed;
    sweep_length    = incident_speed * dt;
}

template <typename T>
void
FastColliderKernel<T>::operator()(Vector3<T>& position,
                                  Vector3<T>& velocity,
                                  const Vector3<T>& incident,
                                  const Vector3<T>& hit_position,
                                  const Vector3<T>& hit_normal,
                                  const T,
                                  const T,
                                  const T,
                                  const Unit<T>& unit,
                                  const ColliderSurfaceInteraction<T>& interaction) const noexcept {
    const Vector3<T> wall_velocity     = surface_velocity(unit, hit_position);
    const Vector3<T> relative_incident = incident - wall_velocity;

    position = hit_position + hit_normal * static_cast<T>(atlas::tol);
    velocity = interaction(relative_incident, hit_normal) + wall_velocity;
}

} // namespace atlas::system
