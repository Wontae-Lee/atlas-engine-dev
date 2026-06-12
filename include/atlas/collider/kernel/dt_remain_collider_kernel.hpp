#pragma once

namespace atlas {

template <typename T>
void
DtRemainColliderKernel<T>::sweep_motion(const Unit<T>&,
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
DtRemainColliderKernel<T>::operator()(Vector3<T>& position,
                                      Vector3<T>& velocity,
                                      const Vector3<T>& incident,
                                      const Vector3<T>& hit_position,
                                      const Vector3<T>& hit_normal,
                                      const T hit_distance,
                                      const T sweep_speed,
                                      const T dt,
                                      const Unit<T>& unit,
                                      const SurfaceInteractionKernel<T>& interaction) const noexcept {
    const Vector3<T> wall_velocity     = FastColliderKernel<T>::surface_velocity(unit, hit_position);
    const Vector3<T> relative_incident = incident - wall_velocity;
    const Vector3<T> reflected         = interaction(relative_incident, hit_normal) + wall_velocity;
    const Vector3<T> offset_position   = hit_position + hit_normal * static_cast<T>(atlas::tol);

    position = offset_position;
    velocity = reflected;

    if (!(sweep_speed > T(atlas::eps))) {
        return;
    }

    const T hit_time  = hit_distance / sweep_speed;
    const T remaining = (dt > hit_time) ? (dt - hit_time) : T(0);

    position = offset_position + reflected * remaining;
}

}