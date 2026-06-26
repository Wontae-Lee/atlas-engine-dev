#pragma once

namespace atlas {

template <typename T>
void
PreciseColliderKernel<T>::sweep_motion(const Unit<T>& unit,
                                       const Vector3<T>& origin,
                                       const Vector3<T>& incident,
                                       const T,
                                       const T dt,
                                       Vector3<T>& sweep_direction,
                                       T& sweep_speed,
                                       T& sweep_length) const noexcept {
    const Vector3<T> relative_velocity = incident - FastColliderKernel<T>::surface_velocity(unit, origin);
    sweep_direction                    = relative_velocity * dt;
    sweep_speed                        = relative_velocity.length();
    sweep_length                       = sweep_speed * dt;
}

template <typename T>
void
PreciseColliderKernel<T>::operator()(Vector3<T>& position,
                                     Vector3<T>& velocity,
                                     const Vector3<T>& incident,
                                     const Vector3<T>& hit_position,
                                     const Vector3<T>& hit_normal,
                                     const T hit_distance,
                                     const T sweep_speed,
                                     const T dt,
                                     const Unit<T>& unit,
                                     const SurfaceInteractionKernel<T>& interaction) const noexcept {
    DtRemainColliderKernel<T> {}(
        position,
        velocity,
        incident,
        hit_position,
        hit_normal,
        hit_distance,
        sweep_speed,
        dt,
        unit,
        interaction);
}

}