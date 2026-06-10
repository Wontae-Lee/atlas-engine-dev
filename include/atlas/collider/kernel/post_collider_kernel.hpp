#pragma once

namespace atlas::system {

template <typename T>
PostColliderKernel<T>::PostColliderKernel() noexcept
    : type(PostColliderType::fast)
    , fast() {
}

template <typename T>
PostColliderKernel<T>::PostColliderKernel(const PostColliderType type_) noexcept
    : type(type_) {
    switch (type) {
    case PostColliderType::fast:
        new (&fast) FastColliderKernel<T>();
        break;
    case PostColliderType::dt_remain:
        new (&dt_remain) DtRemainColliderKernel<T>();
        break;
    case PostColliderType::precise:
        new (&precise) PreciseColliderKernel<T>();
        break;
    }
}

template <typename T>
PostColliderKernel<T>::PostColliderKernel(const PostColliderKernel& other) noexcept {
    copy_from(other);
}

template <typename T>
PostColliderKernel<T>&
PostColliderKernel<T>::operator=(const PostColliderKernel& other) noexcept {
    if (this != &other) {
        destroy_active();
        copy_from(other);
    }

    return *this;
}

template <typename T>
PostColliderKernel<T>::~PostColliderKernel() noexcept {
    destroy_active();
}

template <typename T>
void
PostColliderKernel<T>::sweep_motion(const Unit<T>& unit,
                                    const Vector3<T>& origin,
                                    const Vector3<T>& incident,
                                    const T incident_speed,
                                    const T dt,
                                    Vector3<T>& sweep_direction,
                                    T& sweep_speed,
                                    T& sweep_length) const noexcept {
    switch (type) {
    case PostColliderType::fast:
        fast.sweep_motion(unit, origin, incident, incident_speed, dt, sweep_direction, sweep_speed, sweep_length);
        return;
    case PostColliderType::dt_remain:
        dt_remain.sweep_motion(unit, origin, incident, incident_speed, dt, sweep_direction, sweep_speed, sweep_length);
        return;
    case PostColliderType::precise:
        precise.sweep_motion(unit, origin, incident, incident_speed, dt, sweep_direction, sweep_speed, sweep_length);
        return;
    }
}

template <typename T>
void
PostColliderKernel<T>::operator()(Vector3<T>& position,
                                  Vector3<T>& velocity,
                                  const Vector3<T>& incident,
                                  const Vector3<T>& hit_position,
                                  const Vector3<T>& hit_normal,
                                  const T hit_distance,
                                  const T sweep_speed,
                                  const T dt,
                                  const Unit<T>& unit,
                                  const SurfaceInteractionKernel<T>& interaction) const noexcept {
    switch (type) {
    case PostColliderType::fast:
        fast(position, velocity, incident, hit_position, hit_normal, hit_distance, sweep_speed, dt, unit, interaction);
        break;
    case PostColliderType::dt_remain:
        dt_remain(position, velocity, incident, hit_position, hit_normal, hit_distance, sweep_speed, dt, unit, interaction);
        break;
    case PostColliderType::precise:
        precise(position, velocity, incident, hit_position, hit_normal, hit_distance, sweep_speed, dt, unit, interaction);
        break;
    }
}

template <typename T>
void
PostColliderKernel<T>::destroy_active() noexcept {
    switch (type) {
    case PostColliderType::fast:
        fast.~FastColliderKernel<T>();
        break;
    case PostColliderType::dt_remain:
        dt_remain.~DtRemainColliderKernel<T>();
        break;
    case PostColliderType::precise:
        precise.~PreciseColliderKernel<T>();
        break;
    }
}

template <typename T>
void
PostColliderKernel<T>::copy_from(const PostColliderKernel& other) noexcept {
    type = other.type;

    switch (type) {
    case PostColliderType::fast:
        new (&fast) FastColliderKernel<T>(other.fast);
        break;
    case PostColliderType::dt_remain:
        new (&dt_remain) DtRemainColliderKernel<T>(other.dt_remain);
        break;
    case PostColliderType::precise:
        new (&precise) PreciseColliderKernel<T>(other.precise);
        break;
    }
}

} // namespace atlas::system
