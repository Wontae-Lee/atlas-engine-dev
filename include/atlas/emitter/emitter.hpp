#ifndef INCLUDE_ATLAS_EMITTER_EMITTER_HPP
#define INCLUDE_ATLAS_EMITTER_EMITTER_HPP
#include <atlas/math/math.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/random.h>

namespace atlas::system {

template <typename T>
Emitter<T>::Emitter(T x_spawn_min, T x_spawn_max,
                    T y_spawn_min, T y_spawn_max,
                    T z_spawn_min, T z_spawn_max,
                    T inject_vx_mean, T inject_vx_jit,
                    T inject_vt_jit)
    : x_spawn_min_(x_spawn_min)
    , x_spawn_max_(x_spawn_max)
    , y_spawn_min_(y_spawn_min)
    , y_spawn_max_(y_spawn_max)
    , z_spawn_min_(z_spawn_min)
    , z_spawn_max_(z_spawn_max)
    , inject_vx_mean_(inject_vx_mean)
    , inject_vx_jit_(inject_vx_jit)
    , inject_vt_jit_(inject_vt_jit) { }

template <typename T>
void
Emitter<T>::set_x_spawn_min(T v) {
    x_spawn_min_ = v;
}
template <typename T>
void
Emitter<T>::set_x_spawn_max(T v) {
    x_spawn_max_ = v;
}
template <typename T>
void
Emitter<T>::set_y_spawn_min(T v) {
    y_spawn_min_ = v;
}
template <typename T>
void
Emitter<T>::set_y_spawn_max(T v) {
    y_spawn_max_ = v;
}
template <typename T>
void
Emitter<T>::set_z_spawn_min(T v) {
    z_spawn_min_ = v;
}
template <typename T>
void
Emitter<T>::set_z_spawn_max(T v) {
    z_spawn_max_ = v;
}

template <typename T>
void
Emitter<T>::set_inject_vx_mean(T v) {
    inject_vx_mean_ = v;
}
template <typename T>
void
Emitter<T>::set_inject_vx_jit(T v) {
    inject_vx_jit_ = v;
}
template <typename T>
void
Emitter<T>::set_inject_vt_jit(T v) {
    inject_vt_jit_ = v;
}

template <typename T>
T
Emitter<T>::x_spawn_min() const {
    return x_spawn_min_;
}
template <typename T>
T
Emitter<T>::x_spawn_max() const {
    return x_spawn_max_;
}
template <typename T>
T
Emitter<T>::y_spawn_min() const {
    return y_spawn_min_;
}
template <typename T>
T
Emitter<T>::y_spawn_max() const {
    return y_spawn_max_;
}
template <typename T>
T
Emitter<T>::z_spawn_min() const {
    return z_spawn_min_;
}
template <typename T>
T
Emitter<T>::z_spawn_max() const {
    return z_spawn_max_;
}

template <typename T>
T
Emitter<T>::inject_vx_mean() const {
    return inject_vx_mean_;
}
template <typename T>
T
Emitter<T>::inject_vx_jit() const {
    return inject_vx_jit_;
}
template <typename T>
T
Emitter<T>::inject_vt_jit() const {
    return inject_vt_jit_;
}

template <typename T>
void
Emitter<T>::operator()(const ParticleDeviceProbe<T>& data, int& active) {

    auto device_pos = data.pos;
    auto device_vel = data.vel;

    T x_spawn_min = x_spawn_min_;
    T x_spawn_max = x_spawn_max_;
    T y_spawn_min = y_spawn_min_;
    T y_spawn_max = y_spawn_max_;
    T z_spawn_min = z_spawn_min_;
    T z_spawn_max = z_spawn_max_;

    T inject_vx_mean = inject_vx_mean_;
    T inject_vx_jit  = inject_vx_jit_;
    T inject_vt_jit  = inject_vt_jit_;
    total_emitted += emit_per_step;
    auto current_total = total_emitted;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        emit_per_step,
        [=] ATLAS_DEVICE(int k) {
            int i = active + k;

            uint32_t seed = atlas::random::hash_u32(
                0x9E3779B9u ^ static_cast<uint32_t>(current_total + k + 1));

            T ry = atlas::random::rng(seed);
            T rz = atlas::random::rng(seed);
            T r0 = atlas::random::rng(seed);
            T r1 = atlas::random::rng(seed);
            T r2 = atlas::random::rng(seed);
            T rx = atlas::random::rng(seed);

            T x = x_spawn_min + (x_spawn_max - x_spawn_min) * rx;
            T y = y_spawn_min + (y_spawn_max - y_spawn_min) * ry;
            T z = z_spawn_min + (z_spawn_max - z_spawn_min) * rz;

            T vx = inject_vx_mean + inject_vx_jit * (r0 - T(0.5)) * T(2.0);
            T vy = inject_vt_jit * (r1 - T(0.5)) * T(2.0);
            T vz = inject_vt_jit * (r2 - T(0.5)) * T(2.0);

            device_pos[i] = Vector3<T> { x, y, z };
            device_vel[i] = Vector3<T> { vx, vy, vz };
        });

    active += emit_per_step;
}

}

#endif