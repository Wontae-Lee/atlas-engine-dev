#ifndef INCLUDE_ATLAS_REMOVER_REMOVER_HPP
#define INCLUDE_ATLAS_REMOVER_REMOVER_HPP

#include <atlas/detail/remove.h>
#include <atlas/detail/tuple.h>
#include <atlas/detail/zip_iterator.h>

namespace atlas::system {

template <typename T>
Remover<T>::Remover(T x_min, T x_max,
                    T y_min, T y_max,
                    T z_min, T z_max)
    : x_min_(x_min)
    , x_max_(x_max)
    , y_min_(y_min)
    , y_max_(y_max)
    , z_min_(z_min)
    , z_max_(z_max) { }

template <typename T>
void
Remover<T>::set_x_min(T v) {
    x_min_ = v;
}

template <typename T>
void
Remover<T>::set_x_max(T v) {
    x_max_ = v;
}

template <typename T>
void
Remover<T>::set_y_min(T v) {
    y_min_ = v;
}

template <typename T>
void
Remover<T>::set_y_max(T v) {
    y_max_ = v;
}

template <typename T>
void
Remover<T>::set_z_min(T v) {
    z_min_ = v;
}

template <typename T>
void
Remover<T>::set_z_max(T v) {
    z_max_ = v;
}

template <typename T>
T
Remover<T>::x_min() const {
    return x_min_;
}

template <typename T>
T
Remover<T>::x_max() const {
    return x_max_;
}

template <typename T>
T
Remover<T>::y_min() const {
    return y_min_;
}

template <typename T>
T
Remover<T>::y_max() const {
    return y_max_;
}

template <typename T>
T
Remover<T>::z_min() const {
    return z_min_;
}

template <typename T>
T
Remover<T>::z_max() const {
    return z_max_;
}

template <typename T>
void
Remover<T>::operator()(const ParticleDeviceProbe<T>& data, int& active) const {

    T x_min = x_min_;
    T x_max = x_max_;
    T y_min = y_min_;
    T y_max = y_max_;
    T z_min = z_min_;
    T z_max = z_max_;

    auto pos = data.pos;
    auto vel = data.vel;

    auto zip_begin = atlas::make_zip_iterator(
        atlas::make_tuple(
            pos,
            vel));
    auto zip_end = zip_begin + active;

    auto new_end = atlas::remove_if(
        atlas::device,
        zip_begin,
        zip_end,
        [=] ATLAS_DEVICE(const atlas::tuple<Vector3F, Vector3F>& t) {
            const Vector3F& p = atlas::get<0>(t);

            return (p.x < x_min) || (p.x > x_max) || (p.y < y_min) || (p.y > y_max) || (p.z < z_min) || (p.z > z_max);
        });

    active = static_cast<int>(new_end - zip_begin);
}

}

#endif