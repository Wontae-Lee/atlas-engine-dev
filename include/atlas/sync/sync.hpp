#pragma once
#include <cmath>
#include <stdexcept>
#include <utility>
namespace atlas::physics {
template <typename T>
constexpr Sync<T>::Sync() noexcept
    : sync_operator() {
}

template <typename T>
constexpr Sync<T>::Sync(const Vector3<T>& translation_,
                        const Quaternion<T>& orientation_) noexcept
    : sync_operator(translation_, orientation_) {
}

template <typename T>
Sync<T>::Sync(const atlas::physics::SyncOperator<T>& op) noexcept
    : sync_operator(op) {
}

template <typename T>
typename Sync<T>::Builder
Sync<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
Vector3<T>
Sync<T>::sync_to_world(const Vector3<T>& local_point) const noexcept {
    Vector3<T> out;
    sync_operator.sync_to_world(local_point, out);
    return out;
}

template <typename T>
Vector3<T>
Sync<T>::sync_to_local(const Vector3<T>& world_point) const noexcept {
    Vector3<T> out;
    sync_operator.sync_to_local(world_point, out);
    return out;
}

template <typename T>
Vector3<T>
Sync<T>::sync_dir_to_world(const Vector3<T>& local_dir) const noexcept {
    Vector3<T> out;
    sync_operator.sync_dir_to_world(local_dir, out);
    return out;
}

template <typename T>
Vector3<T>
Sync<T>::sync_dir_to_local(const Vector3<T>& world_dir) const noexcept {
    Vector3<T> out;
    sync_operator.sync_dir_to_local(world_dir, out);
    return out;
}

template <typename T>
Ray<T>
Sync<T>::sync_to_world(const Ray<T>& local_ray) const noexcept {
    Ray<T> out;
    sync_operator.sync_to_world(local_ray, out);
    return out;
}

template <typename T>
Ray<T>
Sync<T>::sync_to_local(const Ray<T>& world_ray) const noexcept {
    Ray<T> out;
    sync_operator.sync_to_local(world_ray, out);
    return out;
}

template <typename T>
void
Sync<T>::sync_to_world(const Ray<T>& local_ray, Ray<T>& world_ray) const noexcept {
    sync_operator.sync_to_world(local_ray, world_ray);
}

template <typename T>
void
Sync<T>::sync_to_local(const Ray<T>& world_ray, Ray<T>& local_ray) const noexcept {
    sync_operator.sync_to_local(world_ray, local_ray);
}

template <typename T>
void
Sync<T>::rebuild_matrices() noexcept {
    sync_operator.rebuild_matrices();
}

template <typename T>
void
Sync<T>::set_translation(const Vector3<T>& translation_) noexcept {
    sync_operator.translation = translation_;
}

template <typename T>
void
Sync<T>::set_orientation(const Quaternion<T>& orientation_) noexcept {
    sync_operator.orientation = orientation_;
    sync_operator.rebuild_matrices();
}

template <typename T>
void
Sync<T>::set_pose(const Vector3<T>& translation_,
                  const Quaternion<T>& orientation_) noexcept {
    sync_operator.translation = translation_;
    sync_operator.orientation = orientation_;
    sync_operator.rebuild_matrices();
}

template <typename T>
const atlas::physics::SyncOperator<T>&
Sync<T>::sync() const noexcept {
    return sync_operator;
}

template <typename T>
atlas::physics::SyncOperator<T>
Sync<T>::make_sync_operator() const noexcept {
    return sync_operator;
}

template <typename T>
typename Sync<T>::Builder&
Sync<T>::Builder::with_rigid_pose(const Vector3<T>& translation_,
                                  const Quaternion<T>& orientation_) noexcept {
    _translation       = translation_;
    _orientation       = orientation_;
    _has_sync_operator = false;
    return *this;
}

template <typename T>
typename Sync<T>::Builder&
Sync<T>::Builder::with_sync_operator(const atlas::physics::SyncOperator<T>& op) noexcept {
    _operator          = op;
    _has_sync_operator = true;
    return *this;
}

template <typename T>
void
Sync<T>::Builder::validate() const {
    const Vector3<T>& translation    = _has_sync_operator ? _operator.translation : _translation;
    const Quaternion<T>& orientation = _has_sync_operator ? _operator.orientation : _orientation;
    if (!std::isfinite(translation.x)
        || !std::isfinite(translation.y)
        || !std::isfinite(translation.z)) {
        throw std::runtime_error(
            "Sync::Builder: translation contains non-finite values.");
    }
    if (!std::isfinite(orientation.w)
        || !std::isfinite(orientation.x)
        || !std::isfinite(orientation.y)
        || !std::isfinite(orientation.z)) {
        throw std::runtime_error(
            "Sync::Builder: orientation contains non-finite values.");
    }
    if (orientation.w == T(0)
        && orientation.x == T(0)
        && orientation.y == T(0)
        && orientation.z == T(0)) {
        throw std::runtime_error(
            "Sync::Builder: zero quaternion is invalid.");
    }
    const T norm = std::sqrt(
        orientation.w * orientation.w
        + orientation.x * orientation.x
        + orientation.y * orientation.y
        + orientation.z * orientation.z);
}

template <typename T>
Sync<T>
Sync<T>::Builder::build() const {
    validate();
    Sync<T> s {};
    if (_has_sync_operator) {
        s.sync_operator = _operator;
    } else {
        s.sync_operator = atlas::physics::SyncOperator<T>(_translation, _orientation);
    }
    s.rebuild_matrices();
    return s;
}

template <typename T>
atlas::host_shared_ptr<Sync<T>>
Sync<T>::Builder::make_host_shared() const {
    auto s = build();
    return atlas::make_host_shared<Sync<T>>(std::move(s));
}

}