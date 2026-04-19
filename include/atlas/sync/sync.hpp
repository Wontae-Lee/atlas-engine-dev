#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas::physics {

template <typename T>
constexpr Sync<T>::Sync() noexcept

    : sync_operator() {
    // Default-construct an identity-like sync operator.
}

template <typename T>
constexpr Sync<T>::Sync(const Vector3<T>& translation_,
                        const Quaternion<T>& orientation_) noexcept

    : sync_operator(translation_, orientation_) {
    // Initialize the sync operator directly from the provided rigid pose.
}

template <typename T>
Sync<T>::Sync(const atlas::physics::SyncOperator<T>& op) noexcept

    : sync_operator(op) {
    // Copy the full sync operator state as-is.
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

    // Delegate the actual coordinate conversion to the sync operator.
    sync_operator.sync_to_world(local_point, out);
    return out;
}

template <typename T>
Vector3<T>
Sync<T>::sync_to_local(const Vector3<T>& world_point) const noexcept {

    Vector3<T> out;

    // Delegate the inverse coordinate conversion to the sync operator.
    sync_operator.sync_to_local(world_point, out);
    return out;
}

template <typename T>
Vector3<T>
Sync<T>::sync_dir_to_world(const Vector3<T>& local_dir) const noexcept {

    Vector3<T> out;

    // Transform a direction without applying translation.
    sync_operator.sync_dir_to_world(local_dir, out);
    return out;
}

template <typename T>
Vector3<T>
Sync<T>::sync_dir_to_local(const Vector3<T>& world_dir) const noexcept {

    Vector3<T> out;

    // Transform a world-space direction back into local space.
    sync_operator.sync_dir_to_local(world_dir, out);
    return out;
}

template <typename T>
Ray<T>
Sync<T>::sync_to_world(const Ray<T>& local_ray) const noexcept {

    Ray<T> out;

    // Transform both the ray origin and direction into world space.
    sync_operator.sync_to_world(local_ray, out);
    return out;
}

template <typename T>
Ray<T>
Sync<T>::sync_to_local(const Ray<T>& world_ray) const noexcept {

    Ray<T> out;

    // Transform both the ray origin and direction into local space.
    sync_operator.sync_to_local(world_ray, out);
    return out;
}

template <typename T>
void
Sync<T>::sync_to_world(const Ray<T>& local_ray, Ray<T>& world_ray) const noexcept {

    // Write the transformed ray into the caller-provided output object.
    sync_operator.sync_to_world(local_ray, world_ray);
}

template <typename T>
void
Sync<T>::sync_to_local(const Ray<T>& world_ray, Ray<T>& local_ray) const noexcept {

    // Write the inverse-transformed ray into the caller-provided output object.
    sync_operator.sync_to_local(world_ray, local_ray);
}

template <typename T>
void
Sync<T>::rebuild_matrices() noexcept {

    // Recompute cached matrices so they remain consistent with pose fields.
    sync_operator.rebuild_matrices();
}

template <typename T>
void
Sync<T>::set_translation(const Vector3<T>& translation_) noexcept {

    // Update translation only.
    // Matrix rebuilding is intentionally not forced here.
    sync_operator.translation = translation_;
}

template <typename T>
void
Sync<T>::set_orientation(const Quaternion<T>& orientation_) noexcept {

    // Update orientation and immediately rebuild dependent matrices.
    sync_operator.orientation = orientation_;
    sync_operator.rebuild_matrices();
}

template <typename T>
void
Sync<T>::set_pose(const Vector3<T>& translation_,
                  const Quaternion<T>& orientation_) noexcept {

    // Update the full rigid pose and rebuild cached matrices afterward.
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

    // Return a copy of the internal operator state.
    return sync_operator;
}

template <typename T>
typename Sync<T>::Builder&
Sync<T>::Builder::with_rigid_pose(const Vector3<T>& translation_,
                                  const Quaternion<T>& orientation_) noexcept {

    _translation = translation_;
    _orientation = orientation_;

    // Mark the builder as pose-driven instead of operator-driven.
    _has_sync_operator = false;
    return *this;
}

template <typename T>
typename Sync<T>::Builder&
Sync<T>::Builder::with_sync_operator(const atlas::physics::SyncOperator<T>& op) noexcept {

    _operator = op;

    // Prefer the explicitly supplied operator over the separate pose fields.
    _has_sync_operator = true;
    return *this;
}

template <typename T>
void
Sync<T>::Builder::validate() const {

    // Validate whichever state source is currently active in the builder.
    const Vector3<T>& translation = _has_sync_operator ? _operator.translation : _translation;

    const Quaternion<T>& orientation = _has_sync_operator ? _operator.orientation : _orientation;

    // Reject non-finite translation values because they would corrupt transforms.
    if (!std::isfinite(translation.x)
        || !std::isfinite(translation.y)
        || !std::isfinite(translation.z)) {

        atlas::logger::error()
            << "Sync::Builder: translation contains non-finite values.";
        throw std::runtime_error(
            "Sync::Builder: translation contains non-finite values.");
    }

    // Reject non-finite quaternion components because they invalidate rotation math.
    if (!std::isfinite(orientation.w)
        || !std::isfinite(orientation.x)
        || !std::isfinite(orientation.y)
        || !std::isfinite(orientation.z)) {

        atlas::logger::error()
            << "Sync::Builder: orientation contains non-finite values.";
        throw std::runtime_error(
            "Sync::Builder: orientation contains non-finite values.");
    }

    // A zero quaternion cannot represent a valid rotation.
    if (orientation.w == T(0)
        && orientation.x == T(0)
        && orientation.y == T(0)
        && orientation.z == T(0)) {

        atlas::logger::error()
            << "Sync::Builder: zero quaternion is invalid.";
        throw std::runtime_error(
            "Sync::Builder: zero quaternion is invalid.");
    }

    // Check quaternion magnitude to detect suspicious input.
    // A non-unit quaternion may still be usable depending on downstream normalization,
    // so this condition is reported as a warning instead of an error.
    const T norm = std::sqrt(
        orientation.w * orientation.w
        + orientation.x * orientation.x
        + orientation.y * orientation.y
        + orientation.z * orientation.z);

    if (std::abs(norm - T(1)) > T(1e-3)) {
        atlas::logger::warn()
            << "Sync::Builder: quaternion not normalized.";
    }
}

template <typename T>
Sync<T>
Sync<T>::Builder::build() const {

    validate();

    Sync<T> s {};

    if (_has_sync_operator) {

        // Reuse the fully prepared operator when one was explicitly provided.
        s.sync_operator = _operator;
    } else {

        // Otherwise construct the operator from the stored rigid pose.
        s.sync_operator = atlas::physics::SyncOperator<T>(_translation, _orientation);
    }

    // Ensure derived matrices are synchronized with the final pose state.
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