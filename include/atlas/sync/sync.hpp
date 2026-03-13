#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
constexpr Sync<T>::Sync() noexcept
    // Default construction delegates to SyncOperator's identity pose:
    // - zero translation
    // - identity rotation
    // - identity cached matrices
    : sync_operator() {
}

template <typename T>
constexpr Sync<T>::Sync(const Vector3<T>& translation_,
                        const Quaternion<T>& orientation_) noexcept
    // Build directly from the explicit rigid pose parameters.
    : sync_operator(translation_, orientation_) {
}

template <typename T>
Sync<T>::Sync(const atlas::system::SyncOperator<T>& op) noexcept
    // Copy the already-prepared low-level operator as-is.
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
    // Sync is intentionally a thin façade. The actual rigid-body mathematics
    // live in SyncOperator; this wrapper forwards to that implementation.
    Vector3<T> out;
    sync_operator.sync_to_world(local_point, out);
    return out;
}

template <typename T>
Vector3<T>
Sync<T>::sync_to_local(const Vector3<T>& world_point) const noexcept {
    // Inverse point transform wrapper.
    Vector3<T> out;
    sync_operator.sync_to_local(world_point, out);
    return out;
}

template <typename T>
Vector3<T>
Sync<T>::sync_dir_to_world(const Vector3<T>& local_dir) const noexcept {
    // Direction transforms omit translation because directions are free vectors.
    Vector3<T> out;
    sync_operator.sync_dir_to_world(local_dir, out);
    return out;
}

template <typename T>
Vector3<T>
Sync<T>::sync_dir_to_local(const Vector3<T>& world_dir) const noexcept {
    // Inverse direction transform wrapper.
    Vector3<T> out;
    sync_operator.sync_dir_to_local(world_dir, out);
    return out;
}

template <typename T>
Ray<T>
Sync<T>::sync_to_world(const Ray<T>& local_ray) const noexcept {
    // For rays, origin is transformed as a point and direction as a direction.
    Ray<T> out;
    sync_operator.sync_to_world(local_ray, out);
    return out;
}

template <typename T>
Ray<T>
Sync<T>::sync_to_local(const Ray<T>& world_ray) const noexcept {
    // Inverse ray transform wrapper.
    Ray<T> out;
    sync_operator.sync_to_local(world_ray, out);
    return out;
}

template <typename T>
void
Sync<T>::sync_to_world(const Ray<T>& local_ray, Ray<T>& world_ray) const noexcept {
    // Output-parameter overload avoids an extra temporary in caller code.
    sync_operator.sync_to_world(local_ray, world_ray);
}

template <typename T>
void
Sync<T>::sync_to_local(const Ray<T>& world_ray, Ray<T>& local_ray) const noexcept {
    // Output-parameter overload for inverse ray transform.
    sync_operator.sync_to_local(world_ray, local_ray);
}

template <typename T>
void
Sync<T>::rebuild_matrices() noexcept {
    // Recompute cached rotation matrices after any orientation mutation.
    sync_operator.rebuild_matrices();
}

template <typename T>
void
Sync<T>::set_translation(const Vector3<T>& translation_) noexcept {
    // Translation affects only the affine offset term t, not the rotation
    // matrix cache, so no rebuild is needed here.
    sync_operator.translation = translation_;
}

template <typename T>
void
Sync<T>::set_orientation(const Quaternion<T>& orientation_) noexcept {
    // Changing orientation changes the rotation matrix R and its inverse, so
    // cached matrices must be rebuilt immediately to preserve correctness.
    sync_operator.orientation = orientation_;
    sync_operator.rebuild_matrices();
}

template <typename T>
void
Sync<T>::set_pose(const Vector3<T>& translation_,
                  const Quaternion<T>& orientation_) noexcept {
    // Update both parts of the rigid pose atomically from the caller's
    // perspective, then rebuild the matrix caches once.
    sync_operator.translation = translation_;
    sync_operator.orientation = orientation_;
    sync_operator.rebuild_matrices();
}

template <typename T>
const atlas::system::SyncOperator<T>&
Sync<T>::sync() const noexcept {
    // Expose the underlying operator by const reference to avoid copies.
    return sync_operator;
}

template <typename T>
atlas::system::SyncOperator<T>
Sync<T>::make_sync_operator() const noexcept {
    // Value-returning accessor when the caller needs an independent copy.
    return sync_operator;
}

template <typename T>
typename Sync<T>::Builder&
Sync<T>::Builder::with_rigid_pose(const Vector3<T>& translation_,
                                  const Quaternion<T>& orientation_) noexcept {
    // Store explicit pose fields and mark the builder as pose-driven rather
    // than operator-driven.
    _translation       = translation_;
    _orientation       = orientation_;
    _has_sync_operator = false;
    return *this;
}

template <typename T>
typename Sync<T>::Builder&
Sync<T>::Builder::with_sync_operator(const atlas::system::SyncOperator<T>& op) noexcept {
    // Preserve a full operator when the caller already built one elsewhere.
    _op_storage        = op;
    _has_sync_operator = true;
    return *this;
}

template <typename T>
void
Sync<T>::Builder::validate() const {
    // Validation is performed against whichever source of truth the builder is
    // currently configured to use:
    // - explicit translation/orientation fields
    // - or a stored SyncOperator
    const Vector3<T>& translation = _has_sync_operator ? _op_storage.translation : _translation;

    const Quaternion<T>& orientation = _has_sync_operator ? _op_storage.orientation : _orientation;

    // Rigid transforms with NaN or infinity coordinates are not meaningful and
    // would poison every downstream matrix/vector operation.
    if (!std::isfinite(translation.x)
        || !std::isfinite(translation.y)
        || !std::isfinite(translation.z)) {

        atlas::logger::error()
            << "Sync::Builder: translation contains non-finite values.";
        throw std::runtime_error(
            "Sync::Builder: translation contains non-finite values.");
    }

    // Quaternion components must also be finite for conversion to rotation
    // matrices to remain numerically meaningful.
    if (!std::isfinite(orientation.w)
        || !std::isfinite(orientation.x)
        || !std::isfinite(orientation.y)
        || !std::isfinite(orientation.z)) {

        atlas::logger::error()
            << "Sync::Builder: orientation contains non-finite values.";
        throw std::runtime_error(
            "Sync::Builder: orientation contains non-finite values.");
    }

    // The zero quaternion has no valid rotational interpretation because it
    // cannot be normalized and does not encode an axis/angle or unit rotation.
    if (orientation.w == T(0)
        && orientation.x == T(0)
        && orientation.y == T(0)
        && orientation.z == T(0)) {

        atlas::logger::error()
            << "Sync::Builder: zero quaternion is invalid.";
        throw std::runtime_error(
            "Sync::Builder: zero quaternion is invalid.");
    }

    // Compute quaternion norm:
    //   ||q|| = sqrt(w^2 + x^2 + y^2 + z^2)
    //
    // A unit quaternion is preferred because the derived rotation matrix is
    // then orthonormal and satisfies R^{-1} = R^T.
    const T norm = std::sqrt(
        orientation.w * orientation.w
        + orientation.x * orientation.x
        + orientation.y * orientation.y
        + orientation.z * orientation.z);

    // Non-unit quaternions are tolerated here for flexibility, but the caller
    // is warned because exact rigid-rotation behavior is no longer guaranteed.
    if (std::abs(norm - T(1)) > T(1e-3)) {
        atlas::logger::warn()
            << "Sync::Builder: quaternion not normalized.";
    }
}

template <typename T>
Sync<T>
Sync<T>::Builder::build() const {
    // Validate before materializing the object so all construction paths obey
    // the same rules.
    validate();

    Sync<T> s {};

    if (_has_sync_operator) {
        // Use the stored operator verbatim when that path was selected.
        s.sync_operator = _op_storage;
    } else {
        // Otherwise synthesize a fresh operator from the explicit pose fields.
        s.sync_operator = atlas::system::SyncOperator<T>(_translation, _orientation);
    }

    // Refresh matrix caches unconditionally so the returned object is
    // self-consistent regardless of how it was built.
    s.rebuild_matrices();
    return s;
}

template <typename T>
atlas::host_shared_ptr<Sync<T>>
Sync<T>::Builder::make_host_shared() const {
    // Build first, then move into host-shared storage to preserve a single
    // validated construction path.
    auto s = build();
    return atlas::make_host_shared<Sync<T>>(std::move(s));
}

} // namespace atlas::system
