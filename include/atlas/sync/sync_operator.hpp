#pragma once

namespace atlas::system {

template <typename T>
constexpr SyncOperator<T>::SyncOperator() noexcept
    : translation(T(0), T(0), T(0))
    , orientation()
    , orientation_matrix(atlas::math::identity3x3<T>())
    , inverse_orientation_matrix(atlas::math::identity3x3<T>()) {
    // Default-construct an identity rigid transform.
    //
    // Initial state:
    // - translation                = (0, 0, 0)
    // - orientation                = default quaternion
    // - orientation_matrix         = identity rotation
    // - inverse_orientation_matrix = identity rotation
    //
    // This represents a transform that leaves points, directions, and rays unchanged.
}

template <typename T>
SyncOperator<T>::SyncOperator(const Vector3<T>& translation_,
                              const Quaternion<T>& orientation_) noexcept
    : translation(translation_)
    , orientation(orientation_)
    , orientation_matrix()
    , inverse_orientation_matrix() {
    // Construct a rigid transform from an explicit translation and orientation.
    //
    // Stored inputs:
    // - translation_ : translation component applied after rotation
    // - orientation_ : quaternion describing the local-to-world rotation
    //
    // Cached matrices are rebuilt immediately so all transform queries can use
    // matrix-based evaluation without recomputing from the quaternion every time.
    rebuild_matrices();
}

template <typename T>
void
SyncOperator<T>::rebuild_matrices() noexcept {
    // Recompute the cached rotation matrices from the current quaternion.
    //
    // orientation_matrix:
    // - used for local -> world rotation
    //
    // inverse_orientation_matrix:
    // - used for world -> local rotation
    //
    // Assumption:
    // - orientation represents a rigid rotation, so the inverse rotation matrix
    //   is the transpose of the forward rotation matrix.
    orientation_matrix         = orientation.to_matrix3x3();
    inverse_orientation_matrix = math::transpose(orientation_matrix);
}

template <typename T>
void
SyncOperator<T>::sync_to_world(const Vector3<T>& local_point,
                               Vector3<T>& world_point) const noexcept {
    // Transform a point from local space into world space.
    //
    // Point transform rule:
    //   world_point = R * local_point + t
    //
    // where:
    // - R = orientation_matrix
    // - t = translation
    //
    // Translation is applied because points represent positions in space.
    world_point = (orientation_matrix * local_point) + translation;
}

template <typename T>
void
SyncOperator<T>::sync_to_local(const Vector3<T>& world_point,
                               Vector3<T>& local_point) const noexcept {
    // Transform a point from world space into local space.
    //
    // Inverse point transform rule:
    //   local_point = R^{-1} * (world_point - t)
    //
    // Steps:
    // 1. remove translation
    // 2. apply inverse rotation
    local_point = inverse_orientation_matrix * (world_point - translation);
}

template <typename T>
void
SyncOperator<T>::sync_dir_to_world(const Vector3<T>& local_dir,
                                   Vector3<T>& world_dir) const noexcept {
    // Transform a direction from local space into world space.
    //
    // Direction transform rule:
    //   world_dir = R * local_dir
    //
    // Translation is intentionally ignored because directions encode orientation
    // and magnitude, not position.
    world_dir = orientation_matrix * local_dir;
}

template <typename T>
void
SyncOperator<T>::sync_dir_to_local(const Vector3<T>& world_dir,
                                   Vector3<T>& local_dir) const noexcept {
    // Transform a direction from world space into local space.
    //
    // Inverse direction transform rule:
    //   local_dir = R^{-1} * world_dir
    //
    // Translation is intentionally ignored.
    local_dir = inverse_orientation_matrix * world_dir;
}

template <typename T>
void
SyncOperator<T>::sync_to_world(const atlas::spatial::Ray<T>& local_ray,
                               atlas::spatial::Ray<T>& world_ray) const noexcept {
    // Transform a ray from local space into world space.
    //
    // Ray components are transformed differently:
    // - origin    : treated as a point  -> R * origin + t
    // - direction : treated as a vector -> R * direction
    //
    // This preserves the geometric meaning of the ray under the rigid transform.
    world_ray.origin    = (orientation_matrix * local_ray.origin) + translation;
    world_ray.direction = orientation_matrix * local_ray.direction;
}

template <typename T>
void
SyncOperator<T>::sync_to_local(const atlas::spatial::Ray<T>& world_ray,
                               atlas::spatial::Ray<T>& local_ray) const noexcept {
    // Transform a ray from world space into local space.
    //
    // Inverse ray transform:
    // - origin    : treated as a point  -> R^{-1} * (origin - t)
    // - direction : treated as a vector -> R^{-1} * direction
    local_ray.origin    = inverse_orientation_matrix * (world_ray.origin - translation);
    local_ray.direction = inverse_orientation_matrix * world_ray.direction;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_to_world(const Vector3<T>& local_point) const noexcept {
    // Return-by-value wrapper for local-point -> world-point transformation.
    //
    // This overload is convenient when the caller does not want to manage an
    // explicit output parameter.
    Vector3<T> out;
    sync_to_world(local_point, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_to_local(const Vector3<T>& world_point) const noexcept {
    // Return-by-value wrapper for world-point -> local-point transformation.
    Vector3<T> out;
    sync_to_local(world_point, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_dir_to_world(const Vector3<T>& local_dir) const noexcept {
    // Return-by-value wrapper for local-direction -> world-direction transformation.
    Vector3<T> out;
    sync_dir_to_world(local_dir, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_dir_to_local(const Vector3<T>& world_dir) const noexcept {
    // Return-by-value wrapper for world-direction -> local-direction transformation.
    Vector3<T> out;
    sync_dir_to_local(world_dir, out);
    return out;
}

template <typename T>
atlas::spatial::Ray<T>
SyncOperator<T>::sync_to_world(const atlas::spatial::Ray<T>& local_ray) const noexcept {
    // Return-by-value wrapper for local-ray -> world-ray transformation.
    atlas::spatial::Ray<T> out;
    sync_to_world(local_ray, out);
    return out;
}

template <typename T>
atlas::spatial::Ray<T>
SyncOperator<T>::sync_to_local(const atlas::spatial::Ray<T>& world_ray) const noexcept {
    // Return-by-value wrapper for world-ray -> local-ray transformation.
    atlas::spatial::Ray<T> out;
    sync_to_local(world_ray, out);
    return out;
}

} // namespace atlas::system