#pragma once

namespace atlas::physics {

template <typename T>
constexpr SyncOperator<T>::SyncOperator() noexcept
    : translation(T(0), T(0), T(0))
    , orientation()
    , orientation_matrix(atlas::math::identity3x3<T>())
    , inverse_orientation_matrix(atlas::math::identity3x3<T>()) {
}

template <typename T>
SyncOperator<T>::SyncOperator(const Vector3<T>& translation_,
                              const Quaternion<T>& orientation_) noexcept
    : translation(translation_)
    , orientation(orientation_)
    , orientation_matrix()
    , inverse_orientation_matrix() {
    // Build matrix representations from the initial orientation.
    rebuild_matrices();
}

template <typename T>
void
SyncOperator<T>::rebuild_matrices() noexcept {
    // Convert the orientation quaternion to a rotation matrix.
    orientation_matrix = orientation.to_matrix3x3();

    // For an orthonormal rotation matrix, the inverse is its transpose.
    inverse_orientation_matrix = math::transpose(orientation_matrix);
}

template <typename T>
void
SyncOperator<T>::sync_to_world(const Vector3<T>& local_point,
                               Vector3<T>& world_point) const noexcept {
    // Point transform: x_world = R * x_local + t.
    world_point = (orientation_matrix * local_point) + translation;
}

template <typename T>
void
SyncOperator<T>::sync_to_local(const Vector3<T>& world_point,
                               Vector3<T>& local_point) const noexcept {
    // Inverse point transform: x_local = R^T * (x_world - t).
    local_point = inverse_orientation_matrix * (world_point - translation);
}

template <typename T>
void
SyncOperator<T>::sync_dir_to_world(const Vector3<T>& local_dir,
                                   Vector3<T>& world_dir) const noexcept {
    // Direction transform uses rotation only, without translation.
    world_dir = orientation_matrix * local_dir;
}

template <typename T>
void
SyncOperator<T>::sync_dir_to_local(const Vector3<T>& world_dir,
                                   Vector3<T>& local_dir) const noexcept {
    // Inverse direction transform also uses rotation only.
    local_dir = inverse_orientation_matrix * world_dir;
}

template <typename T>
void
SyncOperator<T>::sync_to_world(const atlas::spatial::Ray<T>& local_ray,
                               atlas::spatial::Ray<T>& world_ray) const noexcept {
    // Transform ray origin as a point and ray direction as a direction.
    world_ray.origin    = (orientation_matrix * local_ray.origin) + translation;
    world_ray.direction = orientation_matrix * local_ray.direction;
}

template <typename T>
void
SyncOperator<T>::sync_to_local(const atlas::spatial::Ray<T>& world_ray,
                               atlas::spatial::Ray<T>& local_ray) const noexcept {
    // Apply inverse point transform to origin and inverse direction transform to direction.
    local_ray.origin    = inverse_orientation_matrix * (world_ray.origin - translation);
    local_ray.direction = inverse_orientation_matrix * world_ray.direction;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_to_world(const Vector3<T>& local_point) const noexcept {
    // Return-value overload for local-to-world point conversion.
    Vector3<T> out;
    sync_to_world(local_point, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_to_local(const Vector3<T>& world_point) const noexcept {
    // Return-value overload for world-to-local point conversion.
    Vector3<T> out;
    sync_to_local(world_point, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_dir_to_world(const Vector3<T>& local_dir) const noexcept {
    // Return-value overload for local-to-world direction conversion.
    Vector3<T> out;
    sync_dir_to_world(local_dir, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_dir_to_local(const Vector3<T>& world_dir) const noexcept {
    // Return-value overload for world-to-local direction conversion.
    Vector3<T> out;
    sync_dir_to_local(world_dir, out);
    return out;
}

template <typename T>
atlas::spatial::Ray<T>
SyncOperator<T>::sync_to_world(const atlas::spatial::Ray<T>& local_ray) const noexcept {
    // Return-value overload for local-to-world ray conversion.
    atlas::spatial::Ray<T> out;
    sync_to_world(local_ray, out);
    return out;
}

template <typename T>
atlas::spatial::Ray<T>
SyncOperator<T>::sync_to_local(const atlas::spatial::Ray<T>& world_ray) const noexcept {
    // Return-value overload for world-to-local ray conversion.
    atlas::spatial::Ray<T> out;
    sync_to_local(world_ray, out);
    return out;
}

} // namespace atlas::physics