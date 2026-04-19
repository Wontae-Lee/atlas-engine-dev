#pragma once

namespace atlas::physics {

template <typename T>
constexpr SyncOperator<T>::SyncOperator() noexcept
    : translation(T(0), T(0), T(0))
    , orientation()
    , orientation_matrix(atlas::math::identity3x3<T>())
    , inverse_orientation_matrix(atlas::math::identity3x3<T>()) {
    // Initialize the operator as an identity rigid transform.
}

template <typename T>
SyncOperator<T>::SyncOperator(const Vector3<T>& translation_,
                              const Quaternion<T>& orientation_) noexcept
    : translation(translation_)
    , orientation(orientation_)
    , orientation_matrix()
    , inverse_orientation_matrix() {

    // Rebuild cached rotation matrices from the provided orientation.
    rebuild_matrices();
}

template <typename T>
void
SyncOperator<T>::rebuild_matrices() noexcept {

    // Build the forward rotation matrix from the current quaternion.
    orientation_matrix = orientation.to_matrix3x3();

    // For a rigid rotation, the inverse matrix is the transpose.
    inverse_orientation_matrix = math::transpose(orientation_matrix);
}

template <typename T>
void
SyncOperator<T>::sync_to_world(const Vector3<T>& local_point,
                               Vector3<T>& world_point) const noexcept {

    // Apply rotation first, then translation, to move the point into world space.
    world_point = (orientation_matrix * local_point) + translation;
}

template <typename T>
void
SyncOperator<T>::sync_to_local(const Vector3<T>& world_point,
                               Vector3<T>& local_point) const noexcept {

    // Undo translation first, then apply the inverse rotation.
    local_point = inverse_orientation_matrix * (world_point - translation);
}

template <typename T>
void
SyncOperator<T>::sync_dir_to_world(const Vector3<T>& local_dir,
                                   Vector3<T>& world_dir) const noexcept {

    // Directions are rotated only; translation must not be applied.
    world_dir = orientation_matrix * local_dir;
}

template <typename T>
void
SyncOperator<T>::sync_dir_to_local(const Vector3<T>& world_dir,
                                   Vector3<T>& local_dir) const noexcept {

    // Convert a world-space direction into local space using the inverse rotation.
    local_dir = inverse_orientation_matrix * world_dir;
}

template <typename T>
void
SyncOperator<T>::sync_to_world(const atlas::spatial::Ray<T>& local_ray,
                               atlas::spatial::Ray<T>& world_ray) const noexcept {

    // Transform the ray origin as a point.
    world_ray.origin = (orientation_matrix * local_ray.origin) + translation;

    // Transform the ray direction as a direction vector.
    world_ray.direction = orientation_matrix * local_ray.direction;
}

template <typename T>
void
SyncOperator<T>::sync_to_local(const atlas::spatial::Ray<T>& world_ray,
                               atlas::spatial::Ray<T>& local_ray) const noexcept {

    // Transform the ray origin back into local space.
    local_ray.origin = inverse_orientation_matrix * (world_ray.origin - translation);

    // Transform the ray direction back into local space.
    local_ray.direction = inverse_orientation_matrix * world_ray.direction;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_to_world(const Vector3<T>& local_point) const noexcept {

    Vector3<T> out;

    // Reuse the output-parameter overload to avoid duplicating transform logic.
    sync_to_world(local_point, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_to_local(const Vector3<T>& world_point) const noexcept {

    Vector3<T> out;

    // Reuse the output-parameter overload to keep behavior centralized.
    sync_to_local(world_point, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_dir_to_world(const Vector3<T>& local_dir) const noexcept {

    Vector3<T> out;

    // Reuse the output-parameter overload for the actual direction transform.
    sync_dir_to_world(local_dir, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_dir_to_local(const Vector3<T>& world_dir) const noexcept {

    Vector3<T> out;

    // Reuse the output-parameter overload for the inverse direction transform.
    sync_dir_to_local(world_dir, out);
    return out;
}

template <typename T>
atlas::spatial::Ray<T>
SyncOperator<T>::sync_to_world(const atlas::spatial::Ray<T>& local_ray) const noexcept {

    atlas::spatial::Ray<T> out;

    // Reuse the output-parameter overload for the full ray transform.
    sync_to_world(local_ray, out);
    return out;
}

template <typename T>
atlas::spatial::Ray<T>
SyncOperator<T>::sync_to_local(const atlas::spatial::Ray<T>& world_ray) const noexcept {

    atlas::spatial::Ray<T> out;

    // Reuse the output-parameter overload for the inverse ray transform.
    sync_to_local(world_ray, out);
    return out;
}

}