#pragma once

namespace atlas {

template <typename T>
constexpr SyncOperator<T>::SyncOperator() noexcept
    : translation(T(0), T(0), T(0))
    , orientation()
    , orientation_matrix(atlas::identity3x3<T>())
    , inverse_orientation_matrix(atlas::identity3x3<T>()) {
}

template <typename T>
SyncOperator<T>::SyncOperator(const Vector3<T>& translation_,
                              const Quaternion<T>& orientation_) noexcept
    : translation(translation_)
    , orientation(orientation_)
    , orientation_matrix()
    , inverse_orientation_matrix() {

    rebuild_matrices();
}

template <typename T>
void
SyncOperator<T>::rebuild_matrices() noexcept {

    orientation_matrix = orientation.to_matrix3x3();

    inverse_orientation_matrix = transpose(orientation_matrix);
}

template <typename T>
void
SyncOperator<T>::sync_to_world(const Vector3<T>& local_point,
                               Vector3<T>& world_point) const noexcept {
    atlas::rotate_translate(orientation_matrix, local_point, translation, world_point);
}

template <typename T>
void
SyncOperator<T>::sync_to_local(const Vector3<T>& world_point,
                               Vector3<T>& local_point) const noexcept {
    atlas::rotate_subtract(inverse_orientation_matrix, world_point, translation, local_point);
}

template <typename T>
void
SyncOperator<T>::sync_dir_to_world(const Vector3<T>& local_dir,
                                   Vector3<T>& world_dir) const noexcept {
    atlas::rotate(orientation_matrix, local_dir, world_dir);
}

template <typename T>
void
SyncOperator<T>::sync_dir_to_local(const Vector3<T>& world_dir,
                                   Vector3<T>& local_dir) const noexcept {
    atlas::rotate(inverse_orientation_matrix, world_dir, local_dir);
}

template <typename T>
void
SyncOperator<T>::sync_to_world(const atlas::Ray<T>& local_ray,
                               atlas::Ray<T>& world_ray) const noexcept {
    atlas::rotate_translate(orientation_matrix, local_ray.origin, translation, world_ray.origin);
    atlas::rotate(orientation_matrix, local_ray.direction, world_ray.direction);
}

template <typename T>
void
SyncOperator<T>::sync_to_local(const atlas::Ray<T>& world_ray,
                               atlas::Ray<T>& local_ray) const noexcept {
    atlas::rotate_subtract(inverse_orientation_matrix, world_ray.origin, translation, local_ray.origin);
    atlas::rotate(inverse_orientation_matrix, world_ray.direction, local_ray.direction);
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_to_world(const Vector3<T>& local_point) const noexcept {

    Vector3<T> out;
    sync_to_world(local_point, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_to_local(const Vector3<T>& world_point) const noexcept {

    Vector3<T> out;
    sync_to_local(world_point, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_dir_to_world(const Vector3<T>& local_dir) const noexcept {

    Vector3<T> out;
    sync_dir_to_world(local_dir, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_dir_to_local(const Vector3<T>& world_dir) const noexcept {

    Vector3<T> out;
    sync_dir_to_local(world_dir, out);
    return out;
}

template <typename T>
atlas::Ray<T>
SyncOperator<T>::sync_to_world(const atlas::Ray<T>& local_ray) const noexcept {

    atlas::Ray<T> out;
    sync_to_world(local_ray, out);
    return out;
}

template <typename T>
atlas::Ray<T>
SyncOperator<T>::sync_to_local(const atlas::Ray<T>& world_ray) const noexcept {

    atlas::Ray<T> out;
    sync_to_local(world_ray, out);
    return out;
}

}