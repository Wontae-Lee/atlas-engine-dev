#pragma once

namespace atlas::system {

template <typename T>
constexpr SyncOperator<T>::SyncOperator() noexcept
    // Identity rigid transform:
    // - translation = 0
    // - rotation    = I
    //
    // Therefore for any local point p and direction d:
    //   p_world = p
    //   d_world = d
    : translation(T(0), T(0), T(0))
    , orientation()
    , orientation_matrix(atlas::math::identity3x3<T>())
    , inverse_orientation_matrix(atlas::math::identity3x3<T>()) {
}

template <typename T>
SyncOperator<T>::SyncOperator(const Vector3<T>& translation_,
                              const Quaternion<T>& orientation_) noexcept
    // Store the pose parameters first; matrix caches are rebuilt below.
    : translation(translation_)
    , orientation(orientation_)
    , orientation_matrix()
    , inverse_orientation_matrix() {

    // Convert quaternion form into matrix form once so repeated transforms can
    // use matrix-vector multiplication directly. This is a classic cache trade:
    // spend one quaternion-to-matrix conversion up front to avoid paying that
    // cost on every point/direction transform.
    rebuild_matrices();
}

template <typename T>
void
SyncOperator<T>::rebuild_matrices() noexcept {
    // Recompute cached rotation matrices from the quaternion.
    //
    // Let q be the stored quaternion and R(q) its 3x3 rotation matrix.
    // Then local/world conversion uses:
    //   p_world = R p_local + t
    //   p_local = R^{-1}(p_world - t)
    //
    // For a unit quaternion:
    // - orientation_matrix is orthonormal
    // - inverse_orientation_matrix can be obtained as transpose(R)
    //
    // Why transpose works:
    // for any orthonormal matrix R,
    //   R^T R = I
    // so
    //   R^{-1} = R^T
    //
    // This is cheaper and numerically cleaner than computing a full generic
    // matrix inverse. The assumption breaks if the quaternion is not normalized.
    orientation_matrix         = orientation.to_matrix3x3();
    inverse_orientation_matrix = math::transpose(orientation_matrix);
}

template <typename T>
void
SyncOperator<T>::sync_to_world(const Vector3<T>& local_point,
                               Vector3<T>& world_point) const noexcept {
    // Point transform (local -> world):
    //   p_w = R p_l + t
    //
    // This is an affine transform:
    // - R rotates the point from local basis into world basis
    // - t shifts the rotated point into its world-space origin
    //
    // In homogeneous coordinates this corresponds to:
    //   [R t] [p_l] = [R p_l + t]
    //   [0 1] [ 1 ]   [     1     ]
    world_point = (orientation_matrix * local_point) + translation;
}

template <typename T>
void
SyncOperator<T>::sync_to_local(const Vector3<T>& world_point,
                               Vector3<T>& local_point) const noexcept {
    // Point transform (world -> local):
    //   p_l = R^{-1} (p_w - t)
    //
    // The order matters:
    // 1. subtract translation to move the world point into the rotated local
    //    frame's origin
    // 2. apply the inverse rotation to express the result in local axes
    local_point = inverse_orientation_matrix * (world_point - translation);
}

template <typename T>
void
SyncOperator<T>::sync_dir_to_world(const Vector3<T>& local_dir,
                                   Vector3<T>& world_dir) const noexcept {
    // Direction transform (local -> world):
    // - rotate only
    // - no translation
    //
    // A direction is a free vector, not a point attached to an origin.
    // Translating a direction would be meaningless because it would change
    // position, not orientation. Only the basis change induced by R applies.
    world_dir = orientation_matrix * local_dir;
}

template <typename T>
void
SyncOperator<T>::sync_dir_to_local(const Vector3<T>& world_dir,
                                   Vector3<T>& local_dir) const noexcept {
    // Direction transform (world -> local):
    // - apply inverse rotation only
    //
    // If R maps local axes into world axes, then R^{-1} maps world-axis vector
    // components back into local-axis components.
    local_dir = inverse_orientation_matrix * world_dir;
}

template <typename T>
void
SyncOperator<T>::sync_to_world(const atlas::spatial::Ray<T>& local_ray,
                               atlas::spatial::Ray<T>& world_ray) const noexcept {
    // Ray transform (local -> world):
    // - origin behaves as a point
    // - direction behaves as a direction
    //
    // If the local ray is
    //   r_l(lambda) = o_l + lambda d_l
    // then the world ray becomes
    //   r_w(lambda) = (R o_l + t) + lambda (R d_l)
    //               = R(o_l + lambda d_l) + t
    //
    // So the set of points traced by the ray is transformed by the same rigid
    // body motion, while preserving the ray parameter lambda.
    world_ray.origin    = (orientation_matrix * local_ray.origin) + translation;
    world_ray.direction = orientation_matrix * local_ray.direction;
}

template <typename T>
void
SyncOperator<T>::sync_to_local(const atlas::spatial::Ray<T>& world_ray,
                               atlas::spatial::Ray<T>& local_ray) const noexcept {
    // Ray transform (world -> local):
    // - origin behaves as a point
    // - direction behaves as a direction
    //
    // Inverse derivation:
    //   o_l = R^{-1}(o_w - t)
    //   d_l = R^{-1} d_w
    //
    // Again, translation is removed only from the origin term, never from the
    // direction term.
    local_ray.origin    = inverse_orientation_matrix * (world_ray.origin - translation);
    local_ray.direction = inverse_orientation_matrix * world_ray.direction;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_to_world(const Vector3<T>& local_point) const noexcept {
    // Convenience wrapper: same mathematics as the output-parameter overload,
    // but returning by value is often easier to compose in expressions.
    Vector3<T> out;
    sync_to_world(local_point, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_to_local(const Vector3<T>& world_point) const noexcept {
    // Returning by value delegates to the canonical implementation so there is
    // only one place where the actual transform formula lives.
    Vector3<T> out;
    sync_to_local(world_point, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_dir_to_world(const Vector3<T>& local_dir) const noexcept {
    // Direction-valued convenience overload.
    Vector3<T> out;
    sync_dir_to_world(local_dir, out);
    return out;
}

template <typename T>
Vector3<T>
SyncOperator<T>::sync_dir_to_local(const Vector3<T>& world_dir) const noexcept {
    // Direction-valued convenience overload.
    Vector3<T> out;
    sync_dir_to_local(world_dir, out);
    return out;
}

template <typename T>
atlas::spatial::Ray<T>
SyncOperator<T>::sync_to_world(const atlas::spatial::Ray<T>& local_ray) const noexcept {
    // Ray-valued convenience overload.
    atlas::spatial::Ray<T> out;
    sync_to_world(local_ray, out);
    return out;
}

template <typename T>
atlas::spatial::Ray<T>
SyncOperator<T>::sync_to_local(const atlas::spatial::Ray<T>& world_ray) const noexcept {
    // Ray-valued convenience overload.
    atlas::spatial::Ray<T> out;
    sync_to_local(world_ray, out);
    return out;
}

} // namespace atlas::system
