#pragma once

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

namespace atlas::system {

/**
 * @brief Rigid local/world coordinate conversion operator.
 *
 * @tparam T Scalar type used by the stored vectors, quaternion, and matrices.
 *
 * `SyncOperator` stores a rigid pose made of:
 * - a translation from local space into world space
 * - an orientation that rotates local axes into world axes
 * - cached 3x3 rotation matrices derived from the quaternion
 *
 * The object is intended for repeated coordinate conversion where the pose is
 * stable for some period of time. Instead of rebuilding a rotation matrix for
 * every transform call, the quaternion is converted once and the resulting
 * matrix pair is reused.
 *
 * The implemented rigid transforms are:
 * - point, local to world: `p_w = R * p_l + t`
 * - point, world to local: `p_l = R^-1 * (p_w - t)`
 * - direction, local to world: `d_w = R * d_l`
 * - direction, world to local: `d_l = R^-1 * d_w`
 *
 * Rays are transformed component-wise:
 * - the ray origin behaves like a point
 * - the ray direction behaves like a direction
 *
 * @note `inverse_orientation_matrix` is computed as `transpose(R)`, not by a
 * general matrix inverse. This is correct only when `orientation` represents a
 * unit quaternion, because only then is `R` orthonormal.
 *
 * @warning If `orientation` is changed after construction, the cached matrices
 * are not updated automatically. Call rebuild_matrices() before performing any
 * transform with the modified quaternion.
 *
 * @warning If `orientation` is not normalized, round-trip transforms and length
 * preservation for direction vectors are not guaranteed.
 */
template <typename T>
struct SyncOperator final {

    /** @brief Translation that maps the local origin into world space. */
    Vector3<T> translation {};

    /**
     * @brief Quaternion that rotates local coordinates into world coordinates.
     *
     * This quaternion is expected to be normalized for the cached inverse matrix
     * to represent the true inverse rotation.
     */
    Quaternion<T> orientation {};

    /**
     * @brief Cached rotation matrix corresponding to @ref orientation.
     *
     * This matrix is used by all local-to-world rotation operations.
     */
    Matrix<T, 3, 3> orientation_matrix {};

    /**
     * @brief Cached inverse rotation matrix corresponding to @ref orientation.
     *
     * The matrix is rebuilt as `transpose(orientation_matrix)`, which is valid
     * for orthonormal rotation matrices produced by unit quaternions.
     */
    Matrix<T, 3, 3> inverse_orientation_matrix {};

    /**
     * @brief Constructs the identity rigid transform.
     *
     * The resulting operator performs no change:
     * - translation is `(0, 0, 0)`
     * - orientation is the identity quaternion
     * - both cached matrices are 3x3 identity matrices
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr SyncOperator() noexcept;

    /**
     * @brief Constructs a rigid transform from translation and orientation.
     *
     * @param translation_ Translation applied after rotation for point
     * conversion from local space to world space.
     * @param orientation_ Quaternion that rotates local coordinates into world
     * coordinates.
     *
     * The constructor stores both values and immediately rebuilds the cached
     * rotation matrices.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    SyncOperator(const Vector3<T>& translation_,
                 const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Recomputes the cached rotation matrices from @ref orientation.
     *
     * Call this after any direct modification of @ref orientation. The function
     * updates:
     * - @ref orientation_matrix with `orientation.to_matrix3x3()`
     * - @ref inverse_orientation_matrix with the transpose of that matrix
     *
     * The result is correct as an inverse rotation when @ref orientation is a
     * unit quaternion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept;

    /**
     * @brief Converts a point from local space to world space.
     *
     * @param local_point Input point expressed in local coordinates.
     * @param world_point Output point expressed in world coordinates.
     *
     * This applies rotation followed by translation:
     * `world_point = R * local_point + t`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Vector3<T>& local_point,
                  Vector3<T>& world_point) const noexcept;

    /**
     * @brief Converts a point from world space to local space.
     *
     * @param world_point Input point expressed in world coordinates.
     * @param local_point Output point expressed in local coordinates.
     *
     * This removes translation first and then applies the inverse rotation:
     * `local_point = R^-1 * (world_point - t)`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Vector3<T>& world_point,
                  Vector3<T>& local_point) const noexcept;

    /**
     * @brief Converts a direction vector from local space to world space.
     *
     * @param local_dir Input direction expressed in local coordinates.
     * @param world_dir Output direction expressed in world coordinates.
     *
     * Translation is intentionally ignored because directions describe
     * orientation and magnitude, not position.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_world(const Vector3<T>& local_dir,
                      Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Converts a direction vector from world space to local space.
     *
     * @param world_dir Input direction expressed in world coordinates.
     * @param local_dir Output direction expressed in local coordinates.
     *
     * Only the inverse rotation is applied; translation is ignored.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_local(const Vector3<T>& world_dir,
                      Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Converts a ray from local space to world space.
     *
     * @param local_ray Input ray expressed in local coordinates.
     * @param world_ray Output ray expressed in world coordinates.
     *
     * The ray origin is transformed as a point, while the ray direction is
     * transformed as a direction.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const atlas::spatial::Ray<T>& local_ray,
                  atlas::spatial::Ray<T>& world_ray) const noexcept;

    /**
     * @brief Converts a ray from world space to local space.
     *
     * @param world_ray Input ray expressed in world coordinates.
     * @param local_ray Output ray expressed in local coordinates.
     *
     * The ray origin is transformed as a point, while the ray direction is
     * transformed as a direction.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const atlas::spatial::Ray<T>& world_ray,
                  atlas::spatial::Ray<T>& local_ray) const noexcept;

    /**
     * @brief Returns the local-to-world point transform result by value.
     *
     * @param local_point Input point expressed in local coordinates.
     * @return Point expressed in world coordinates.
     *
     * This is a convenience overload equivalent to calling the output-parameter
     * version and returning the produced point.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_world(const Vector3<T>& local_point) const noexcept;

    /**
     * @brief Returns the world-to-local point transform result by value.
     *
     * @param world_point Input point expressed in world coordinates.
     * @return Point expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_local(const Vector3<T>& world_point) const noexcept;

    /**
     * @brief Returns the local-to-world direction transform result by value.
     *
     * @param local_dir Input direction expressed in local coordinates.
     * @return Direction expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_world(const Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Returns the world-to-local direction transform result by value.
     *
     * @param world_dir Input direction expressed in world coordinates.
     * @return Direction expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_local(const Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Returns the local-to-world ray transform result by value.
     *
     * @param local_ray Input ray expressed in local coordinates.
     * @return Ray expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::spatial::Ray<T>
    sync_to_world(const atlas::spatial::Ray<T>& local_ray) const noexcept;

    /**
     * @brief Returns the world-to-local ray transform result by value.
     *
     * @param world_ray Input ray expressed in world coordinates.
     * @return Ray expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::spatial::Ray<T>
    sync_to_local(const atlas::spatial::Ray<T>& world_ray) const noexcept;
};

} // namespace atlas::system

namespace atlas {

/** @brief Public namespace alias for @ref atlas::system::SyncOperator. */
template <typename T>
using SyncOperator = system::SyncOperator<T>;

} // namespace atlas

#include <atlas/sync/sync_operator.hpp>
