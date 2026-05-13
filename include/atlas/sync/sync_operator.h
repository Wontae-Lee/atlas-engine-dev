#pragma once

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

namespace atlas::physics {

/**
 * @brief Lightweight rigid-transform operator for converting geometry between
 *        local and world coordinate spaces.
 *
 * SyncOperator stores a rigid-body transform composed of a translation vector
 * and an orientation quaternion. The orientation is also cached as a rotation
 * matrix and an inverse rotation matrix to make repeated coordinate conversions
 * efficient.
 *
 * The transform maps a local-space point @f$\mathbf{x}_{\mathrm{local}}@f$ to a
 * world-space point @f$\mathbf{x}_{\mathrm{world}}@f$ as:
 *
 * @f[
 *     \mathbf{x}_{\mathrm{world}}
 *     =
 *     \mathbf{R}\mathbf{x}_{\mathrm{local}}
 *     +
 *     \mathbf{t},
 * @f]
 *
 * where:
 *
 * - @f$\mathbf{R}@f$ is the rotation matrix derived from @ref orientation,
 * - @f$\mathbf{t}@f$ is @ref translation.
 *
 * The inverse point transform is:
 *
 * @f[
 *     \mathbf{x}_{\mathrm{local}}
 *     =
 *     \mathbf{R}^{-1}
 *     \left(
 *         \mathbf{x}_{\mathrm{world}} - \mathbf{t}
 *     \right).
 * @f]
 *
 * Since @f$\mathbf{R}@f$ is expected to be an orthonormal rotation matrix,
 * the inverse rotation is cached as the transpose:
 *
 * @f[
 *     \mathbf{R}^{-1}
 *     =
 *     \mathbf{R}^{T}.
 * @f]
 *
 * Direction vectors are transformed only by rotation because directions do not
 * have position. Therefore, translation is not applied to direction vectors:
 *
 * @f[
 *     \mathbf{d}_{\mathrm{world}}
 *     =
 *     \mathbf{R}\mathbf{d}_{\mathrm{local}},
 *     \qquad
 *     \mathbf{d}_{\mathrm{local}}
 *     =
 *     \mathbf{R}^{-1}\mathbf{d}_{\mathrm{world}}.
 * @f]
 *
 * Rays are transformed by applying the point transform to the ray origin and the
 * direction transform to the ray direction.
 *
 * @tparam T Floating-point scalar type used for transform computations.
 *
 * @note The cached matrices are not automatically updated when @ref orientation
 *       is modified directly. Call rebuild_matrices() after changing
 *       @ref orientation.
 * @note The inverse matrix is computed as the transpose of the orientation
 *       matrix. This assumes that @ref orientation produces a valid orthonormal
 *       rotation matrix.
 * @note This type is intentionally lightweight and suitable for use as an
 *       operator object in host/device code.
 */
template <typename T>
struct SyncOperator final {

    /**
     * @brief Translation component of the rigid transform.
     *
     * This vector represents the positional offset @f$\mathbf{t}@f$ used in the
     * local-to-world point transform:
     *
     * @f[
     *     \mathbf{x}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{x}_{\mathrm{local}}
     *     +
     *     \mathbf{t}.
     * @f]
     *
     * Translation is applied to points and ray origins. It is not applied to
     * direction vectors.
     */
    Vector3<T> translation {};

    /**
     * @brief Orientation component of the rigid transform as a quaternion.
     *
     * The quaternion represents the rotational part of the transform. The cached
     * matrix @ref orientation_matrix is derived from this quaternion by
     * rebuild_matrices().
     *
     * If the quaternion is denoted by @f$q@f$, then the corresponding rotation
     * matrix is:
     *
     * @f[
     *     \mathbf{R}
     *     =
     *     \mathbf{R}(q).
     * @f]
     *
     * @note Directly modifying this value does not automatically update
     *       @ref orientation_matrix or @ref inverse_orientation_matrix. Call
     *       rebuild_matrices() after changing it.
     */
    Quaternion<T> orientation {};

    /**
     * @brief Cached rotation matrix derived from @ref orientation.
     *
     * This matrix represents the rotational part @f$\mathbf{R}@f$ of the rigid
     * transform. It is used for local-to-world point, direction, and ray
     * transformations.
     *
     * Local-to-world point transform:
     *
     * @f[
     *     \mathbf{x}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{x}_{\mathrm{local}}
     *     +
     *     \mathbf{t}.
     * @f]
     *
     * Local-to-world direction transform:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{d}_{\mathrm{local}}.
     * @f]
     *
     * @note This matrix must be kept consistent with @ref orientation by calling
     *       rebuild_matrices().
     */
    Matrix<T, 3, 3> orientation_matrix {};

    /**
     * @brief Cached inverse rotation matrix derived from @ref orientation.
     *
     * This matrix represents @f$\mathbf{R}^{-1}@f$ and is used for world-to-local
     * point, direction, and ray transformations.
     *
     * For a valid rotation matrix:
     *
     * @f[
     *     \mathbf{R}^{-1}
     *     =
     *     \mathbf{R}^{T}.
     * @f]
     *
     * World-to-local point transform:
     *
     * @f[
     *     \mathbf{x}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}
     *     \left(
     *         \mathbf{x}_{\mathrm{world}} - \mathbf{t}
     *     \right).
     * @f]
     *
     * World-to-local direction transform:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}\mathbf{d}_{\mathrm{world}}.
     * @f]
     */
    Matrix<T, 3, 3> inverse_orientation_matrix {};

    /**
     * @brief Default constructor.
     *
     * Constructs an identity rigid transform.
     *
     * The default transform is:
     *
     * @f[
     *     \mathbf{t}
     *     =
     *     \begin{bmatrix}
     *         0 \\ 0 \\ 0
     *     \end{bmatrix},
     *     \qquad
     *     \mathbf{R}
     *     =
     *     \mathbf{I}.
     * @f]
     *
     * Therefore, local-to-world and world-to-local transformations initially
     * leave points and directions unchanged.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr SyncOperator() noexcept;

    /**
     * @brief Constructs a transform operator from translation and orientation.
     *
     * The given translation and orientation define the rigid transform:
     *
     * @f[
     *     \mathbf{x}_{\mathrm{world}}
     *     =
     *     \mathbf{R}(q)\mathbf{x}_{\mathrm{local}}
     *     +
     *     \mathbf{t},
     * @f]
     *
     * where @f$q@f$ is @p orientation_ and @f$\mathbf{t}@f$ is
     * @p translation_.
     *
     * The cached rotation matrices are rebuilt during construction.
     *
     * @param translation_ Translation component @f$\mathbf{t}@f$.
     * @param orientation_ Orientation quaternion @f$q@f$.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    SyncOperator(const Vector3<T>& translation_,
                 const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Rebuilds cached rotation matrices from the current orientation.
     *
     * This function updates:
     *
     * @f[
     *     \mathbf{R}
     *     =
     *     \mathbf{R}(q),
     *     \qquad
     *     \mathbf{R}^{-1}
     *     =
     *     \mathbf{R}^{T},
     * @f]
     *
     * where @f$q@f$ is @ref orientation.
     *
     * Call this function whenever @ref orientation is changed directly. Without
     * rebuilding, subsequent coordinate transformations may use stale matrices.
     *
     * @note The inverse matrix is computed as the transpose of the rotation
     *       matrix, assuming an orthonormal rotation matrix.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept;

    /**
     * @brief Transforms a local-space point into world space.
     *
     * This function applies the full rigid transform to a point:
     *
     * @f[
     *     \mathbf{x}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{x}_{\mathrm{local}}
     *     +
     *     \mathbf{t}.
     * @f]
     *
     * Points are affected by both rotation and translation.
     *
     * @param local_point Input point @f$\mathbf{x}_{\mathrm{local}}@f$.
     * @param world_point Output point @f$\mathbf{x}_{\mathrm{world}}@f$.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Vector3<T>& local_point,
                  Vector3<T>& world_point) const noexcept;

    /**
     * @brief Transforms a world-space point into local space.
     *
     * This function applies the inverse rigid transform to a point:
     *
     * @f[
     *     \mathbf{x}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}
     *     \left(
     *         \mathbf{x}_{\mathrm{world}} - \mathbf{t}
     *     \right).
     * @f]
     *
     * The translation is removed first, and then the inverse rotation is applied.
     *
     * @param world_point Input point @f$\mathbf{x}_{\mathrm{world}}@f$.
     * @param local_point Output point @f$\mathbf{x}_{\mathrm{local}}@f$.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Vector3<T>& world_point,
                  Vector3<T>& local_point) const noexcept;

    /**
     * @brief Transforms a local-space direction into world space.
     *
     * This function applies only the rotational part of the transform:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{d}_{\mathrm{local}}.
     * @f]
     *
     * Translation is not applied because a direction vector represents orientation
     * or displacement, not a point location.
     *
     * @param local_dir Input direction @f$\mathbf{d}_{\mathrm{local}}@f$.
     * @param world_dir Output direction @f$\mathbf{d}_{\mathrm{world}}@f$.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_world(const Vector3<T>& local_dir,
                      Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Transforms a world-space direction into local space.
     *
     * This function applies only the inverse rotational part of the transform:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}\mathbf{d}_{\mathrm{world}}.
     * @f]
     *
     * Translation is not applied because direction vectors do not represent
     * absolute positions.
     *
     * @param world_dir Input direction @f$\mathbf{d}_{\mathrm{world}}@f$.
     * @param local_dir Output direction @f$\mathbf{d}_{\mathrm{local}}@f$.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_local(const Vector3<T>& world_dir,
                      Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Transforms a local-space ray into world space.
     *
     * A ray is treated as:
     *
     * @f[
     *     r(s)
     *     =
     *     \mathbf{o}
     *     +
     *     s\mathbf{d},
     * @f]
     *
     * where @f$\mathbf{o}@f$ is the ray origin and @f$\mathbf{d}@f$ is the ray
     * direction.
     *
     * The origin is transformed as a point:
     *
     * @f[
     *     \mathbf{o}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{o}_{\mathrm{local}}
     *     +
     *     \mathbf{t},
     * @f]
     *
     * while the direction is transformed as a direction vector:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{d}_{\mathrm{local}}.
     * @f]
     *
     * @param local_ray Input ray in local coordinates.
     * @param world_ray Output ray in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const atlas::spatial::Ray<T>& local_ray,
                  atlas::spatial::Ray<T>& world_ray) const noexcept;

    /**
     * @brief Transforms a world-space ray into local space.
     *
     * A ray is treated as:
     *
     * @f[
     *     r(s)
     *     =
     *     \mathbf{o}
     *     +
     *     s\mathbf{d}.
     * @f]
     *
     * The origin is transformed with the inverse point transform:
     *
     * @f[
     *     \mathbf{o}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}
     *     \left(
     *         \mathbf{o}_{\mathrm{world}} - \mathbf{t}
     *     \right),
     * @f]
     *
     * while the direction is transformed with the inverse direction transform:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}\mathbf{d}_{\mathrm{world}}.
     * @f]
     *
     * @param world_ray Input ray in world coordinates.
     * @param local_ray Output ray in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const atlas::spatial::Ray<T>& world_ray,
                  atlas::spatial::Ray<T>& local_ray) const noexcept;

    /**
     * @brief Transforms a local-space point into world space and returns the result.
     *
     * This overload returns the transformed point by value. It is equivalent to:
     *
     * @code
     * Vector3<T> out;
     * sync_to_world(local_point, out);
     * return out;
     * @endcode
     *
     * Mathematically:
     *
     * @f[
     *     \mathbf{x}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{x}_{\mathrm{local}}
     *     +
     *     \mathbf{t}.
     * @f]
     *
     * @param local_point Input point @f$\mathbf{x}_{\mathrm{local}}@f$.
     * @return Point @f$\mathbf{x}_{\mathrm{world}}@f$.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_world(const Vector3<T>& local_point) const noexcept;

    /**
     * @brief Transforms a world-space point into local space and returns the result.
     *
     * This overload returns the transformed point by value. It is equivalent to:
     *
     * @code
     * Vector3<T> out;
     * sync_to_local(world_point, out);
     * return out;
     * @endcode
     *
     * Mathematically:
     *
     * @f[
     *     \mathbf{x}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}
     *     \left(
     *         \mathbf{x}_{\mathrm{world}} - \mathbf{t}
     *     \right).
     * @f]
     *
     * @param world_point Input point @f$\mathbf{x}_{\mathrm{world}}@f$.
     * @return Point @f$\mathbf{x}_{\mathrm{local}}@f$.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_local(const Vector3<T>& world_point) const noexcept;

    /**
     * @brief Transforms a local-space direction into world space and returns the result.
     *
     * This overload returns the transformed direction by value.
     *
     * Mathematically:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{d}_{\mathrm{local}}.
     * @f]
     *
     * @param local_dir Input direction @f$\mathbf{d}_{\mathrm{local}}@f$.
     * @return Direction @f$\mathbf{d}_{\mathrm{world}}@f$.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_world(const Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Transforms a world-space direction into local space and returns the result.
     *
     * This overload returns the transformed direction by value.
     *
     * Mathematically:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}\mathbf{d}_{\mathrm{world}}.
     * @f]
     *
     * @param world_dir Input direction @f$\mathbf{d}_{\mathrm{world}}@f$.
     * @return Direction @f$\mathbf{d}_{\mathrm{local}}@f$.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_local(const Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Transforms a local-space ray into world space and returns the result.
     *
     * This overload returns the transformed ray by value. The origin is transformed
     * as a point and the direction is transformed as a direction:
     *
     * @f[
     *     \mathbf{o}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{o}_{\mathrm{local}}
     *     +
     *     \mathbf{t},
     *     \qquad
     *     \mathbf{d}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{d}_{\mathrm{local}}.
     * @f]
     *
     * @param local_ray Input ray in local coordinates.
     * @return Ray in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::spatial::Ray<T>
    sync_to_world(const atlas::spatial::Ray<T>& local_ray) const noexcept;

    /**
     * @brief Transforms a world-space ray into local space and returns the result.
     *
     * This overload returns the transformed ray by value. The origin is transformed
     * as a point and the direction is transformed as a direction:
     *
     * @f[
     *     \mathbf{o}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}
     *     \left(
     *         \mathbf{o}_{\mathrm{world}} - \mathbf{t}
     *     \right),
     *     \qquad
     *     \mathbf{d}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}\mathbf{d}_{\mathrm{world}}.
     * @f]
     *
     * @param world_ray Input ray in world coordinates.
     * @return Ray in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::spatial::Ray<T>
    sync_to_local(const atlas::spatial::Ray<T>& world_ray) const noexcept;
};

} // namespace atlas::physics

namespace atlas {

/**
 * @brief Alias for atlas::physics::SyncOperator.
 *
 * This alias exposes the synchronization transform operator directly in the
 * atlas namespace.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SyncOperator = physics::SyncOperator<T>;

} // namespace atlas

#include <atlas/sync/sync_operator.hpp>