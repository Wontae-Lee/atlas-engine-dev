#pragma once

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

namespace atlas::physics {

/**
 * @brief Lightweight transform operator for converting between local and world spaces.
 *
 * This structure stores a rigid transform expressed as translation and orientation,
 * along with cached rotation matrices for forward and inverse transformations.
 *
 * It provides conversion utilities for:
 * - points
 * - direction vectors
 * - rays
 *
 * @tparam T Floating-point scalar type used for transform computations.
 */
template <typename T>
struct SyncOperator final {

    /**
     * @brief Translation component of the rigid transform in world coordinates.
     */
    Vector3<T> translation {};

    /**
     * @brief Orientation component of the rigid transform as a quaternion.
     */
    Quaternion<T> orientation {};

    /**
     * @brief Cached rotation matrix derived from @ref orientation.
     *
     * This matrix is used for local-to-world direction and point rotation.
     */
    Matrix<T, 3, 3> orientation_matrix {};

    /**
     * @brief Cached inverse rotation matrix derived from @ref orientation.
     *
     * This matrix is used for world-to-local direction and point rotation.
     */
    Matrix<T, 3, 3> inverse_orientation_matrix {};

    /**
     * @brief Default constructor.
     *
     * Constructs a default transform operator with default-initialized members.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr SyncOperator() noexcept;

    /**
     * @brief Constructs a transform operator from translation and orientation.
     *
     * @param translation_ Translation component in world coordinates.
     * @param orientation_ Orientation quaternion.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    SyncOperator(const Vector3<T>& translation_,
                 const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Rebuilds cached rotation matrices from the current orientation.
     *
     * This function must be called whenever the orientation changes and the
     * cached matrices must remain consistent with it.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept;

    /**
     * @brief Transforms a local-space point into world space.
     *
     * This applies both rotation and translation.
     *
     * @param local_point Input point in local coordinates.
     * @param world_point Output point in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Vector3<T>& local_point,
                  Vector3<T>& world_point) const noexcept;

    /**
     * @brief Transforms a world-space point into local space.
     *
     * This applies the inverse rigid transform.
     *
     * @param world_point Input point in world coordinates.
     * @param local_point Output point in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Vector3<T>& world_point,
                  Vector3<T>& local_point) const noexcept;

    /**
     * @brief Transforms a local-space direction into world space.
     *
     * Translation is not applied because directions represent orientation only.
     *
     * @param local_dir Input direction in local coordinates.
     * @param world_dir Output direction in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_world(const Vector3<T>& local_dir,
                      Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Transforms a world-space direction into local space.
     *
     * Translation is not applied because directions represent orientation only.
     *
     * @param world_dir Input direction in world coordinates.
     * @param local_dir Output direction in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_local(const Vector3<T>& world_dir,
                      Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Transforms a local-space ray into world space.
     *
     * The ray origin is transformed as a point and the ray direction is
     * transformed as a direction vector.
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
     * The ray origin is transformed as a point and the ray direction is
     * transformed as a direction vector.
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
     * @param local_point Input point in local coordinates.
     * @return Point in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_world(const Vector3<T>& local_point) const noexcept;

    /**
     * @brief Transforms a world-space point into local space and returns the result.
     *
     * @param world_point Input point in world coordinates.
     * @return Point in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_local(const Vector3<T>& world_point) const noexcept;

    /**
     * @brief Transforms a local-space direction into world space and returns the result.
     *
     * @param local_dir Input direction in local coordinates.
     * @return Direction in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_world(const Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Transforms a world-space direction into local space and returns the result.
     *
     * @param world_dir Input direction in world coordinates.
     * @return Direction in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_local(const Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Transforms a local-space ray into world space and returns the result.
     *
     * @param local_ray Input ray in local coordinates.
     * @return Ray in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::spatial::Ray<T>
    sync_to_world(const atlas::spatial::Ray<T>& local_ray) const noexcept;

    /**
     * @brief Transforms a world-space ray into local space and returns the result.
     *
     * @param world_ray Input ray in world coordinates.
     * @return Ray in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::spatial::Ray<T>
    sync_to_local(const atlas::spatial::Ray<T>& world_ray) const noexcept;
};

}

namespace atlas {

/**
 * @brief Alias for atlas::physics::SyncOperator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SyncOperator = physics::SyncOperator<T>;

}

#include <atlas/sync/sync_operator.hpp>