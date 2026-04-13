#pragma once

/**
 * @file sync_operator.h
 * @brief Declares the backend-portable rigid-transform operator used for local/world coordinate conversion.
 *
 * @details
 * This header defines @ref atlas::system::SyncOperator, a lightweight value-type
 * rigid transform that stores:
 * - a translation vector,
 * - an orientation quaternion,
 * - a forward rotation matrix,
 * - an inverse rotation matrix.
 *
 * ## Purpose
 * The sync operator is the low-level transform primitive used throughout Atlas
 * whenever an object, geometry, ray, or direction must be converted between:
 * - **local space**, where a shape is naturally defined, and
 * - **world space**, where that shape is placed in the simulation scene.
 *
 * It is intentionally designed as a compact, backend-portable value object so it
 * can be:
 * - copied into device buffers,
 * - embedded inside higher-level runtime objects,
 * - used in host/device query code without additional ownership machinery.
 *
 * ## Rigid-transform model
 * The operator represents a rigid pose composed of:
 * - a translation \f$\mathbf{t}\f$,
 * - an orientation \f$R\f$ derived from a quaternion.
 *
 * Typical forward transformations are:
 * - point: \f$\mathbf{x}_w = R \mathbf{x}_l + \mathbf{t}\f$
 * - direction: \f$\mathbf{d}_w = R \mathbf{d}_l\f$
 *
 * The corresponding inverse transformations use the inverse rotation matrix and
 * reverse the translation.
 *
 * ## Cached matrices
 * To avoid repeatedly reconstructing rotation data from the quaternion, the
 * operator stores:
 * - @ref orientation_matrix for forward rotation,
 * - @ref inverse_orientation_matrix for inverse rotation.
 *
 * These matrices are refreshed by calling @ref rebuild_matrices.
 *
 * ## Conversion coverage
 * The operator provides overloads for:
 * - point transformation using output parameters,
 * - point transformation returning values,
 * - direction transformation using output parameters,
 * - direction transformation returning values,
 * - ray transformation using output parameters,
 * - ray transformation returning values.
 *
 * ## Host/device usage
 * Most member functions are marked `ATLAS_ALL_DEVICE`, allowing the operator to
 * be used in:
 * - host-side geometry orchestration,
 * - device-side collision and query kernels,
 * - backend-portable runtime pipelines.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for translation, rotation, and coordinate conversion.
 */

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

namespace atlas::system {

/**
 * @brief Backend-portable rigid-transform operator for local/world coordinate conversion.
 *
 * @details
 * @ref SyncOperator is the core value-type rigid transform used by Atlas runtime
 * systems. It stores a pose and exposes conversion helpers for:
 * - points,
 * - directions,
 * - rays.
 *
 * ## Data members
 * The operator stores:
 * - @ref translation, the world-space translation,
 * - @ref orientation, the quaternion describing rotation,
 * - @ref orientation_matrix, the forward rotation matrix,
 * - @ref inverse_orientation_matrix, the inverse rotation matrix.
 *
 * ## Transform semantics
 * The operator distinguishes between:
 * - **points**, which are affected by both rotation and translation,
 * - **directions**, which are affected only by rotation,
 * - **rays**, whose origin is treated as a point and whose direction is treated
 *   as a direction.
 *
 * ## Matrix maintenance
 * After changing the stored quaternion, the derived matrices should be refreshed
 * through @ref rebuild_matrices before using the transform for queries.
 *
 * ## Typical usage
 * A sync operator is commonly embedded in:
 * - @ref atlas::system::Unit for movable scene objects,
 * - source/sink/collider logic,
 * - geometry-placement code,
 * - visualization and world-space query paths.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the simulation.
 */
template <typename T>
struct SyncOperator final {

    /**
     * @brief World-space translation component of the rigid transform.
     *
     * @details
     * This vector is added after forward rotation when mapping a point from local
     * space into world space.
     */
    Vector3<T> translation {};

    /**
     * @brief Orientation quaternion of the rigid transform.
     *
     * @details
     * Encodes the rotational part of the pose. The corresponding forward and
     * inverse rotation matrices are cached in:
     * - @ref orientation_matrix
     * - @ref inverse_orientation_matrix
     */
    Quaternion<T> orientation {};

    /**
     * @brief Cached forward rotation matrix derived from @ref orientation.
     *
     * @details
     * Used for local-to-world rotation of points and directions.
     */
    Matrix<T, 3, 3> orientation_matrix {};

    /**
     * @brief Cached inverse rotation matrix derived from @ref orientation.
     *
     * @details
     * Used for world-to-local rotation of points and directions.
     */
    Matrix<T, 3, 3> inverse_orientation_matrix {};

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a default rigid transform, typically corresponding to the
     * identity pose:
     * - zero translation,
     * - identity orientation,
     * - identity forward/inverse rotation matrices.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr SyncOperator() noexcept;

    /**
     * @brief Construct a rigid transform from explicit translation and orientation.
     *
     * @details
     * Initializes the pose from the supplied translation and quaternion. The
     * implementation may also initialize the cached matrices directly or defer
     * that work until @ref rebuild_matrices is called.
     *
     * @param translation_ World-space translation.
     * @param orientation_ Orientation quaternion.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    SyncOperator(const Vector3<T>& translation_,
                 const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Rebuild the cached forward and inverse rotation matrices.
     *
     * @details
     * This function synchronizes the matrix caches with the current quaternion
     * stored in @ref orientation.
     *
     * It should be called whenever the quaternion has changed and the transform is
     * expected to be used for subsequent coordinate conversion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept;

    /**
     * @brief Transform a point from local space to world space into an output parameter.
     *
     * @details
     * Applies the forward rigid transform:
     * \f[
     * \mathbf{x}_w = R \mathbf{x}_l + \mathbf{t}
     * \f]
     *
     * @param local_point Input point in local coordinates.
     * @param world_point Output point in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Vector3<T>& local_point,
                  Vector3<T>& world_point) const noexcept;

    /**
     * @brief Transform a point from world space to local space into an output parameter.
     *
     * @details
     * Applies the inverse rigid transform:
     * \f[
     * \mathbf{x}_l = R^{-1} (\mathbf{x}_w - \mathbf{t})
     * \f]
     *
     * @param world_point Input point in world coordinates.
     * @param local_point Output point in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Vector3<T>& world_point,
                  Vector3<T>& local_point) const noexcept;

    /**
     * @brief Transform a direction from local space to world space into an output parameter.
     *
     * @details
     * Directions are rotated but not translated:
     * \f[
     * \mathbf{d}_w = R \mathbf{d}_l
     * \f]
     *
     * @param local_dir Input direction in local coordinates.
     * @param world_dir Output direction in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_world(const Vector3<T>& local_dir,
                      Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Transform a direction from world space to local space into an output parameter.
     *
     * @details
     * Directions are inverse-rotated but not translated:
     * \f[
     * \mathbf{d}_l = R^{-1} \mathbf{d}_w
     * \f]
     *
     * @param world_dir Input direction in world coordinates.
     * @param local_dir Output direction in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_local(const Vector3<T>& world_dir,
                      Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Transform a ray from local space to world space into an output parameter.
     *
     * @details
     * The ray origin is transformed as a point and the ray direction is
     * transformed as a direction.
     *
     * @param local_ray Input ray in local coordinates.
     * @param world_ray Output ray in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const atlas::spatial::Ray<T>& local_ray,
                  atlas::spatial::Ray<T>& world_ray) const noexcept;

    /**
     * @brief Transform a ray from world space to local space into an output parameter.
     *
     * @details
     * The ray origin is transformed as a point and the ray direction is
     * transformed as a direction using the inverse pose.
     *
     * @param world_ray Input ray in world coordinates.
     * @param local_ray Output ray in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const atlas::spatial::Ray<T>& world_ray,
                  atlas::spatial::Ray<T>& local_ray) const noexcept;

    /**
     * @brief Transform a point from local space to world space and return the result by value.
     *
     * @param local_point Input point in local coordinates.
     * @return The transformed point in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_world(const Vector3<T>& local_point) const noexcept;

    /**
     * @brief Transform a point from world space to local space and return the result by value.
     *
     * @param world_point Input point in world coordinates.
     * @return The transformed point in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_local(const Vector3<T>& world_point) const noexcept;

    /**
     * @brief Transform a direction from local space to world space and return the result by value.
     *
     * @param local_dir Input direction in local coordinates.
     * @return The transformed direction in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_world(const Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Transform a direction from world space to local space and return the result by value.
     *
     * @param world_dir Input direction in world coordinates.
     * @return The transformed direction in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_local(const Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Transform a ray from local space to world space and return the result by value.
     *
     * @param local_ray Input ray in local coordinates.
     * @return The transformed ray in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::spatial::Ray<T>
    sync_to_world(const atlas::spatial::Ray<T>& local_ray) const noexcept;

    /**
     * @brief Transform a ray from world space to local space and return the result by value.
     *
     * @param world_ray Input ray in world coordinates.
     * @return The transformed ray in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::spatial::Ray<T>
    sync_to_local(const atlas::spatial::Ray<T>& world_ray) const noexcept;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::SyncOperator.
 *
 * @tparam T Floating-point scalar type used by the transform.
 */
template <typename T>
using SyncOperator = system::SyncOperator<T>;

} // namespace atlas

#include <atlas/sync/sync_operator.hpp>