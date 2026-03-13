#pragma once

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sync/sync_operator.h>

namespace atlas::system {

/**
 * @brief High-level rigid synchronization wrapper around @ref SyncOperator.
 *
 * @tparam T Scalar type used by translation, orientation, and transformed data.
 *
 * `Sync` is a value-type façade over @ref SyncOperator that provides:
 * - point conversion between local and world frames
 * - direction conversion between local and world frames
 * - ray conversion between local and world frames
 * - mutators for translation/orientation/pose
 * - a builder-based construction path with basic validation
 *
 * Conceptually, the stored rigid pose is:
 * - translation `t`
 * - rotation `R` derived from a quaternion
 *
 * and the transforms are:
 * - point local to world: `p_w = R p_l + t`
 * - point world to local: `p_l = R^-1 (p_w - t)`
 * - direction local to world: `d_w = R d_l`
 * - direction world to local: `d_l = R^-1 d_w`
 *
 * `Sync` itself mostly delegates the mathematical work to the contained
 * @ref SyncOperator while providing a simpler stateful API.
 */
template <typename T>
class Sync final {
public:
    /** @brief Fluent builder for constructing validated @ref Sync instances. */
    class Builder;

    /**
     * @brief Constructs the identity rigid transform.
     *
     * The resulting object performs no coordinate change.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE constexpr Sync() noexcept;

    /**
     * @brief Constructs a rigid transform from translation and orientation.
     *
     * @param translation_ Translation applied to local points after rotation.
     * @param orientation_ Quaternion rotating local coordinates into world
     * coordinates.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE constexpr Sync(const Vector3<T>& translation_,
                                                 const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Constructs a wrapper from an existing @ref SyncOperator.
     *
     * @param op Synchronization operator to copy into this object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Sync(const atlas::system::SyncOperator<T>& op) noexcept;

    /**
     * @brief Creates a builder for @ref Sync.
     *
     * @return Default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ~Sync() = default;

    /**
     * @brief Returns a point transformed from local space to world space.
     *
     * @param local_point Point expressed in local coordinates.
     * @return Point expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_world(const Vector3<T>& local_point) const noexcept;

    /**
     * @brief Returns a point transformed from world space to local space.
     *
     * @param world_point Point expressed in world coordinates.
     * @return Point expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_local(const Vector3<T>& world_point) const noexcept;

    /**
     * @brief Returns a direction transformed from local space to world space.
     *
     * @param local_dir Direction expressed in local coordinates.
     * @return Direction expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_world(const Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Returns a direction transformed from world space to local space.
     *
     * @param world_dir Direction expressed in world coordinates.
     * @return Direction expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_local(const Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Returns a ray transformed from local space to world space.
     *
     * @param local_ray Ray expressed in local coordinates.
     * @return Ray expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray<T>
    sync_to_world(const Ray<T>& local_ray) const noexcept;

    /**
     * @brief Returns a ray transformed from world space to local space.
     *
     * @param world_ray Ray expressed in world coordinates.
     * @return Ray expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray<T>
    sync_to_local(const Ray<T>& world_ray) const noexcept;

    /**
     * @brief Transforms a ray from local space to world space in-place via
     * output parameter.
     *
     * @param local_ray Input ray expressed in local coordinates.
     * @param world_ray Output ray expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Ray<T>& local_ray,
                  Ray<T>& world_ray) const noexcept;

    /**
     * @brief Transforms a ray from world space to local space in-place via
     * output parameter.
     *
     * @param world_ray Input ray expressed in world coordinates.
     * @param local_ray Output ray expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Ray<T>& world_ray,
                  Ray<T>& local_ray) const noexcept;

    /**
     * @brief Rebuilds cached rotation matrices from the stored quaternion.
     *
     * Call this after direct state mutations that bypass the higher-level pose
     * setters, or whenever cache coherence needs to be restored manually.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept;

    /**
     * @brief Replaces the translation component of the stored rigid pose.
     *
     * @param translation_ New translation vector.
     *
     * Rotation caches remain valid because translation does not affect them.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_translation(const Vector3<T>& translation_) noexcept;

    /**
     * @brief Replaces the orientation component of the stored rigid pose.
     *
     * @param orientation_ New quaternion orientation.
     *
     * Cached rotation matrices are rebuilt automatically.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_orientation(const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Replaces both translation and orientation of the stored rigid pose.
     *
     * @param translation_ New translation vector.
     * @param orientation_ New quaternion orientation.
     *
     * Cached rotation matrices are rebuilt automatically.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_pose(const Vector3<T>& translation_,
             const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Returns a const reference to the underlying synchronization
     * operator.
     *
     * @return Stored @ref SyncOperator reference.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::system::SyncOperator<T>&
    sync() const noexcept;

    /**
     * @brief Returns a copy of the underlying synchronization operator.
     *
     * @return Value copy of the stored @ref SyncOperator.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::system::SyncOperator<T>
    make_sync_operator() const noexcept;

private:
    friend class Builder;

private:
    atlas::system::SyncOperator<T> sync_operator;
};

template <typename T>
class Sync<T>::Builder final {
public:
    /** @brief Creates a builder initialized to the identity rigid pose. */
    Builder() = default;

    /**
     * @brief Configures the builder from translation and orientation values.
     *
     * @param translation_ Translation component of the rigid pose.
     * @param orientation_ Quaternion component of the rigid pose.
     * @return Reference to this builder.
     *
     * Calling this clears any previously supplied stored sync operator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rigid_pose(const Vector3<T>& translation_,
                    const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Configures the builder from a preconstructed @ref SyncOperator.
     *
     * @param op Operator to copy into builder storage.
     * @return Reference to this builder.
     *
     * Calling this takes precedence over the separately stored translation and
     * orientation values.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sync_operator(const atlas::system::SyncOperator<T>& op) noexcept;

    /**
     * @brief Builds a validated @ref Sync value.
     *
     * @return Constructed synchronization object.
     *
     * @throws std::runtime_error If validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sync<T>
    build() const;

    /**
     * @brief Builds a validated host-side shared pointer to @ref Sync.
     *
     * @return Shared pointer owning the constructed object.
     *
     * @throws std::runtime_error If validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sync<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the builder state before object construction.
     *
     * Validation checks:
     * - translation coordinates are finite
     * - quaternion coordinates are finite
     * - quaternion is not the zero quaternion
     *
     * Non-unit quaternions are allowed but produce a warning because rigid
     * rotation behavior is only mathematically correct for normalized
     * quaternions.
     *
     * @throws std::runtime_error If any hard validation rule is violated.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /** @brief Translation stored when building from explicit pose values. */
    Vector3<T> _translation { T(0), T(0), T(0) };

    /** @brief Orientation stored when building from explicit pose values. */
    Quaternion<T> _orientation { T(1), T(0), T(0), T(0) };

    /** @brief Whether `_op_storage` should be used instead of pose fields. */
    bool _has_sync_operator = false;

    /** @brief Stored operator used when `with_sync_operator()` was selected. */
    atlas::system::SyncOperator<T> _op_storage {};
};

} // namespace atlas::system

namespace atlas {

/** @brief Public namespace alias for @ref atlas::system::Sync. */
template <typename T>
using Sync = atlas::system::Sync<T>;

/** @brief Host shared-pointer alias for @ref atlas::system::Sync. */
template <typename T>
using SyncHostPtr = atlas::host_shared_ptr<atlas::system::Sync<T>>;

/** @brief Device shared-pointer alias for @ref atlas::system::Sync. */
template <typename T>
using SyncDevicePtr = atlas::device_shared_ptr<atlas::system::Sync<T>>;

} // namespace atlas

#include <atlas/sync/sync.hpp>
