#pragma once

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sync/sync_operator.h>

namespace atlas::physics {

/**
 * @brief Represents a rigid transform state used to convert between local and world spaces.
 *
 * This class encapsulates translation, orientation, and the derived transformation
 * matrices required for coordinate conversion. It provides helper functions for
 * transforming points, directions, and rays between local and world coordinate systems.
 *
 * @tparam T Floating-point scalar type used for transform computations.
 */
template <typename T>
class Sync final {
public:
    /**
     * @brief Builder type for constructing a Sync instance on the host side.
     */
    class Builder;

    /**
     * @brief Constructs an identity sync state.
     *
     * The default state uses zero translation and identity orientation.
     * Derived matrices are expected to match the identity transform.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE constexpr Sync() noexcept;

    /**
     * @brief Constructs a sync state from translation and orientation.
     *
     * @param translation_ Translation component in world coordinates.
     * @param orientation_ Orientation quaternion.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE constexpr Sync(const Vector3<T>& translation_,
                                                 const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Constructs a sync state from an existing sync operator.
     *
     * @param op Existing sync operator containing the full transform state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Sync(const atlas::physics::SyncOperator<T>& op) noexcept;

    /**
     * @brief Creates a builder instance for host-side construction.
     *
     * @return Builder object for chained configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Default destructor.
     */
    ~Sync() = default;

    /**
     * @brief Transforms a point from local space to world space.
     *
     * This transformation applies both rotation and translation.
     *
     * @param local_point Point expressed in local coordinates.
     * @return Point expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_world(const Vector3<T>& local_point) const noexcept;

    /**
     * @brief Transforms a point from world space to local space.
     *
     * This transformation applies the inverse rigid transform.
     *
     * @param world_point Point expressed in world coordinates.
     * @return Point expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_local(const Vector3<T>& world_point) const noexcept;

    /**
     * @brief Transforms a direction vector from local space to world space.
     *
     * Translation is not applied because directions represent orientation only.
     *
     * @param local_dir Direction expressed in local coordinates.
     * @return Direction expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_world(const Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Transforms a direction vector from world space to local space.
     *
     * Translation is not applied because directions represent orientation only.
     *
     * @param world_dir Direction expressed in world coordinates.
     * @return Direction expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_local(const Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Transforms a ray from local space to world space.
     *
     * The ray origin is transformed as a point and the direction is transformed
     * as a direction vector.
     *
     * @param local_ray Ray expressed in local coordinates.
     * @return Ray expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray<T>
    sync_to_world(const Ray<T>& local_ray) const noexcept;

    /**
     * @brief Transforms a ray from world space to local space.
     *
     * The ray origin is transformed as a point and the direction is transformed
     * as a direction vector.
     *
     * @param world_ray Ray expressed in world coordinates.
     * @return Ray expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray<T>
    sync_to_local(const Ray<T>& world_ray) const noexcept;

    /**
     * @brief Transforms a local-space ray into a preallocated world-space ray.
     *
     * @param local_ray Input ray in local coordinates.
     * @param world_ray Output ray in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Ray<T>& local_ray,
                  Ray<T>& world_ray) const noexcept;

    /**
     * @brief Transforms a world-space ray into a preallocated local-space ray.
     *
     * @param world_ray Input ray in world coordinates.
     * @param local_ray Output ray in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Ray<T>& world_ray,
                  Ray<T>& local_ray) const noexcept;

    /**
     * @brief Rebuilds the cached transformation matrices from translation and orientation.
     *
     * This function should be called after directly modifying pose components
     * when the cached matrices must remain consistent with the current state.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept;

    /**
     * @brief Sets the translation component of the pose.
     *
     * This function updates only the translation value. Matrix rebuilding behavior
     * depends on the implementation.
     *
     * @param translation_ New translation in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_translation(const Vector3<T>& translation_) noexcept;

    /**
     * @brief Sets the orientation component of the pose.
     *
     * This function updates only the orientation value. Matrix rebuilding behavior
     * depends on the implementation.
     *
     * @param orientation_ New orientation quaternion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_orientation(const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Sets both translation and orientation of the pose.
     *
     * @param translation_ New translation in world coordinates.
     * @param orientation_ New orientation quaternion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_pose(const Vector3<T>& translation_,
             const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Returns the underlying sync operator by constant reference.
     *
     * @return Constant reference to the sync operator.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::physics::SyncOperator<T>&
    sync() const noexcept;

    /**
     * @brief Creates and returns a copy of the underlying sync operator.
     *
     * @return Copy of the sync operator.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::physics::SyncOperator<T>
    make_sync_operator() const noexcept;

private:
    friend class Builder;

private:
    /**
     * @brief Internal sync operator storing pose and derived transform data.
     */
    atlas::physics::SyncOperator<T> sync_operator;
};

/**
 * @brief Host-side builder for Sync construction.
 *
 * This builder provides a readable construction flow for initializing a rigid pose
 * or directly supplying a prebuilt sync operator.
 *
 * @tparam T Floating-point scalar type used by the target Sync object.
 */
template <typename T>
class Sync<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the rigid pose using translation and orientation.
     *
     * @param translation_ Translation in world coordinates.
     * @param orientation_ Orientation quaternion.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rigid_pose(const Vector3<T>& translation_,
                    const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Sets the builder state from an existing sync operator.
     *
     * @param op Existing sync operator.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sync_operator(const atlas::physics::SyncOperator<T>& op) noexcept;

    /**
     * @brief Validates the current builder state and constructs a Sync instance.
     *
     * @return Fully constructed Sync object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sync<T>
    build() const;

    /**
     * @brief Builds a Sync instance and wraps it in a host shared pointer.
     *
     * @return Shared host pointer owning the constructed Sync object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sync<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the internal builder state before construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Translation to use when constructing from a rigid pose.
     */
    Vector3<T> _translation { T(0), T(0), T(0) };

    /**
     * @brief Orientation to use when constructing from a rigid pose.
     */
    Quaternion<T> _orientation { T(1), T(0), T(0), T(0) };

    /**
     * @brief Indicates whether a full sync operator has been explicitly provided.
     */
    bool _has_sync_operator = false;

    /**
     * @brief Cached sync operator supplied through the builder.
     */
    atlas::physics::SyncOperator<T> _operator {};
};

}

namespace atlas {

/**
 * @brief Alias for atlas::physics::Sync.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Sync = atlas::physics::Sync<T>;

/**
 * @brief Host shared pointer alias for Sync.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SyncHostPtr = atlas::host_shared_ptr<atlas::physics::Sync<T>>;

/**
 * @brief Device shared pointer alias for Sync.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SyncDevicePtr = atlas::device_shared_ptr<atlas::physics::Sync<T>>;

}

#include <atlas/sync/sync.hpp>