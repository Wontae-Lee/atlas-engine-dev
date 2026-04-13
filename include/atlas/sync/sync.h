#pragma once

/**
 * @file sync.h
 * @brief Declares a host-friendly rigid-transform wrapper around @ref atlas::system::SyncOperator.
 *
 * @details
 * This header defines @ref atlas::system::Sync, a small host-oriented wrapper
 * that stores a rigid transform and exposes convenience functions for converting
 * points, directions, and rays between local space and world space.
 *
 * The wrapper is built on top of @ref atlas::system::SyncOperator, which acts as
 * the underlying value-type transform representation used by runtime code.
 *
 * ## Purpose
 * A @ref Sync object represents the pose of an object in space through:
 * - a translation,
 * - an orientation quaternion,
 * - cached matrices or equivalent transform state maintained by the backing
 *   @ref SyncOperator.
 *
 * This makes it suitable for:
 * - geometry placement,
 * - local/world coordinate conversion,
 * - unit pose storage,
 * - source, sink, collider, and visualization transforms.
 *
 * ## Coordinate-space conventions
 * The transform conceptually maps:
 * - **local space** → **world space** using the stored pose,
 * - **world space** → **local space** using the inverse pose.
 *
 * Separate helpers are provided for:
 * - points, which are affected by rotation and translation,
 * - directions, which are affected by rotation only,
 * - rays, whose origin and direction are transformed consistently.
 *
 * ## Host/device usage
 * Although @ref Sync is described as host-friendly, most conversion and mutation
 * functions are marked `ATLAS_ALL_DEVICE`, which allows the object to be used in:
 * - host-side logic,
 * - backend-portable device code,
 * - geometry and unit query paths.
 *
 * ## Construction
 * A @ref Sync may be:
 * - default-constructed as an identity transform,
 * - constructed from explicit translation and orientation,
 * - constructed from an existing @ref SyncOperator,
 * - configured through the nested fluent @ref Builder.
 *
 * ## Builder behavior
 * The builder supports two staged configuration modes:
 * - explicit rigid pose through translation and orientation,
 * - direct injection of a prebuilt @ref SyncOperator.
 *
 * Validation and exact precedence rules are implementation-defined in `sync.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for translation, rotation, and spatial queries.
 */

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sync/sync_operator.h>

namespace atlas::system {

/**
 * @brief Stores a rigid transform and exposes local/world conversion helpers.
 *
 * @details
 * @ref Sync is a thin owning wrapper over @ref atlas::system::SyncOperator.
 * It provides a higher-level, host-friendly interface for managing rigid poses
 * while still exposing backend-portable conversion functions.
 *
 * A sync object is typically used anywhere Atlas needs to associate a local
 * geometry definition with a world-space placement, such as:
 * - @ref atlas::system::Unit,
 * - sources and sinks,
 * - colliders,
 * - visualization layers.
 *
 * ## Internal representation
 * The actual transform logic is delegated to the stored @ref sync_operator.
 * The wrapper mainly provides:
 * - object-oriented storage,
 * - conversion convenience methods,
 * - builder integration,
 * - explicit pose mutation and matrix rebuild helpers.
 *
 * ## Transform semantics
 * - `sync_to_world(...)` applies the forward transform,
 * - `sync_to_local(...)` applies the inverse transform,
 * - direction transforms ignore translation,
 * - ray transforms convert both origin and direction.
 *
 * ## Matrix maintenance
 * When translation or orientation changes, the cached transform state may need
 * to be refreshed. This is done explicitly through @ref rebuild_matrices or
 * implicitly depending on the implementation of the setter functions in
 * `sync.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Sync final {
public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Sync.
     *
     * @details
     * The builder stages either:
     * - a rigid pose (translation + orientation), or
     * - a prebuilt @ref SyncOperator,
     * validates the staged state, and constructs either:
     * - a @ref Sync by value, or
     * - a host-owned shared pointer to a @ref Sync.
     */
    class Builder;

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an identity rigid transform.
     *
     * A typical identity pose corresponds to:
     * - zero translation,
     * - identity quaternion orientation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE constexpr Sync() noexcept;

    /**
     * @brief Construct a rigid transform from explicit translation and orientation.
     *
     * @param translation_ World-space translation component.
     * @param orientation_ Orientation quaternion.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE constexpr Sync(const Vector3<T>& translation_,
                                                 const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Construct a wrapper from an existing @ref SyncOperator.
     *
     * @param op Backing transform operator to store.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Sync(const atlas::system::SyncOperator<T>& op) noexcept;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Destructor.
     */
    ~Sync() = default;

    /**
     * @brief Transform a point from local space to world space.
     *
     * @details
     * Applies the forward rigid transform:
     * - rotate by the stored orientation,
     * - then translate by the stored translation.
     *
     * @param local_point Point expressed in local coordinates.
     * @return The corresponding point in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_world(const Vector3<T>& local_point) const noexcept;

    /**
     * @brief Transform a point from world space to local space.
     *
     * @details
     * Applies the inverse rigid transform:
     * - remove translation,
     * - apply the inverse rotation.
     *
     * @param world_point Point expressed in world coordinates.
     * @return The corresponding point in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_local(const Vector3<T>& world_point) const noexcept;

    /**
     * @brief Transform a direction from local space to world space.
     *
     * @details
     * Directions are rotated but not translated.
     *
     * @param local_dir Direction expressed in local coordinates.
     * @return The corresponding world-space direction.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_world(const Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Transform a direction from world space to local space.
     *
     * @details
     * Directions are inverse-rotated but not translated.
     *
     * @param world_dir Direction expressed in world coordinates.
     * @return The corresponding local-space direction.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_local(const Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Transform a ray from local space to world space.
     *
     * @details
     * The ray origin is transformed as a point and the ray direction is
     * transformed as a direction.
     *
     * @param local_ray Ray expressed in local coordinates.
     * @return The corresponding ray in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray<T>
    sync_to_world(const Ray<T>& local_ray) const noexcept;

    /**
     * @brief Transform a ray from world space to local space.
     *
     * @details
     * The ray origin is transformed as a point and the ray direction is
     * transformed as a direction using the inverse pose.
     *
     * @param world_ray Ray expressed in world coordinates.
     * @return The corresponding ray in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray<T>
    sync_to_local(const Ray<T>& world_ray) const noexcept;

    /**
     * @brief Transform a ray from local space to world space into an output parameter.
     *
     * @param local_ray Input ray in local coordinates.
     * @param world_ray Output ray in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Ray<T>& local_ray,
                  Ray<T>& world_ray) const noexcept;

    /**
     * @brief Transform a ray from world space to local space into an output parameter.
     *
     * @param world_ray Input ray in world coordinates.
     * @param local_ray Output ray in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Ray<T>& world_ray,
                  Ray<T>& local_ray) const noexcept;

    /**
     * @brief Rebuild cached transform matrices or equivalent derived state.
     *
     * @details
     * This function should be called after pose mutation if the implementation
     * requires explicit refresh of cached matrix data.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept;

    /**
     * @brief Set the translation component of the pose.
     *
     * @param translation_ New translation vector.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_translation(const Vector3<T>& translation_) noexcept;

    /**
     * @brief Set the orientation component of the pose.
     *
     * @param orientation_ New orientation quaternion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_orientation(const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Set the full rigid pose.
     *
     * @param translation_ New translation vector.
     * @param orientation_ New orientation quaternion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_pose(const Vector3<T>& translation_,
             const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Return const access to the backing sync operator.
     *
     * @return Const reference to the stored @ref SyncOperator.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::system::SyncOperator<T>&
    sync() const noexcept;

    /**
     * @brief Return a copy of the backing sync operator.
     *
     * @return A value copy of the stored @ref SyncOperator.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::system::SyncOperator<T>
    make_sync_operator() const noexcept;

private:
    /// @brief Allow the builder to configure internals directly.
    friend class Builder;

private:
    /**
     * @brief Backing rigid transform operator.
     *
     * @details
     * Stores the actual transform state used by all conversion helpers.
     */
    atlas::system::SyncOperator<T> sync_operator; ///< Backing rigid transform operator.
};

/**
 * @brief Fluent builder for @ref Sync.
 *
 * @details
 * The builder provides a controlled way to construct a @ref Sync while staging
 * either:
 * - an explicit rigid pose, or
 * - a prebuilt @ref SyncOperator.
 *
 * ## Typical usage
 * @code
 * auto sync = atlas::system::Sync<float>::builder()
 *     .with_rigid_pose({1.0f, 0.0f, 0.0f}, atlas::Quaternion<float>())
 *     .build();
 * @endcode
 *
 * or:
 *
 * @code
 * auto sync = atlas::system::Sync<float>::builder()
 *     .with_sync_operator(op)
 *     .make_host_shared();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - quaternion validity/normalization expectations,
 * - consistency of staged pose state,
 * - precedence rules between pose-based and operator-based staging.
 *
 * The exact validation rules are implementation-defined in `sync.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Sync<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the staged pose to the identity transform.
     */
    Builder() = default;

    /**
     * @brief Stage an explicit rigid pose.
     *
     * @param translation_ Translation component of the pose.
     * @param orientation_ Orientation quaternion of the pose.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rigid_pose(const Vector3<T>& translation_,
                    const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Stage a prebuilt sync operator.
     *
     * @details
     * This can be used when transform state has already been prepared elsewhere.
     *
     * @param op Sync operator to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sync_operator(const atlas::system::SyncOperator<T>& op) noexcept;

    /**
     * @brief Build a configured @ref Sync by value after validation.
     *
     * @return Constructed sync object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sync<T>
    build() const;

    /**
     * @brief Build a configured @ref Sync in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<Sync<T>>` owning the constructed object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sync<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs any implementation-defined consistency checks on the staged pose
     * or sync operator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending translation component.
     */
    Vector3<T> _translation { T(0), T(0), T(0) };

    /**
     * @brief Pending orientation quaternion.
     */
    Quaternion<T> _orientation { T(1), T(0), T(0), T(0) };

    /**
     * @brief Whether a prebuilt sync operator has been staged explicitly.
     */
    bool _has_sync_operator = false;

    /**
     * @brief Storage for a staged sync operator.
     */
    atlas::system::SyncOperator<T> _operator {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::Sync.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Sync = atlas::system::Sync<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::system::Sync.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SyncHostPtr = atlas::host_shared_ptr<atlas::system::Sync<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::system::Sync.
 *
 * @details
 * Although @ref Sync is primarily a host-friendly wrapper, this alias is
 * provided for API symmetry with other Atlas components.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SyncDevicePtr = atlas::device_shared_ptr<atlas::system::Sync<T>>;

} // namespace atlas

#include <atlas/sync/sync.hpp>