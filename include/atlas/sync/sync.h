#pragma once

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sync/sync_operator.h>

namespace atlas::physics {

/**
 * @brief Rigid transform state used to convert points, directions, and rays
 *        between local and world coordinate spaces.
 *
 * Sync is a host-side transform state wrapper around SyncOperator. It stores a
 * rigid pose through an internal SyncOperator and exposes convenience functions
 * for coordinate conversion.
 *
 * A rigid transform is composed of:
 *
 * - a translation vector @f$\mathbf{t}@f$,
 * - an orientation quaternion @f$q@f$,
 * - a cached rotation matrix @f$\mathbf{R}@f$ derived from @f$q@f$,
 * - a cached inverse rotation matrix @f$\mathbf{R}^{-1}@f$.
 *
 * For a local-space point @f$\mathbf{x}_{\mathrm{local}}@f$, the corresponding
 * world-space point is computed as:
 *
 * @f[
 *     \mathbf{x}_{\mathrm{world}}
 *     =
 *     \mathbf{R}\mathbf{x}_{\mathrm{local}}
 *     +
 *     \mathbf{t}.
 * @f]
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
 * Direction vectors are transformed using only the rotational component because
 * they represent orientation or displacement, not absolute position:
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
 * Rays are transformed by treating the ray origin as a point and the ray
 * direction as a direction vector.
 *
 * @tparam T Floating-point scalar type used for transform computations.
 *
 * @note This class owns a SyncOperator and forwards transform operations to it.
 * @note The cached transform matrices must remain consistent with the current
 *       orientation. Use set_orientation(), set_pose(), or rebuild_matrices()
 *       when changing orientation-dependent state.
 * @note Translation updates do not require rebuilding rotation matrices.
 *
 * @see SyncOperator
 */
template <typename T>
class Sync final {
public:
    /**
     * @brief Builder type for constructing a Sync instance on the host side.
     *
     * The builder validates translation and orientation values before producing
     * a Sync object.
     */
    class Builder;

    /**
     * @brief Constructs an identity sync state.
     *
     * The default state represents the identity rigid transform:
     *
     * @f[
     *     \mathbf{t}
     *     =
     *     \begin{bmatrix}
     *         0 \\ 0 \\ 0
     *     \end{bmatrix},
     *     \qquad
     *     q
     *     =
     *     \left(
     *         1, 0, 0, 0
     *     \right),
     *     \qquad
     *     \mathbf{R}
     *     =
     *     \mathbf{I}.
     * @f]
     *
     * Under this transform, points, directions, and rays are unchanged by
     * local-to-world or world-to-local conversion.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE constexpr Sync() noexcept;

    /**
     * @brief Constructs a sync state from translation and orientation.
     *
     * The supplied pose defines the rigid transform:
     *
     * @f[
     *     \mathbf{x}_{\mathrm{world}}
     *     =
     *     \mathbf{R}(q)\mathbf{x}_{\mathrm{local}}
     *     +
     *     \mathbf{t},
     * @f]
     *
     * where @f$\mathbf{t}@f$ is @p translation_ and @f$q@f$ is
     * @p orientation_.
     *
     * @param translation_ Translation component @f$\mathbf{t}@f$.
     * @param orientation_ Orientation quaternion @f$q@f$.
     *
     * @note The internal SyncOperator is responsible for building the cached
     *       transform matrices.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE constexpr Sync(const Vector3<T>& translation_,
                                                 const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Constructs a sync state from an existing sync operator.
     *
     * The supplied operator is copied into the internal transform state.
     *
     * @param op Existing sync operator containing translation, orientation, and
     *           cached transform matrices.
     *
     * @note This constructor assumes that @p op already represents a consistent
     *       transform state.
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
     * This function applies the full rigid transform:
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
     * @param local_point Point @f$\mathbf{x}_{\mathrm{local}}@f$ expressed in
     *                    local coordinates.
     * @return Point @f$\mathbf{x}_{\mathrm{world}}@f$ expressed in world
     *         coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_world(const Vector3<T>& local_point) const noexcept;

    /**
     * @brief Transforms a point from world space to local space.
     *
     * This function applies the inverse rigid transform:
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
     * @param world_point Point @f$\mathbf{x}_{\mathrm{world}}@f$ expressed in
     *                    world coordinates.
     * @return Point @f$\mathbf{x}_{\mathrm{local}}@f$ expressed in local
     *         coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_local(const Vector3<T>& world_point) const noexcept;

    /**
     * @brief Transforms a direction vector from local space to world space.
     *
     * This function applies only the rotational component:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{d}_{\mathrm{local}}.
     * @f]
     *
     * Translation is not applied because a direction vector does not represent an
     * absolute position.
     *
     * @param local_dir Direction @f$\mathbf{d}_{\mathrm{local}}@f$ expressed in
     *                  local coordinates.
     * @return Direction @f$\mathbf{d}_{\mathrm{world}}@f$ expressed in world
     *         coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_world(const Vector3<T>& local_dir) const noexcept;

    /**
     * @brief Transforms a direction vector from world space to local space.
     *
     * This function applies only the inverse rotational component:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}\mathbf{d}_{\mathrm{world}}.
     * @f]
     *
     * Translation is not applied because direction vectors are independent of
     * position.
     *
     * @param world_dir Direction @f$\mathbf{d}_{\mathrm{world}}@f$ expressed in
     *                  world coordinates.
     * @return Direction @f$\mathbf{d}_{\mathrm{local}}@f$ expressed in local
     *         coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_local(const Vector3<T>& world_dir) const noexcept;

    /**
     * @brief Transforms a ray from local space to world space.
     *
     * A ray can be written as:
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
     * The ray origin is transformed as a point:
     *
     * @f[
     *     \mathbf{o}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{o}_{\mathrm{local}}
     *     +
     *     \mathbf{t},
     * @f]
     *
     * and the ray direction is transformed as a direction vector:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{world}}
     *     =
     *     \mathbf{R}\mathbf{d}_{\mathrm{local}}.
     * @f]
     *
     * @param local_ray Ray expressed in local coordinates.
     * @return Ray expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray<T>
    sync_to_world(const Ray<T>& local_ray) const noexcept;

    /**
     * @brief Transforms a ray from world space to local space.
     *
     * A ray can be written as:
     *
     * @f[
     *     r(s)
     *     =
     *     \mathbf{o}
     *     +
     *     s\mathbf{d}.
     * @f]
     *
     * The ray origin is transformed with the inverse point transform:
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
     * and the ray direction is transformed with the inverse direction transform:
     *
     * @f[
     *     \mathbf{d}_{\mathrm{local}}
     *     =
     *     \mathbf{R}^{-1}\mathbf{d}_{\mathrm{world}}.
     * @f]
     *
     * @param world_ray Ray expressed in world coordinates.
     * @return Ray expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray<T>
    sync_to_local(const Ray<T>& world_ray) const noexcept;

    /**
     * @brief Transforms a local-space ray into a preallocated world-space ray.
     *
     * This output-parameter overload avoids returning the ray by value. It applies
     * the same operation as sync_to_world(const Ray<T>&):
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
     * @param local_ray Input ray expressed in local coordinates.
     * @param world_ray Output ray expressed in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Ray<T>& local_ray,
                  Ray<T>& world_ray) const noexcept;

    /**
     * @brief Transforms a world-space ray into a preallocated local-space ray.
     *
     * This output-parameter overload avoids returning the ray by value. It applies
     * the same operation as sync_to_local(const Ray<T>&):
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
     * @param world_ray Input ray expressed in world coordinates.
     * @param local_ray Output ray expressed in local coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Ray<T>& world_ray,
                  Ray<T>& local_ray) const noexcept;

    /**
     * @brief Rebuilds cached transformation matrices from the current orientation.
     *
     * This function forwards to the internal SyncOperator and updates the cached
     * rotation matrix and inverse rotation matrix:
     *
     * @f[
     *     \mathbf{R}
     *     =
     *     \mathbf{R}(q),
     *     \qquad
     *     \mathbf{R}^{-1}
     *     =
     *     \mathbf{R}^{T}.
     * @f]
     *
     * where @f$q@f$ is the current orientation quaternion.
     *
     * This function should be called after directly changing orientation data
     * when the cached matrices must remain consistent with the current pose.
     *
     * @note Translation changes do not require rebuilding these matrices.
     * @note For a valid rotation matrix, the inverse is the transpose.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept;

    /**
     * @brief Sets the translation component of the pose.
     *
     * This function updates only the translation vector:
     *
     * @f[
     *     \mathbf{t}
     *     \leftarrow
     *     \mathbf{t}_{\mathrm{new}}.
     * @f]
     *
     * The orientation and cached rotation matrices are left unchanged because
     * translation does not affect the rotational part of the transform.
     *
     * @param translation_ New translation vector.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_translation(const Vector3<T>& translation_) noexcept;

    /**
     * @brief Sets the orientation component of the pose.
     *
     * This function updates the orientation quaternion and rebuilds the cached
     * rotation matrices:
     *
     * @f[
     *     q
     *     \leftarrow
     *     q_{\mathrm{new}},
     *     \qquad
     *     \mathbf{R}
     *     \leftarrow
     *     \mathbf{R}(q).
     * @f]
     *
     * @param orientation_ New orientation quaternion.
     *
     * @note The supplied quaternion is assigned directly. This function does not
     *       explicitly normalize it unless SyncOperator or Quaternion implements
     *       normalization internally.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_orientation(const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Sets both translation and orientation of the pose.
     *
     * This function updates the full rigid pose:
     *
     * @f[
     *     \mathbf{t}
     *     \leftarrow
     *     \mathbf{t}_{\mathrm{new}},
     *     \qquad
     *     q
     *     \leftarrow
     *     q_{\mathrm{new}}.
     * @f]
     *
     * After assigning the new orientation, the cached rotation matrices are
     * rebuilt so that future coordinate conversions use the updated pose.
     *
     * @param translation_ New translation vector.
     * @param orientation_ New orientation quaternion.
     *
     * @note The supplied quaternion is assigned directly. This function does not
     *       explicitly normalize it unless SyncOperator or Quaternion implements
     *       normalization internally.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_pose(const Vector3<T>& translation_,
             const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Returns the underlying sync operator by constant reference.
     *
     * The returned reference provides read-only access to the internal operator,
     * including translation, orientation, and cached transform matrices.
     *
     * @return Constant reference to the internal sync operator.
     *
     * @note The returned reference remains valid only while this Sync object is
     *       alive and unmodified in a way that would invalidate object storage.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::physics::SyncOperator<T>&
    sync() const noexcept;

    /**
     * @brief Creates and returns a copy of the underlying sync operator.
     *
     * This function is useful when a lightweight operator object is needed for
     * device-side kernels or other code paths that should not depend on the Sync
     * owner object.
     *
     * @return Copy of the internal sync operator.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::physics::SyncOperator<T>
    make_sync_operator() const noexcept;

private:
    friend class Builder;

private:
    /**
     * @brief Internal sync operator storing pose and cached transform data.
     *
     * This object contains:
     *
     * - translation @f$\mathbf{t}@f$,
     * - orientation quaternion @f$q@f$,
     * - rotation matrix @f$\mathbf{R}@f$,
     * - inverse rotation matrix @f$\mathbf{R}^{-1}@f$.
     *
     * All coordinate conversion functions delegate to this operator.
     */
    atlas::physics::SyncOperator<T> sync_operator;
};

/**
 * @brief Host-side builder for validated Sync construction.
 *
 * The builder supports two construction modes:
 *
 * 1. pose mode, where translation and orientation are supplied separately,
 * 2. operator mode, where a complete SyncOperator is supplied directly.
 *
 * In pose mode, the final SyncOperator is constructed from:
 *
 * @f[
 *     \left(
 *         \mathbf{t}, q
 *     \right).
 * @f]
 *
 * In operator mode, the supplied SyncOperator is copied into the Sync object.
 *
 * Validation checks that the selected translation and orientation contain finite
 * values and that the selected orientation quaternion is not the zero quaternion.
 *
 * @tparam T Floating-point scalar type used by the target Sync object.
 */
template <typename T>
class Sync<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * Creates a builder initialized with an identity pose:
     *
     * @f[
     *     \mathbf{t}
     *     =
     *     \begin{bmatrix}
     *         0 \\ 0 \\ 0
     *     \end{bmatrix},
     *     \qquad
     *     q
     *     =
     *     \left(
     *         1, 0, 0, 0
     *     \right).
     * @f]
     */
    Builder() = default;

    /**
     * @brief Sets the rigid pose using translation and orientation.
     *
     * This switches the builder into pose mode. Any SyncOperator previously set
     * through with_sync_operator() is ignored for the final construction.
     *
     * @param translation_ Translation vector @f$\mathbf{t}@f$.
     * @param orientation_ Orientation quaternion @f$q@f$.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rigid_pose(const Vector3<T>& translation_,
                    const Quaternion<T>& orientation_) noexcept;

    /**
     * @brief Sets the builder state from an existing sync operator.
     *
     * This switches the builder into operator mode. The supplied operator is
     * copied and used directly during construction.
     *
     * @param op Existing sync operator.
     * @return Reference to this builder.
     *
     * @note Validation is performed on the translation and orientation contained
     *       in @p op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sync_operator(const atlas::physics::SyncOperator<T>& op) noexcept;

    /**
     * @brief Validates the current builder state and constructs a Sync instance.
     *
     * The selected construction state is validated first. Then the Sync object is
     * built either from the stored pose or from the stored SyncOperator.
     *
     * @return Fully constructed Sync object.
     *
     * @throws std::runtime_error Thrown if translation contains non-finite values.
     * @throws std::runtime_error Thrown if orientation contains non-finite values.
     * @throws std::runtime_error Thrown if orientation is the zero quaternion.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sync<T>
    build() const;

    /**
     * @brief Builds a Sync instance and wraps it in a host shared pointer.
     *
     * This function performs the same validation and construction process as
     * build(), then stores the resulting Sync object in Atlas host-managed shared
     * storage.
     *
     * @return Shared host pointer owning the constructed Sync object.
     *
     * @throws std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sync<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the internal builder state before construction.
     *
     * The validated pose is selected according to the active builder mode:
     *
     * - if a SyncOperator was supplied, its translation and orientation are used,
     * - otherwise, the separately stored translation and orientation are used.
     *
     * Validation rejects:
     *
     * - non-finite translation components,
     * - non-finite quaternion components,
     * - the zero quaternion @f$(0,0,0,0)@f$.
     *
     * @throws std::runtime_error Thrown if translation contains non-finite values.
     * @throws std::runtime_error Thrown if orientation contains non-finite values.
     * @throws std::runtime_error Thrown if orientation is the zero quaternion.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Translation used when constructing from a rigid pose.
     *
     * This value is used only when the builder is in pose mode.
     */
    Vector3<T> _translation { T(0), T(0), T(0) };

    /**
     * @brief Orientation used when constructing from a rigid pose.
     *
     * This value is used only when the builder is in pose mode. The default value
     * is the identity quaternion:
     *
     * @f[
     *     q
     *     =
     *     \left(
     *         1, 0, 0, 0
     *     \right).
     * @f]
     */
    Quaternion<T> _orientation { T(1), T(0), T(0), T(0) };

    /**
     * @brief Indicates whether a complete sync operator has been provided.
     *
     * If true, build() uses @ref _operator. If false, build() constructs a
     * SyncOperator from @ref _translation and @ref _orientation.
     */
    bool _has_sync_operator = false;

    /**
     * @brief Cached sync operator supplied through the builder.
     *
     * This value is used only when the builder is in operator mode.
     */
    atlas::physics::SyncOperator<T> _operator {};
};

} // namespace atlas::physics

namespace atlas {

/**
 * @brief Alias for atlas::physics::Sync.
 *
 * This alias exposes the rigid transform state type directly in the atlas
 * namespace.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Sync = atlas::physics::Sync<T>;

/**
 * @brief Host shared pointer alias for Sync.
 *
 * This alias represents a host-managed shared pointer to atlas::physics::Sync.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SyncHostPtr = atlas::host_shared_ptr<atlas::physics::Sync<T>>;

/**
 * @brief Device shared pointer alias for Sync.
 *
 * This alias represents a device-managed shared pointer to atlas::physics::Sync.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SyncDevicePtr = atlas::device_shared_ptr<atlas::physics::Sync<T>>;

} // namespace atlas

#include <atlas/sync/sync.hpp>