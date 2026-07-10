#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/ray.h>

namespace atlas {

/**
 * @brief A rigid-body pose: the local-to-world transform of a single body.
 *
 * A Sync bundles a translation and an orientation quaternion together with the
 * two rotation matrices derived from that quaternion (forward and its inverse),
 * so that point/direction/ray transforms in either direction are a matrix
 * multiply with no per-call quaternion-to-matrix conversion. The stored
 * matrices are a cache: any mutation of @ref orientation must be followed by
 * @ref rebuild_matrices (the setters do this for you).
 *
 * Because the type is trivially copyable and every accessor is annotated
 * `__host__ __device__`, a Sync can be captured by value into a device lambda
 * and used to move geometry between a body's local frame and world space on the
 * GPU. It is the transform half of a @ref Unit (which pairs it with a Geometry).
 *
 * @note The orientation is assumed to be a unit quaternion, so the forward
 *       rotation matrix is orthonormal and its inverse is simply its transpose;
 *       @ref inverse_orientation_matrix is stored as that transpose. Feeding a
 *       non-unit quaternion breaks this assumption; the Builder rejects the zero
 *       quaternion but does not otherwise renormalize.
 */
class Sync final {
public:
    /// Host-side fluent builder that validates and constructs a Sync.
    class Builder;

public:
    Float3 translation;                  ///< World-space position of the body origin.
    Quaternion orientation;              ///< Body orientation; assumed to be a unit quaternion.
    Float3x3 orientation_matrix;         ///< Cached local-to-world rotation (from @ref orientation).
    Float3x3 inverse_orientation_matrix; ///< Cached world-to-local rotation (transpose of the above).

    /**
     * @brief Constructs the identity pose: origin at the world origin, no rotation.
     *
     * Translation is zero, orientation is the identity quaternion, and both
     * cached matrices are the 3x3 identity, so all transforms are no-ops.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Sync() noexcept
        : translation(0.0f, 0.0f, 0.0f)
        , orientation()
        , orientation_matrix(atlas::identity3x3())
        , inverse_orientation_matrix(atlas::identity3x3()) { }

    /**
     * @brief Constructs a pose from a translation and orientation.
     *
     * The cached rotation matrices are derived immediately from @p orientation_
     * via @ref rebuild_matrices, so the pose is ready to transform on return.
     *
     * @param translation_ World-space position of the body origin.
     * @param orientation_ Body orientation; expected to be a unit quaternion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Sync(const Float3& translation_, const Quaternion& orientation_) noexcept
        : translation(translation_)
        , orientation(orientation_)
        , orientation_matrix()
        , inverse_orientation_matrix() {
        rebuild_matrices();
    }

    /// Copy constructor; copies the pose and its cached matrices verbatim.
    Sync(const Sync&) noexcept = default;
    /// Move constructor; equivalent to a copy for this trivially copyable type.
    Sync(Sync&&) noexcept      = default;
    /// Copy assignment; copies the pose and its cached matrices verbatim.
    Sync&
    operator=(const Sync&) noexcept = default;
    /// Move assignment; equivalent to a copy for this trivially copyable type.
    Sync&
    operator=(Sync&&) noexcept = default;

    /// Trivial destructor; the type owns no external resources.
    ~Sync() noexcept = default;

    /**
     * @brief Returns a fresh host-side @ref Builder for constructing a Sync.
     *
     * @return A default-initialized Builder (identity pose until configured).
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Recomputes the cached rotation matrices from @ref orientation.
     *
     * Must be called after any direct write to @ref orientation to keep the two
     * cached matrices consistent with it. Sets @ref orientation_matrix to the
     * quaternion's local-to-world rotation and @ref inverse_orientation_matrix
     * to its transpose (valid inverse only while the quaternion stays unit).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept {
        orientation_matrix         = orientation.to_matrix3x3();
        inverse_orientation_matrix = transpose(orientation_matrix);
    }

    /**
     * @brief Transforms a point from local (body) space to world space.
     *
     * Applies the rotation followed by the translation: @p world_point =
     * orientation_matrix * @p local_point + translation. Out-parameter form to
     * avoid a temporary in hot device code.
     *
     * @param local_point  Point expressed in the body frame.
     * @param world_point  Receives the point in world space; may alias nothing
     *                     the caller still needs, but must not alias @p local_point.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Float3& local_point, Float3& world_point) const noexcept {
        atlas::rotate_translate(orientation_matrix, local_point, translation, world_point);
    }

    /**
     * @brief Transforms a point from world space into local (body) space.
     *
     * The exact inverse of @ref sync_to_world for a point: subtracts the
     * translation, then applies the inverse rotation, giving @p local_point =
     * inverse_orientation_matrix * (@p world_point - translation).
     *
     * @param world_point  Point expressed in world space.
     * @param local_point  Receives the point in the body frame.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Float3& world_point, Float3& local_point) const noexcept {
        atlas::rotate_subtract(inverse_orientation_matrix, world_point, translation, local_point);
    }

    /**
     * @brief Rotates a direction from local space to world space (no translation).
     *
     * Directions are translation-invariant, so only the rotation is applied.
     * The input is treated as a free vector; its length is preserved.
     *
     * @param local_dir  Direction in the body frame.
     * @param world_dir  Receives the direction in world space.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_world(const Float3& local_dir, Float3& world_dir) const noexcept {
        atlas::rotate(orientation_matrix, local_dir, world_dir);
    }

    /**
     * @brief Rotates a direction from world space to local space (no translation).
     *
     * Applies the inverse rotation only, the exact inverse of
     * @ref sync_dir_to_world.
     *
     * @param world_dir  Direction in world space.
     * @param local_dir  Receives the direction in the body frame.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_local(const Float3& world_dir, Float3& local_dir) const noexcept {
        atlas::rotate(inverse_orientation_matrix, world_dir, local_dir);
    }

    /**
     * @brief Transforms a whole ray from local space to world space.
     *
     * The origin is treated as a point (rotated and translated) while the
     * direction is treated as a free vector (rotated only), keeping the ray's
     * parameterization intact.
     *
     * @param local_ray  Ray in the body frame.
     * @param world_ray  Receives the ray in world space.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Ray& local_ray, Ray& world_ray) const noexcept {
        atlas::rotate_translate(orientation_matrix, local_ray.origin, translation, world_ray.origin);
        atlas::rotate(orientation_matrix, local_ray.direction, world_ray.direction);
    }

    /**
     * @brief Transforms a whole ray from world space into local space.
     *
     * Inverse of the ray overload of @ref sync_to_world: the origin is
     * un-translated and inverse-rotated, the direction inverse-rotated only.
     * Used by @ref Unit::trace to push a world-space query ray into a body's
     * local frame before intersecting its geometry.
     *
     * @param world_ray  Ray in world space.
     * @param local_ray  Receives the ray in the body frame.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Ray& world_ray, Ray& local_ray) const noexcept {
        atlas::rotate_subtract(inverse_orientation_matrix, world_ray.origin, translation, local_ray.origin);
        atlas::rotate(inverse_orientation_matrix, world_ray.direction, local_ray.direction);
    }

    /**
     * @brief Value-returning convenience overload of the point local-to-world transform.
     * @param local_point  Point in the body frame.
     * @return The point in world space.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    sync_to_world(const Float3& local_point) const noexcept {
        Float3 out;
        sync_to_world(local_point, out);
        return out;
    }

    /**
     * @brief Value-returning convenience overload of the point world-to-local transform.
     * @param world_point  Point in world space.
     * @return The point in the body frame.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    sync_to_local(const Float3& world_point) const noexcept {
        Float3 out;
        sync_to_local(world_point, out);
        return out;
    }

    /**
     * @brief Value-returning convenience overload of the direction local-to-world rotation.
     * @param local_dir  Direction in the body frame.
     * @return The direction in world space (length preserved).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    sync_dir_to_world(const Float3& local_dir) const noexcept {
        Float3 out;
        sync_dir_to_world(local_dir, out);
        return out;
    }

    /**
     * @brief Value-returning convenience overload of the direction world-to-local rotation.
     * @param world_dir  Direction in world space.
     * @return The direction in the body frame (length preserved).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    sync_dir_to_local(const Float3& world_dir) const noexcept {
        Float3 out;
        sync_dir_to_local(world_dir, out);
        return out;
    }

    /**
     * @brief Value-returning convenience overload of the ray local-to-world transform.
     * @param local_ray  Ray in the body frame.
     * @return The ray in world space.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Ray
    sync_to_world(const Ray& local_ray) const noexcept {
        Ray out;
        sync_to_world(local_ray, out);
        return out;
    }

    /**
     * @brief Value-returning convenience overload of the ray world-to-local transform.
     * @param world_ray  Ray in world space.
     * @return The ray in the body frame.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Ray
    sync_to_local(const Ray& world_ray) const noexcept {
        Ray out;
        sync_to_local(world_ray, out);
        return out;
    }

    /**
     * @brief Sets the translation only; leaves orientation and cached matrices untouched.
     *
     * The rotation cache does not depend on translation, so no rebuild is needed.
     *
     * @param translation_ New world-space body origin.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_translation(const Float3& translation_) noexcept {
        translation = translation_;
    }

    /**
     * @brief Sets the orientation and rebuilds the cached rotation matrices.
     *
     * @param orientation_ New body orientation; expected to be a unit quaternion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_orientation(const Quaternion& orientation_) noexcept {
        orientation = orientation_;
        rebuild_matrices();
    }

    /**
     * @brief Sets translation and orientation together, then rebuilds the matrices.
     *
     * @param translation_ New world-space body origin.
     * @param orientation_ New body orientation; expected to be a unit quaternion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_pose(const Float3& translation_, const Quaternion& orientation_) noexcept {
        translation = translation_;
        orientation = orientation_;
        rebuild_matrices();
    }
};

/**
 * @brief Host-only builder that validates inputs and constructs a @ref Sync.
 *
 * Follows the project builder convention: fluent `with_*` setters, a private
 * @ref validate that throws on bad input, and terminal @ref build /
 * @ref make_host_shared. The default configuration is the identity pose, so a
 * Sync can be built without calling any setter.
 */
class Sync::Builder final {
public:
    /// Constructs a builder pre-loaded with the identity pose.
    Builder() = default;

    /**
     * @brief Sets both the translation and orientation of the pose to build.
     *
     * @param translation_ World-space body origin.
     * @param orientation_ Body orientation; should be a unit quaternion (the zero
     *                     quaternion is rejected by @ref validate at build time).
     * @return `*this`, to allow call chaining.
     */
    ATLAS_HOST Builder&
    with_rigid_pose(const Float3& translation_, const Quaternion& orientation_) noexcept;

    /**
     * @brief Validates the configured pose and returns a value-type @ref Sync.
     *
     * @return A Sync with the configured translation/orientation and freshly
     *         built rotation matrices.
     * @throws std::runtime_error if the translation or orientation is non-finite,
     *         or if the orientation is the zero quaternion.
     */
    ATLAS_NODISCARD ATLAS_HOST Sync
    build() const;

    /**
     * @brief Builds a @ref Sync and wraps it in a host shared pointer.
     *
     * @return A @ref SyncHostPtr owning a heap-allocated copy of @ref build's result.
     * @throws std::runtime_error under the same conditions as @ref build.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Sync>
    make_host_shared() const;

private:
    /**
     * @brief Throws if the staged translation or orientation is unusable.
     *
     * @throws std::runtime_error when the translation is non-finite, the
     *         orientation is non-finite, or the orientation is the zero
     *         quaternion (which has no valid rotation matrix).
     */
    ATLAS_HOST void
    validate() const;

private:
    Float3 _translation = Float3(0.0f, 0.0f, 0.0f); ///< Staged translation; defaults to the origin.

    Quaternion _orientation = Quaternion(1.0f, 0.0f, 0.0f, 0.0f); ///< Staged orientation; defaults to identity.
};

/// Host-side shared-ownership handle to a @ref Sync.
using SyncHostPtr = atlas::host_shared_ptr<Sync>;

/// Device-side shared-ownership handle to a @ref Sync.
using SyncDevicePtr = atlas::device_shared_ptr<Sync>;

}