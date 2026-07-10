#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>
#include <atlas/sync/sync.h>

#include <optional>
#include <utility>

namespace atlas {

/**
 * @brief A placeable, optionally moving rigid body: a Geometry plus its pose and kinematics.
 *
 * A Unit is the world-space instance of a shape. It couples a @ref Geometry
 * (the collision surface, expressed in a fixed local frame) with a @ref Sync
 * (its local-to-world pose) and an optional first-order kinematic state
 * (linear/angular velocity and acceleration). All accessors and the integrator
 * are `__host__ __device__` and the whole object is trivially movable, so a Unit
 * can be handed to device code that ray-traces against boundaries or samples
 * their surface velocity.
 *
 * The kinematic fields are `std::optional`: a body with no velocity is static
 * and @ref update is a no-op for it. The four fields are kept in a canonical
 * pairing by @ref canonicalize_kinematics — if an acceleration is supplied
 * without its matching velocity (or vice versa) the missing partner is
 * defaulted to zero, so @ref update never has to test acceleration without a
 * velocity to integrate into.
 *
 * @note Rotation and translation live in the embedded @ref Sync; @ref move and
 *       @ref rotate mutate that pose in place (the rotate path also refreshes
 *       Sync's cached matrices). Geometry is never modified — motion is entirely
 *       carried by the pose.
 */
class Unit final {
public:
    /// Host-side fluent builder that validates and constructs a Unit.
    class Builder;

public:
    /// Constructs an empty static Unit: default (empty) geometry, identity pose, no motion.
    Unit() = default;

    /// Trivial destructor; the Unit owns its members by value and nothing external.
    ~Unit() = default;

    /**
     * @brief Constructs a static Unit from a geometry and a pose, with no kinematics.
     *
     * The resulting body has no velocity or acceleration, so @ref update leaves
     * it fixed. Both arguments are moved in.
     *
     * @param geometry The collision shape, in its local frame.
     * @param sync     The local-to-world pose to place it at.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Unit(Geometry geometry,
         Sync sync) noexcept
        : _geometry(std::move(geometry))
        , _sync(std::move(sync)) {
    }

    /**
     * @brief Constructs a Unit with an explicit (optional) kinematic state.
     *
     * The four optional kinematic arguments are first passed through
     * @ref canonicalize_kinematics, which fills in a zero partner for any lone
     * velocity/acceleration so the pairs are consistent, and only then stored.
     *
     * @param geometry             The collision shape, in its local frame.
     * @param sync                 The local-to-world pose to place it at.
     * @param velocity             Linear velocity in world units/second, or none.
     * @param acceleration         Linear acceleration in world units/second^2, or none.
     * @param angular_velocity     Angular velocity as an axis scaled by rad/second, or none.
     * @param angular_acceleration Angular acceleration as an axis scaled by rad/second^2, or none.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Unit(Geometry geometry,
         Sync sync,
         std::optional<Float3> velocity,
         std::optional<Float3> acceleration,
         std::optional<Float3> angular_velocity,
         std::optional<Float3> angular_acceleration) noexcept
        : _geometry(std::move(geometry))
        , _sync(std::move(sync)) {
        Unit::canonicalize_kinematics(
            velocity,
            acceleration,
            angular_velocity,
            angular_acceleration);

        _velocity             = std::move(velocity);
        _acceleration         = std::move(acceleration);
        _angular_velocity     = std::move(angular_velocity);
        _angular_acceleration = std::move(angular_acceleration);
    }

    /**
     * @brief Returns a fresh host-side @ref Builder for constructing a Unit.
     * @return A default-initialized Builder with no geometry or sync yet set.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Replaces the collision geometry, leaving pose and kinematics unchanged.
     * @param geometry The new local-frame shape; moved in.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_geometry(Geometry geometry) noexcept {
        _geometry = std::move(geometry);
    }

    /**
     * @brief Replaces the pose, leaving geometry and kinematics unchanged.
     * @param sync The new local-to-world pose; moved in.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_sync(Sync sync) noexcept {
        _sync = std::move(sync);
    }

    /**
     * @brief Advances the pose by one time step using the stored kinematic state.
     *
     * Semi-implicit (symplectic) Euler: acceleration is integrated into velocity
     * first, then the updated velocity displaces the pose. Linear and angular
     * channels are handled independently and each is skipped entirely when its
     * velocity is absent (a static body). For the angular channel the angular
     * velocity's magnitude is the rotation rate and its direction the axis; a
     * zero-magnitude angular velocity applies no rotation to avoid normalizing a
     * zero axis.
     *
     * @param dt Time step in seconds. Non-positive @p dt (including NaN, which
     *           fails the `> 0` test) is ignored and the pose is left untouched.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    update(const float dt) noexcept {
        if (!(dt > 0.0f)) return;

        if (_velocity.has_value()) {

            if (_acceleration.has_value()) {
                *_velocity += (*_acceleration) * dt;
            }

            move((*_velocity) * dt);
        }

        if (_angular_velocity.has_value()) {

            if (_angular_acceleration.has_value()) {
                *_angular_velocity += (*_angular_acceleration) * dt;
            }

            const float omega = _angular_velocity->length();

            if (omega > 0.0f) {
                rotate(*_angular_velocity, omega * dt);
            }
        }
    }

    /**
     * @brief Translates the body in world space by @p delta.
     *
     * Adds @p delta directly to the pose translation; orientation and the cached
     * rotation matrices are untouched.
     *
     * @param delta World-space displacement to add to the current position.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    move(const Float3& delta) noexcept {
        _sync.translation += delta;
    }

    /**
     * @brief Rotates the body about a world-space axis by a given angle.
     *
     * Composes the incremental rotation on the left of the current orientation
     * (world-frame rotation) and renormalizes to counter drift, then rebuilds
     * the pose's cached matrices. A zero-length axis is a no-op, guarding against
     * building a rotation from a degenerate axis.
     *
     * @param axis      Rotation axis in world space; need not be unit length (it
     *                  is normalized internally) but must be non-zero to have effect.
     * @param angle_rad Rotation angle in radians.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rotate(const Float3& axis, const float angle_rad) noexcept {
        const float axis_len2 = axis.length_squared();

        if (axis_len2 <= 0.0f) return;

        const Float3 normalized_axis = atlas::normalized_or(
            axis,
            Float3(0.0f, 0.0f, 0.0f));

        const Quaternion rotation = Quaternion::from_axis_angle(normalized_axis, angle_rad);

        // Left-multiply: the new rotation is applied in the world frame, and the
        // result is renormalized so repeated steps do not accumulate scale drift.
        _sync.orientation = (rotation * _sync.orientation).normalized();

        _sync.rebuild_matrices();
    }

    /**
     * @brief Returns the collision geometry in its local frame.
     * @return Const reference to the embedded @ref Geometry.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const Geometry&
    geometry() const noexcept {
        return _geometry;
    }

    /**
     * @brief Returns the local-to-world pose of this body.
     * @return Const reference to the embedded @ref Sync.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const Sync&
    sync() const noexcept {
        return _sync;
    }

    /**
     * @brief Intersects a world-space ray against this body and returns the hit in world space.
     *
     * The query ray is pulled into the body's local frame, intersected against
     * the local geometry, and — only on a hit — the resulting point and normal
     * are pushed back out to world space (the normal as a direction, so it stays
     * a rotation with no translation). The hit distance is left as reported by
     * the geometry; it is a valid world-space distance because the pose is rigid
     * (rotation preserves lengths).
     *
     * @param world_ray The query ray in world space.
     * @return A @ref HitSurface; `is_intersecting` is false with default fields
     *         when the ray misses the geometry.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& world_ray) const noexcept {
        const Ray local_ray = _sync.sync_to_local(world_ray);
        HitSurface hit      = _geometry.trace(local_ray);

        if (hit.is_intersecting) {
            hit.point  = _sync.sync_to_world(hit.point);
            hit.normal = _sync.sync_dir_to_world(hit.normal);
        }

        return hit;
    }

    /**
     * @brief Computes the world-space velocity of a point on the body's surface.
     *
     * Sums the linear velocity and the rotational contribution omega x r, where
     * r is the vector from the body origin (the pose translation) to the query
     * point. Absent kinematic components contribute zero, so a static body yields
     * the zero vector. Used by colliders to give reflected particles the wall's
     * local surface speed.
     *
     * @param surface_point A point on the body surface, in world space.
     * @return The instantaneous world-space velocity of that material point.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    surface_velocity(const Float3& surface_point) const noexcept {
        Float3 velocity(0.0f, 0.0f, 0.0f);

        if (_velocity.has_value()) {
            velocity += *_velocity;
        }

        if (_angular_velocity.has_value()) {
            const Float3 radius = surface_point - _sync.translation;
            velocity += atlas::cross(*_angular_velocity, radius);
        }

        return velocity;
    }

    /**
     * @brief Computes a world-space axis-aligned bound enclosing the posed geometry.
     *
     * Transforms all eight corners of the geometry's local bound into world
     * space and merges them into a fresh AABB. This is a conservative bound of
     * the rotated box (it may be larger than the tightest possible bound) but is
     * always valid. If the local geometry has no valid bound, an empty (reset)
     * AABB is returned.
     *
     * @return A world-space @ref AABB enclosing the body, or an empty AABB when
     *         the geometry has no valid local bound.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    world_bound() const noexcept {
        const AABB local_bound = _geometry.bound();

        AABB transformed {};

        if (!local_bound.is_valid()) {
            return transformed;
        }

        for (int corner = 0; corner < 8; ++corner) {
            transformed.merge(_sync.sync_to_world(local_bound.corner(static_cast<std::size_t>(corner))));
        }

        return transformed;
    }

    /**
     * @brief Returns the linear velocity, if any.
     * @return Const reference to the optional world-space linear velocity (units/second).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const std::optional<Float3>&
    velocity() const noexcept {
        return _velocity;
    }

    /**
     * @brief Returns the linear acceleration, if any.
     * @return Const reference to the optional world-space linear acceleration (units/second^2).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const std::optional<Float3>&
    acceleration() const noexcept {
        return _acceleration;
    }

    /**
     * @brief Returns the angular velocity, if any.
     * @return Const reference to the optional angular velocity (axis scaled by rad/second).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const std::optional<Float3>&
    angular_velocity() const noexcept {
        return _angular_velocity;
    }

    /**
     * @brief Returns the angular acceleration, if any.
     * @return Const reference to the optional angular acceleration (axis scaled by rad/second^2).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const std::optional<Float3>&
    angular_acceleration() const noexcept {
        return _angular_acceleration;
    }

    /**
     * @brief Reports whether the body carries any motion.
     *
     * @return true if a linear or angular velocity is present, meaning
     *         @ref update can change the pose; false for a purely static body.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    dynamic() const noexcept {
        return _velocity.has_value() || _angular_velocity.has_value();
    }

private:
    /**
     * @brief Fills in zero partners so each velocity/acceleration pair is consistent.
     *
     * Semi-implicit Euler integration reads acceleration only through its paired
     * velocity, and reports motion via velocity presence. To keep those
     * invariants simple, this normalizes the two channels: if an acceleration is
     * present without its velocity (or a velocity without its acceleration), the
     * missing member is set to zero. Applied to both the linear and angular
     * pairs. Values that are already present are left untouched.
     *
     * @param velocity             In/out linear velocity optional.
     * @param acceleration         In/out linear acceleration optional.
     * @param angular_velocity     In/out angular velocity optional.
     * @param angular_acceleration In/out angular acceleration optional.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    canonicalize_kinematics(std::optional<Float3>& velocity,
                            std::optional<Float3>& acceleration,
                            std::optional<Float3>& angular_velocity,
                            std::optional<Float3>& angular_acceleration) noexcept {
        if (acceleration.has_value() && !velocity.has_value()) {
            velocity = Float3(0.0f, 0.0f, 0.0f);
        }

        if (velocity.has_value() && !acceleration.has_value()) {
            acceleration = Float3(0.0f, 0.0f, 0.0f);
        }

        if (angular_acceleration.has_value() && !angular_velocity.has_value()) {
            angular_velocity = Float3(0.0f, 0.0f, 0.0f);
        }

        if (angular_velocity.has_value() && !angular_acceleration.has_value()) {
            angular_acceleration = Float3(0.0f, 0.0f, 0.0f);
        }
    }

    /// Grants the builder access to move members directly into a bare Unit.
    friend class Builder;

private:
    Geometry _geometry; ///< Collision shape in its local frame.

    Sync _sync; ///< Local-to-world pose of the body.

    std::optional<Float3> _velocity; ///< Linear velocity (units/s), or empty when static.

    std::optional<Float3> _acceleration; ///< Linear acceleration (units/s^2), or empty.

    std::optional<Float3> _angular_velocity; ///< Angular velocity (axis * rad/s), or empty.

    std::optional<Float3> _angular_acceleration; ///< Angular acceleration (axis * rad/s^2), or empty.
};

/**
 * @brief Host-only builder that validates and assembles a @ref Unit.
 *
 * Geometry and sync are required; the four kinematic fields are optional. The
 * sync is supplied as a @ref SyncHostPtr and dereferenced into a value copy, so
 * the built Unit owns its pose independently of the shared handle. Kinematic
 * inputs are canonicalized (missing partners zero-filled) at @ref build time,
 * matching the value constructor's behavior. Consuming a builder via @ref build
 * resets its staged state, so it must be reconfigured before reuse.
 */
class Unit::Builder final {
public:
    /// Constructs an empty builder with no geometry, sync, or kinematics set.
    Builder() = default;

    /**
     * @brief Sets the required collision geometry.
     * @param geometry Local-frame shape; copied into the builder.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_geometry(const Geometry& geometry);

    /**
     * @brief Sets the required pose from a shared @ref Sync handle.
     * @param sync Non-null handle whose pointee is copied into the builder.
     * @return `*this`, for chaining.
     * @throws std::runtime_error if @p sync is null.
     */
    ATLAS_HOST Builder&
    with_sync(const SyncHostPtr& sync);

    /**
     * @brief Sets the linear velocity.
     * @param v World-space velocity in units/second.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_velocity(const Float3& v) noexcept;

    /**
     * @brief Sets the linear acceleration.
     *
     * @param a World-space acceleration in units/second^2.
     * @return `*this`, for chaining.
     * @note @ref validate rejects an acceleration set without a velocity; the
     *       canonicalization at build time only zero-fills a velocity when an
     *       acceleration is present, so supply a velocity as well.
     */
    ATLAS_HOST Builder&
    with_acceleration(const Float3& a) noexcept;

    /**
     * @brief Sets the angular velocity.
     * @param w Angular velocity as a world-space axis scaled by rad/second.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_angular_velocity(const Float3& w) noexcept;

    /**
     * @brief Sets the angular acceleration.
     *
     * @param alpha Angular acceleration as a world-space axis scaled by rad/second^2.
     * @return `*this`, for chaining.
     * @note @ref validate rejects an angular acceleration set without an angular
     *       velocity.
     */
    ATLAS_HOST Builder&
    with_angular_acceleration(const Float3& alpha) noexcept;

    /**
     * @brief Validates the staged fields, canonicalizes kinematics, and builds the Unit.
     *
     * Moves the geometry, pose, and (zero-filled) kinematic optionals into a new
     * Unit and then resets the builder's staged state. Not const because it
     * consumes the builder.
     *
     * @return The assembled @ref Unit.
     * @throws std::runtime_error if geometry or sync is unset, or if an
     *         acceleration is present without its matching velocity.
     */
    ATLAS_NODISCARD ATLAS_HOST Unit
    build();

    /**
     * @brief Builds a @ref Unit and wraps it in a host shared pointer.
     * @return A host shared handle owning the built Unit.
     * @throws std::runtime_error under the same conditions as @ref build.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Unit>
    make_host_shared();

private:
    /**
     * @brief Throws unless the staged configuration can produce a valid Unit.
     *
     * @throws std::runtime_error when geometry is unset, sync is unset, a linear
     *         acceleration is present without a linear velocity, or an angular
     *         acceleration is present without an angular velocity.
     */
    ATLAS_HOST void
    validate() const;

private:
    std::optional<Geometry> _geometry; ///< Staged geometry; required before build.

    std::optional<Sync> _sync; ///< Staged pose; required before build.

    std::optional<Float3> _velocity; ///< Staged linear velocity, or unset.

    std::optional<Float3> _acceleration; ///< Staged linear acceleration, or unset.

    std::optional<Float3> _angular_velocity; ///< Staged angular velocity, or unset.

    std::optional<Float3> _angular_acceleration; ///< Staged angular acceleration, or unset.
};

}