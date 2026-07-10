#pragma once

#include <atlas/collider/diffuse_sampling.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sampling/sampling.h>
#include <atlas/spatial/ray.h>
#include <atlas/unit/unit.h>

#include <optional>
#include <utility>

namespace atlas {

/**
 * @brief Collider leaf: an isothermal moving wall with specular/diffuse reflection.
 *
 * The only `Collider` leaf today. It owns a `Unit` (geometry + rigid motion) by
 * value and reflects fluid particles off that surface in the wall's moving
 * frame. "Isothermal" means the reflection preserves the incident speed up to a
 * scalar `restitution` factor rather than re-drawing speed from a wall
 * temperature: the model deliberately skips the thermal part and keeps only the
 * directional (specular vs. diffuse) split, in line with the engine's
 * simplicity-over-fine-detail philosophy.
 *
 * The type is trivially copyable (a `Unit` by value, three POD reflection
 * parameters, and a cached `AABB`), so it can be a `union` member of the
 * `DeviceVariant`-based `Collider` umbrella, stored in a `DeviceBuffer`, and
 * captured by value into a device lambda. Every hot-path method is
 * `ATLAS_ALL_DEVICE` so it runs identically on host and device.
 *
 * Satisfies `ConceptCollider` (`trace`/`collide`/`advance`/`bound`). The
 * per-particle loop lives in the caller (`System::advance`): the caller broad-
 * phases against `bound()`, calls `trace` to find the nearest wall hit, then
 * `collide` to apply it.
 */
class IsothermalCollider final {
public:
    /// Host-only fluent builder; the sole validated way to construct this leaf.
    class Builder;

public:
    /**
     * @brief Constructs an empty collider with a default `Unit` and unit
     *        reflection parameters.
     *
     * Required so `IsothermalCollider` can be a union member of `Collider`
     * (the umbrella default-constructs the active leaf). The resulting collider
     * has no meaningful geometry; use `Builder` to make a usable one.
     */
    IsothermalCollider() = default;

    /**
     * @brief Constructs a fully specified collider leaf.
     *
     * Caches the wall's world-space AABB from `unit.world_bound()` at
     * construction so the broad phase can query `bound()` without recomputing it
     * per particle. The cache is refreshed by `advance`.
     *
     * @param unit Wall geometry plus rigid motion; taken by value and moved in.
     * @param momentum_accommodation_coefficient Diffuse fraction in [0, 1]:
     *        0 = purely specular, 1 = purely diffuse. Validated by the builder.
     * @param restitution Non-negative speed retained after reflection
     *        (1 = elastic). Validated by the builder.
     * @param diffuse_sampling Hemisphere law for the diffuse branch.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    IsothermalCollider(Unit unit,
                       const float momentum_accommodation_coefficient,
                       const float restitution,
                       const DiffuseSampling diffuse_sampling) noexcept
        : _unit(std::move(unit))
        , _momentum_accommodation_coefficient(momentum_accommodation_coefficient)
        , _restitution(restitution)
        , _diffuse_sampling(diffuse_sampling)
        , _bound(_unit.world_bound()) {
    }

    /**
     * @brief Returns a fresh, empty `Builder` for this leaf. Host-only.
     * @return A default-constructed builder.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief The owned wall `Unit` (geometry and motion). Host and device.
     * @return Const reference to the internally owned unit.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const Unit&
    unit() const noexcept {
        return _unit;
    }

    /**
     * @brief The diffuse fraction of the reflection model, in [0, 1].
     * @return 0 for purely specular, 1 for purely diffuse reflection.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    momentum_accommodation_coefficient() const noexcept {
        return _momentum_accommodation_coefficient;
    }

    /**
     * @brief Cached world-space bounding box of the wall for the broad phase.
     *
     * Returns the AABB computed at construction / last `advance`. It is invalid
     * (see `AABB::is_valid`) for unbounded geometry such as an infinite plane; a
     * caller must treat an invalid bound as "cannot be culled" and always run the
     * narrow-phase `trace` (this is exactly what `System::advance` does).
     *
     * @return Const reference to the cached world AABB.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const AABB&
    bound() const noexcept {
        return _bound;
    }

    /**
     * @brief Advances the wall by one time step and refreshes the cached bound.
     *
     * Steps the owned `Unit`'s rigid motion by `dt`, then recomputes the cached
     * world AABB from the moved geometry so a subsequent broad phase stays
     * correct. The collider carries no particle state, so nothing else needs
     * invalidating. Runs on host or device.
     *
     * @param dt Time step in seconds to advance the wall's motion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    advance(const float dt) noexcept {
        _unit.update(dt);
        _bound = _unit.world_bound();
    }

    /**
     * @brief Narrow phase: sweeps the particle's path this step against the wall.
     *
     * Builds the world-space ray from `position` along `velocity` and traces it
     * against the owned unit, then rejects the hit unless it lies within the
     * distance the particle actually travels this step (`|velocity| * dt`). A
     * zero/near-zero sweep (particle at rest or `dt`≈0) returns an empty hit
     * without tracing. Purely read-only; the caller keeps the nearest hit across
     * all colliders and passes it to `collide`.
     *
     * @param position World-space particle position at the start of the step.
     * @param velocity World-space particle velocity (its magnitude sets the
     *        sweep length; the ray is built along its direction).
     * @param dt Time step in seconds; scales the sweep length.
     * @return A `HitSurface` with `is_intersecting == true` and world-space
     *         point/normal/distance on a hit within the sweep, otherwise a
     *         default (non-intersecting) `HitSurface`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Float3& position, const Float3& velocity, const float dt) const noexcept {
        const float speed        = velocity.length();
        const float sweep_length = speed * dt;

        if (sweep_length <= atlas::eps || speed <= atlas::eps) {
            return HitSurface {};
        }

        const atlas::Ray world_ray(position, velocity * dt);
        const HitSurface hit = _unit.trace(world_ray);

        if (!hit.is_intersecting || hit.distance > sweep_length) {
            return HitSurface {};
        }

        return hit;
    }

    /**
     * @brief Resolves a hit produced by `trace`, writing back the post-collision state.
     *
     * On a valid hit the particle is reflected in the wall's moving frame:
     * the incident velocity is shifted into the wall frame
     * (`velocity - wall_velocity`), reflected by `reflect`, then shifted back by
     * adding the wall's surface velocity — so a moving/rotating wall imparts
     * momentum. The position is snapped to the hit point and pushed off the
     * surface by `atlas::tol` along the outward normal to keep the particle from
     * re-intersecting the same face on the next step. A non-intersecting `hit`
     * is a no-op, leaving `position`/`velocity` untouched.
     *
     * @param hit The nearest hit chosen by the caller (from `trace`); must carry
     *        world-space `point` and `normal`.
     * @param position In/out world-space position; set to the pushed-off hit point.
     * @param velocity In/out world-space velocity; set to the reflected velocity.
     * @param dt Unused: the reflection is impulsive and time-independent; the
     *        parameter exists only to satisfy the `ConceptCollider` signature.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    collide(const HitSurface& hit, Float3& position, Float3& velocity, const float) const noexcept {
        if (!hit.is_intersecting) {
            return;
        }

        const Float3 wall_velocity     = _unit.surface_velocity(hit.point);
        const Float3 relative_incident = velocity - wall_velocity;

        position = hit.point + hit.normal * atlas::tol;
        velocity = reflect(relative_incident, hit.normal) + wall_velocity;
    }

    /**
     * @brief Computes the reflected velocity in the wall frame (specular/diffuse blend).
     *
     * Given the incident velocity relative to the wall and the surface normal,
     * returns the outgoing velocity, also in the wall frame:
     * - A degenerate incident (speed <= `atlas::tol`) reflects to zero.
     * - With `momentum_accommodation_coefficient <= 0` the result is purely
     *   specular: the mirror direction scaled by `incident_speed * restitution`.
     * - Otherwise it draws two hashed pseudo-random numbers (deterministic in the
     *   incident/normal geometry, so no RNG state is threaded through the device
     *   kernel), samples a diffuse direction by `_diffuse_sampling`, then a third
     *   hashed number decides per hit whether to emit the diffuse or the specular
     *   direction, with diffuse chosen with probability
     *   `momentum_accommodation_coefficient`. The chosen unit direction is scaled
     *   by `incident_speed * restitution`, so speed is conserved up to
     *   restitution regardless of branch — the "isothermal" simplification.
     *
     * @param incident Incident velocity in the wall's frame (magnitude = speed).
     * @param normal World-space surface normal at the hit point.
     * @return Outgoing velocity in the wall frame; add the wall velocity to
     *         return to world frame (done by `collide`).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    reflect(const Float3& incident, const Float3& normal) const noexcept {
        const float incident_speed = incident.length();

        if (incident_speed <= atlas::tol) {
            return Float3(0.0f, 0.0f, 0.0f);
        }

        const Float3 specular_unit = atlas::reflected(incident, normal).normalized();

        if (_momentum_accommodation_coefficient <= 0.0f) {
            return specular_unit * (incident_speed * _restitution);
        }

        const float u1 = atlas::sample_hashed_unit_interval(
            incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U1);

        const float u2 = atlas::sample_hashed_unit_interval(
            normal + incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U2);

        Float3 diffuse_dir {};

        if (_diffuse_sampling == DiffuseSampling::cosine_weighted) {
            diffuse_dir = atlas::sample_cosine_hemisphere(normal, u1, u2);
        } else {
            diffuse_dir = atlas::sample_uniform_hemisphere(normal, u1, u2);
        }

        const float mix = atlas::sample_hashed_unit_interval(
            incident + normal * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX,
            atlas::RANDOM_HASH_SALT_MIX);

        const Float3 out_unit = (mix < _momentum_accommodation_coefficient)
            ? diffuse_dir.normalized()
            : specular_unit;

        return out_unit * (incident_speed * _restitution);
    }

private:
    Unit _unit; ///< Owned wall geometry and rigid motion; source of hits and surface velocity.

    /// Diffuse fraction in [0, 1]: 0 = specular, 1 = fully diffuse. Enforced by the builder.
    float _momentum_accommodation_coefficient { 1.0f };

    float _restitution { 1.0f }; ///< Speed retained after reflection; >= 0, 1 = elastic.

    /// Hemisphere sampling law used for the diffuse reflection branch.
    DiffuseSampling _diffuse_sampling { DiffuseSampling::uniform };

    AABB _bound {}; ///< Cached world AABB of `_unit`; refreshed by `advance`, read by broad phase.
};

/**
 * @brief Fluent, validated builder for `IsothermalCollider`.
 *
 * Host-only. Collects the wall unit and reflection parameters via chained
 * `with_*` setters, then materializes a collider with `build()` (or a shared
 * pointer with `make_host_shared()`). The unit is required; the numeric
 * parameters default to unit (elastic, fully diffuse) and are range-checked by
 * `validate()` at build time. The builder is single-use: `build()` moves the
 * stored unit out and resets all fields, so a second `build()` on the same
 * instance throws for a missing unit rather than reusing stale state.
 */
class IsothermalCollider::Builder final {
public:
    /// Constructs an empty builder with no unit and unit-valued defaults.
    Builder() = default;

    /**
     * @brief Sets the wall unit (required). Takes ownership by value.
     * @param unit Wall geometry plus rigid motion; moved into the builder.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_unit(Unit unit);

    /**
     * @brief Sets the diffuse fraction. Validated to be finite and in [0, 1] at build.
     * @param momentum_accommodation_coefficient 0 = specular, 1 = fully diffuse.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_momentum_accommodation_coefficient(float momentum_accommodation_coefficient) noexcept;

    /**
     * @brief Sets the speed retained on reflection. Validated finite and >= 0 at build.
     * @param restitution 1 = elastic; <1 damps, >1 amplifies the reflected speed.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_restitution(float restitution) noexcept;

    /**
     * @brief Selects the hemisphere sampling law for diffuse reflection.
     * @param mode Cosine-weighted or uniform.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_diffuse_sampling(DiffuseSampling mode) noexcept;

    /**
     * @brief Validates the collected fields and builds the collider by value.
     *
     * Runs `validate()`, then constructs the collider by moving the unit out.
     * Consumes the builder's state (all fields reset to defaults) so the same
     * instance cannot silently build twice from stale data.
     *
     * @return The constructed `IsothermalCollider`.
     * @throws std::runtime_error if the unit is unset or a parameter is out of range.
     */
    ATLAS_NODISCARD ATLAS_HOST IsothermalCollider
    build();

    /**
     * @brief Builds the collider and wraps it in a host shared pointer.
     * @return A `host_shared_ptr` owning the freshly built collider.
     * @throws std::runtime_error under the same conditions as `build()`.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<IsothermalCollider>
    make_host_shared();

private:
    /**
     * @brief Throws unless every collected field is present and in range.
     * @throws std::runtime_error if the unit is unset, the accommodation
     *         coefficient is not finite or outside [0, 1], or restitution is not
     *         finite or negative.
     */
    ATLAS_HOST void
    validate() const;

private:
    std::optional<Unit> _unit; ///< Required wall unit; empty until `with_unit`.

    /// Pending diffuse fraction; range-checked by `validate()`.
    float _momentum_accommodation_coefficient { 1.0f };

    float _restitution { 1.0f }; ///< Pending restitution; range-checked by `validate()`.

    /// Pending diffuse sampling law.
    DiffuseSampling _diffuse_sampling { DiffuseSampling::uniform };
};

/// Host-side shared-ownership handle to an `IsothermalCollider`.
using IsothermalColliderHostPtr = atlas::host_shared_ptr<IsothermalCollider>;

/// Device-side shared-ownership handle to an `IsothermalCollider`.
using IsothermalColliderDevicePtr = atlas::device_shared_ptr<IsothermalCollider>;

}