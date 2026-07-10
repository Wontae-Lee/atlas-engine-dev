#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/ray.h>
#include <atlas/unit/unit.h>

#include <optional>
#include <utility>

namespace atlas {

/**
 * @brief Sink leaf that despawns a particle about to cross a boundary this step.
 *
 * Where @ref SurfaceSink and @ref VolumeSink test the particle's current point,
 * a @ref TracingSink is swept: it casts a ray from the particle along its velocity
 * and removes the particle if the boundary is hit within the distance it would
 * travel this step (`speed * dt`). This catches fast particles that would tunnel
 * through a thin surface between two point-sample steps, so it needs no tolerance
 * band. It owns only the boundary @ref Unit.
 *
 * The type is trivially copyable so it can be stored inline in a @ref Sink
 * (`DeviceVariant`), placed in a `DeviceBuffer<Sink>`, and captured by value in a
 * device lambda. The per-particle iteration and compaction of despawned particles
 * are caller concerns and live outside this leaf.
 *
 * @see ConceptSink for the leaf contract this satisfies.
 */
class TracingSink final {
public:
    /// Host-side fluent builder; see @ref TracingSink::Builder.
    class Builder;

public:
    /// Default-constructs an empty sink (identity unit); use @ref Builder to configure.
    TracingSink() = default;

    /**
     * @brief Constructs a ready-to-use sink from an owned unit.
     *
     * `explicit` to avoid an implicit @ref Unit-to-sink conversion.
     *
     * @param unit Boundary geometry and pose; moved into @ref _unit.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit TracingSink(Unit unit) noexcept
        : _unit(std::move(unit)) {
    }

    /**
     * @brief Returns a fresh host-side @ref Builder for this leaf.
     * @return A default-constructed builder; host-only.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Read-only access to the owned boundary unit.
     * @return Const reference to @ref _unit; valid for the lifetime of this sink.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const Unit&
    unit() const noexcept {
        return _unit;
    }

    /**
     * @brief Advances the owned unit's pose by one time step.
     *
     * Delegates to `Unit::update`; non-positive @p dt is ignored by the unit.
     *
     * @param dt Time step in seconds.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    advance(const float dt) noexcept {
        _unit.update(dt);
    }

    /**
     * @brief Tests whether the particle's motion this step crosses the boundary.
     *
     * Builds a ray at @p position aimed along @p velocity and traces it against the
     * unit (`Unit::trace` handles the world<->local transform). The particle is
     * removed when the ray hits and the hit distance lies in `[0, speed * dt]`, i.e.
     * the surface is reachable within this step's travel.
     *
     * A non-positive @p dt or a zero-speed (stationary) particle can travel no
     * distance, so the trace is skipped and `false` is returned; this also guards
     * against building a ray from a degenerate zero-length direction. The `!(x > 0)`
     * form additionally rejects NaN.
     *
     * @param position Particle position in world space (ray origin).
     * @param velocity Particle velocity in world space; its length is the speed and
     *                 its direction the ray direction.
     * @param dt       Time step in seconds.
     * @return `true` if the boundary is crossed within this step (should despawn),
     *         else `false`.
     * @note Device-callable and const; performs no removal itself.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    despawn(const Float3& position, const Float3& velocity, const float dt) const noexcept {
        const float speed = velocity.length();

        if (!(dt > 0.0f) || !(speed > 0.0f)) {
            return false;
        }

        const HitSurface hit = _unit.trace(atlas::Ray(position, velocity));

        return hit.is_intersecting && hit.distance >= 0.0f && hit.distance <= speed * dt;
    }

private:
    /// Owned boundary geometry and its (possibly moving) pose.
    Unit _unit;
};

/**
 * @brief Host-only fluent builder that validates and constructs a @ref TracingSink.
 *
 * Setters return `*this` for chaining. The @ref _unit is the only field and is
 * required. @ref build validates, then moves it into a @ref TracingSink and resets
 * the builder so it can be reused.
 */
class TracingSink::Builder final {
public:
    /// Constructs an empty builder with no unit.
    Builder() = default;

    /**
     * @brief Sets the boundary unit (required).
     * @param unit Geometry and pose; moved into the builder.
     * @return `*this` for chaining.
     */
    ATLAS_HOST Builder&
    with_unit(Unit unit);

    /**
     * @brief Validates and produces a configured @ref TracingSink.
     *
     * On success the stored unit is moved out and the builder is cleared so it may
     * be reused for another sink.
     *
     * @return The constructed sink.
     * @throws std::runtime_error if the unit is unset (see @ref validate).
     */
    ATLAS_NODISCARD ATLAS_HOST TracingSink
    build();

    /**
     * @brief Builds the sink and wraps it in a host-side shared pointer.
     * @return `host_shared_ptr` owning a freshly built @ref TracingSink.
     * @throws std::runtime_error if the unit is unset.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<TracingSink>
    make_host_shared();

private:
    /**
     * @brief Throws if the accumulated configuration is invalid.
     * @throws std::runtime_error if the unit is unset.
     */
    ATLAS_HOST void
    validate() const;

private:
    /// Pending boundary unit; must be set before @ref build.
    std::optional<Unit> _unit;
};

/// Host-side shared owner of a @ref TracingSink.
using TracingSinkHostPtr = atlas::host_shared_ptr<TracingSink>;

/// Device-side shared owner of a @ref TracingSink.
using TracingSinkDevicePtr = atlas::device_shared_ptr<TracingSink>;

}