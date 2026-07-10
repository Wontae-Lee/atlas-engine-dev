#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>

#include <optional>
#include <utility>

namespace atlas {

/**
 * @brief Sink leaf that despawns a particle sitting on a boundary unit's surface.
 *
 * A @ref SurfaceSink owns one @ref Unit (its moving geometry plus pose) and a
 * non-negative @ref _tolerance thickness. @ref despawn transforms the particle
 * position into the unit's local frame and reports whether it lies within
 * @ref _tolerance of the geometry's surface. This models an absorbing thin wall:
 * particles that reach the surface are removed.
 *
 * The type is trivially copyable so it can be stored inline in a @ref Sink
 * (`DeviceVariant`), placed in a `DeviceBuffer<Sink>`, and captured by value in a
 * device lambda; construction and both hot-path methods are `ATLAS_ALL_DEVICE`.
 * The per-particle iteration and the actual compaction of despawned particles are
 * caller concerns and live outside this leaf.
 *
 * @see ConceptSink for the leaf contract this satisfies.
 */
class SurfaceSink final {
public:
    /// Host-side fluent builder; see @ref SurfaceSink::Builder.
    class Builder;

public:
    /// Default-constructs an empty sink (identity unit, zero tolerance); use @ref Builder to configure.
    SurfaceSink() = default;

    /**
     * @brief Constructs a ready-to-use sink from an owned unit and tolerance.
     *
     * @param unit      Boundary geometry and pose; moved into @ref _unit.
     * @param tolerance Half-thickness band, in world units, around the surface
     *                  within which a particle counts as "on" it. Expected to be
     *                  finite and non-negative (the builder enforces this); not
     *                  re-validated here so the constructor stays device-callable.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SurfaceSink(Unit unit, const float tolerance) noexcept
        : _unit(std::move(unit))
        , _tolerance(tolerance) {
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
     * Delegates to `Unit::update`, so a moving/rotating boundary tracks the
     * simulation clock. Non-positive @p dt is ignored by the unit.
     *
     * @param dt Time step in seconds.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    advance(const float dt) noexcept {
        _unit.update(dt);
    }

    /**
     * @brief Tests whether a particle should be removed for touching the surface.
     *
     * Transforms @p position into the unit's local frame via the cached sync
     * transform, then asks the geometry whether the local point lies within
     * @ref _tolerance of its surface. Velocity and time step are unused for this
     * static point test and are intentionally left unnamed.
     *
     * @param position Particle position in world space.
     * @return `true` if the particle is on the surface (should despawn), else `false`.
     * @note Device-callable and const; performs no removal itself.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    despawn(const Float3& position, const Float3&, const float) const noexcept {
        const Float3 local = _unit.sync().sync_to_local(position);
        return _unit.geometry().is_on_surface(local, _tolerance);
    }

private:
    /// Owned boundary geometry and its (possibly moving) pose.
    Unit _unit;

    /// Half-thickness band around the surface, in world units; non-negative.
    float _tolerance { 0.0f };
};

/**
 * @brief Host-only fluent builder that validates and constructs a @ref SurfaceSink.
 *
 * Setters return `*this` for chaining. The @ref _unit is required; @ref _tolerance
 * defaults to `0.0f`. @ref build validates, then moves the accumulated state into a
 * @ref SurfaceSink and resets the builder so it can be reused.
 */
class SurfaceSink::Builder final {
public:
    /// Constructs an empty builder with no unit and zero tolerance.
    Builder() = default;

    /**
     * @brief Sets the boundary unit (required).
     * @param unit Geometry and pose; moved into the builder.
     * @return `*this` for chaining.
     */
    ATLAS_HOST Builder&
    with_unit(Unit unit);

    /**
     * @brief Sets the surface half-thickness tolerance.
     * @param tolerance Finite, non-negative band width in world units.
     * @return `*this` for chaining.
     */
    ATLAS_HOST Builder&
    with_tolerance(float tolerance) noexcept;

    /**
     * @brief Validates and produces a configured @ref SurfaceSink.
     *
     * On success the builder's stored state is consumed (unit moved out, tolerance
     * reset to `0.0f`) so the builder may be reused for another sink.
     *
     * @return The constructed sink.
     * @throws std::runtime_error if the unit is unset, or the tolerance is
     *         non-finite or negative (see @ref validate).
     */
    ATLAS_NODISCARD ATLAS_HOST SurfaceSink
    build();

    /**
     * @brief Builds the sink and wraps it in a host-side shared pointer.
     * @return `host_shared_ptr` owning a freshly built @ref SurfaceSink.
     * @throws std::runtime_error under the same conditions as @ref build.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<SurfaceSink>
    make_host_shared();

private:
    /**
     * @brief Throws if the accumulated configuration is invalid.
     * @throws std::runtime_error if the unit is unset, or the tolerance is
     *         non-finite or negative.
     */
    ATLAS_HOST void
    validate() const;

private:
    /// Pending boundary unit; must be set before @ref build.
    std::optional<Unit> _unit;

    /// Pending surface tolerance; finite and non-negative.
    float _tolerance { 0.0f };
};

/// Host-side shared owner of a @ref SurfaceSink.
using SurfaceSinkHostPtr = atlas::host_shared_ptr<SurfaceSink>;

/// Device-side shared owner of a @ref SurfaceSink.
using SurfaceSinkDevicePtr = atlas::device_shared_ptr<SurfaceSink>;

}