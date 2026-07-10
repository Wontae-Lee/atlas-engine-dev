#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>

#include <optional>
#include <utility>

namespace atlas {

/**
 * @brief Sink leaf that despawns a particle found inside a boundary unit's volume.
 *
 * A @ref VolumeSink owns one @ref Unit and a non-negative @ref _tolerance. Unlike
 * @ref SurfaceSink, which fires on a thin surface band, this leaf fires whenever
 * the particle is anywhere within the solid interior of the geometry (grown by
 * @ref _tolerance). It models a solid absorbing region: particles that penetrate
 * the body are removed.
 *
 * The type is trivially copyable so it can be stored inline in a @ref Sink
 * (`DeviceVariant`), placed in a `DeviceBuffer<Sink>`, and captured by value in a
 * device lambda. The per-particle iteration and compaction of despawned particles
 * are caller concerns and live outside this leaf.
 *
 * @see ConceptSink for the leaf contract this satisfies.
 */
class VolumeSink final {
public:
    /// Host-side fluent builder; see @ref VolumeSink::Builder.
    class Builder;

public:
    /// Default-constructs an empty sink (identity unit, zero tolerance); use @ref Builder to configure.
    VolumeSink() = default;

    /**
     * @brief Constructs a ready-to-use sink from an owned unit and tolerance.
     *
     * @param unit      Boundary geometry and pose; moved into @ref _unit.
     * @param tolerance Distance, in world units, by which the interior test is
     *                  widened outward from the surface. Expected finite and
     *                  non-negative (the builder enforces this); not re-validated
     *                  here so the constructor stays device-callable.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    VolumeSink(Unit unit, const float tolerance) noexcept
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
     * Delegates to `Unit::update`; non-positive @p dt is ignored by the unit.
     *
     * @param dt Time step in seconds.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    advance(const float dt) noexcept {
        _unit.update(dt);
    }

    /**
     * @brief Tests whether a particle should be removed for being inside the volume.
     *
     * Transforms @p position into the unit's local frame, then asks the geometry
     * whether the local point lies inside the body (expanded by @ref _tolerance).
     * Velocity and time step are unused for this static containment test and are
     * intentionally left unnamed.
     *
     * @param position Particle position in world space.
     * @return `true` if the particle is inside the volume (should despawn), else `false`.
     * @note Device-callable and const; performs no removal itself.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    despawn(const Float3& position, const Float3&, const float) const noexcept {
        const Float3 local = _unit.sync().sync_to_local(position);
        return _unit.geometry().is_inside(local, _tolerance);
    }

private:
    /// Owned boundary geometry and its (possibly moving) pose.
    Unit _unit;

    /// Outward widening of the interior test, in world units; non-negative.
    float _tolerance { 0.0f };
};

/**
 * @brief Host-only fluent builder that validates and constructs a @ref VolumeSink.
 *
 * Setters return `*this` for chaining. The @ref _unit is required; @ref _tolerance
 * defaults to `0.0f`. @ref build validates, then moves the accumulated state into a
 * @ref VolumeSink and resets the builder so it can be reused.
 */
class VolumeSink::Builder final {
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
     * @brief Sets the interior-test tolerance.
     * @param tolerance Finite, non-negative widening in world units.
     * @return `*this` for chaining.
     */
    ATLAS_HOST Builder&
    with_tolerance(float tolerance) noexcept;

    /**
     * @brief Validates and produces a configured @ref VolumeSink.
     *
     * On success the builder's stored state is consumed (unit moved out, tolerance
     * reset to `0.0f`) so the builder may be reused for another sink.
     *
     * @return The constructed sink.
     * @throws std::runtime_error if the unit is unset, or the tolerance is
     *         non-finite or negative (see @ref validate).
     */
    ATLAS_NODISCARD ATLAS_HOST VolumeSink
    build();

    /**
     * @brief Builds the sink and wraps it in a host-side shared pointer.
     * @return `host_shared_ptr` owning a freshly built @ref VolumeSink.
     * @throws std::runtime_error under the same conditions as @ref build.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<VolumeSink>
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

    /// Pending interior tolerance; finite and non-negative.
    float _tolerance { 0.0f };
};

/// Host-side shared owner of a @ref VolumeSink.
using VolumeSinkHostPtr = atlas::host_shared_ptr<VolumeSink>;

/// Device-side shared owner of a @ref VolumeSink.
using VolumeSinkDevicePtr = atlas::device_shared_ptr<VolumeSink>;

}