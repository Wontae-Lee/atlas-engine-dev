#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace atlas {

/**
 * @brief Source leaf that emits particles from the **surface shell** of a `Unit`.
 *
 * At construction the leaf grid-samples the unit geometry's bounding box at
 * `spacing` intervals and keeps every sample that lies on the surface within
 * `tolerance`, storing those accepted points — in the unit's **local** frame —
 * into a `DeviceBuffer<Float3>` cache. The expensive geometry query runs exactly
 * once here; each later `spawn` only transforms the cached local points to world
 * space with a device kernel.
 *
 * Because it owns a `DeviceBuffer` (a `thrust::device_vector`, whose copy and
 * destructor are host-only), this leaf is move-only and is stored by the `Source`
 * umbrella through a `HostVariant` rather than a device-capable `DeviceVariant`.
 *
 * @note Satisfies `ConceptSource` via `spawn` and `advance`.
 * @see Source, VolumeSource
 */
class SurfaceSource final {
public:
    /// Host-side fluent builder that validates inputs and constructs the leaf.
    class Builder;

public:
    /**
     * @brief Constructs an empty source with a default `Unit` and no cached points.
     *
     * Needed so the leaf can be a union member of `Source`; a default-constructed
     * source has an empty cache and every `spawn` returns 0. Use `Builder` /
     * `builder()` to make a usable source.
     */
    SurfaceSource() = default;

    /**
     * @brief Constructs the source and builds its local surface-point cache.
     *
     * Takes ownership of @p unit, then grid-samples the unit geometry's bounding
     * box at @p spacing and caches every sample point for which
     * `geometry().is_on_surface(point, tolerance)` holds. If the geometry's bound
     * is invalid or @p spacing is not positive, the cache is left empty (no throw).
     *
     * @param unit      Boundary unit that owns the geometry and its local↔world sync.
     * @param tolerance Half-thickness of the surface shell, in local units; a point
     *                  counts as "on the surface" when within this distance. Expected
     *                  finite and non-negative (the `Builder` enforces this).
     * @param spacing   Grid step, in local units, for sampling the bounding box on
     *                  each axis; must be positive for any point to be produced.
     */
    ATLAS_HOST
    SurfaceSource(Unit unit, float tolerance, float spacing);

    /**
     * @brief Returns a fresh `Builder` for fluent construction.
     * @return A default-initialized builder.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Returns the owned boundary unit (geometry plus local↔world sync).
     * @return Const reference to the internal `Unit`; valid for this source's lifetime.
     */
    ATLAS_NODISCARD ATLAS_HOST const Unit&
    unit() const noexcept {
        return _unit;
    }

    /**
     * @brief Number of accepted surface sample points held in the device cache.
     * @return Cache size; also the maximum number of particles a single `spawn`
     *         can emit (an actual `spawn` may write fewer if the target buffer is
     *         nearly full).
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    cached_count() const noexcept {
        return _cache.size();
    }

    /**
     * @brief Advances the owned unit's motion by @p dt seconds.
     *
     * Only the unit's transform changes; the cache is stored in local space and so
     * stays valid across motion — the next `spawn` picks up the new pose through
     * `unit().sync()`.
     *
     * @param dt Timestep in seconds.
     */
    ATLAS_HOST void
    advance(const float dt) noexcept {
        _unit.update(dt);
    }

    /**
     * @brief Emits the cached points into a fluid position buffer, in world space.
     *
     * Launches a device kernel that, for each cached local point `i`, writes
     * `unit().sync().sync_to_world(point)` into `positions->data()[offset + i]`.
     * The number written is clamped to the space remaining in the target buffer,
     * `buffer.size() - offset`, so it never overruns.
     *
     * @param positions Destination fluid position state; a null pointer writes nothing.
     * @param offset    Index of the first slot to write, i.e. the current live
     *                  particle count. When it is at or past the buffer end, nothing
     *                  is written.
     * @return Count of particles actually written (0 when the source is empty, the
     *         pointer is null, or the buffer has no room at @p offset).
     */
    ATLAS_NODISCARD ATLAS_HOST int
    spawn(FluidPositionState* positions, std::size_t offset) const;

private:
    Unit _unit; ///< Boundary unit: owns geometry and the local↔world transform.

    float _tolerance { 0.0f }; ///< Surface-shell half-thickness, in local units.

    float _spacing { 0.1f }; ///< Grid sampling step, in local units.

    DeviceBuffer<Float3> _cache; ///< Accepted local surface points; built once, host-only lifetime.
};

/**
 * @brief Fluent builder for `SurfaceSource`; validates then constructs.
 *
 * Setters return `*this` for chaining. `build()` calls `validate()` (which throws
 * on bad input), constructs the source — triggering the one-time cache build — and
 * then resets the builder's own state so it can be reused.
 */
class SurfaceSource::Builder final {
public:
    /// Constructs a builder with no unit and default tolerance/spacing.
    Builder() = default;

    /**
     * @brief Sets the boundary unit (required).
     * @param unit Unit to emit from; taken by value and moved into the builder.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_unit(Unit unit);

    /**
     * @brief Sets the surface-shell half-thickness.
     * @param tolerance Distance, in local units; must be finite and non-negative.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_tolerance(float tolerance) noexcept;

    /**
     * @brief Sets the bounding-box grid sampling step.
     * @param spacing Step, in local units; must be finite and positive.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_spacing(float spacing) noexcept;

    /**
     * @brief Validates the configuration and builds a `SurfaceSource`.
     *
     * On success the builder's fields are reset (unit cleared, tolerance/spacing
     * back to defaults) so the same builder can be reused.
     *
     * @return The constructed source, with its surface-point cache already built.
     * @throws std::runtime_error if the unit is unset, or tolerance/spacing are
     *         non-finite or out of range.
     */
    ATLAS_NODISCARD ATLAS_HOST SurfaceSource
    build();

    /**
     * @brief Builds the source and wraps it in a host `shared_ptr`.
     * @return Shared owner of a heap-allocated `SurfaceSource`.
     * @throws std::runtime_error under the same conditions as `build()`.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<SurfaceSource>
    make_host_shared();

private:
    /**
     * @brief Throws if the accumulated configuration is invalid.
     * @throws std::runtime_error when the unit is unset, or tolerance/spacing are
     *         non-finite or outside their required ranges.
     */
    ATLAS_HOST void
    validate() const;

private:
    std::optional<Unit> _unit; ///< Pending unit; empty until `with_unit`, required by `validate`.

    float _tolerance { 0.0f }; ///< Pending surface-shell half-thickness.

    float _spacing { 0.1f }; ///< Pending grid sampling step.
};

/// Host-side shared-ownership pointer to a `SurfaceSource`.
using SurfaceSourceHostPtr = atlas::host_shared_ptr<SurfaceSource>;

/// Device-side shared-ownership pointer alias for a `SurfaceSource`.
using SurfaceSourceDevicePtr = atlas::device_shared_ptr<SurfaceSource>;

}