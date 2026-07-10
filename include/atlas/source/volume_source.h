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
 * @brief Source leaf that emits particles from the **interior volume** of a `Unit`.
 *
 * The mirror of `SurfaceSource`: at construction it grid-samples the unit
 * geometry's bounding box at `spacing` intervals and keeps every sample that lies
 * *inside* the geometry within `tolerance`, caching those accepted points — in the
 * unit's **local** frame — into a `DeviceBuffer<Float3>`. The geometry query runs
 * once; each `spawn` only transforms cached local points to world space on device.
 *
 * The only behavioral difference from `SurfaceSource` is the accept predicate:
 * `geometry().is_inside(point, tolerance)` instead of `is_on_surface`.
 *
 * Because it owns a `DeviceBuffer` (host-only copy/destructor), this leaf is
 * move-only and is stored by the `Source` umbrella via `HostVariant`.
 *
 * @note Satisfies `ConceptSource` via `spawn` and `advance`.
 * @see Source, SurfaceSource
 */
class VolumeSource final {
public:
    /// Host-side fluent builder that validates inputs and constructs the leaf.
    class Builder;

public:
    /**
     * @brief Constructs an empty source with a default `Unit` and no cached points.
     *
     * Required so the leaf can be a union member of `Source`; the cache is empty
     * and every `spawn` returns 0 until built via `Builder` / `builder()`.
     */
    VolumeSource() = default;

    /**
     * @brief Constructs the source and builds its local interior-point cache.
     *
     * Takes ownership of @p unit, then grid-samples the unit geometry's bounding
     * box at @p spacing and caches every sample point for which
     * `geometry().is_inside(point, tolerance)` holds. If the geometry's bound is
     * invalid or @p spacing is not positive, the cache is left empty (no throw).
     *
     * @param unit      Boundary unit that owns the geometry and its local↔world sync.
     * @param tolerance Inclusion slack, in local units, passed to `is_inside`;
     *                  expected finite and non-negative (enforced by `Builder`).
     * @param spacing   Grid step, in local units, for sampling the bounding box on
     *                  each axis; must be positive for any point to be produced.
     */
    ATLAS_HOST
    VolumeSource(Unit unit, float tolerance, float spacing);

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
     * @brief Number of accepted interior sample points held in the device cache.
     * @return Cache size; also the maximum number of particles a single `spawn`
     *         can emit (fewer if the target buffer is nearly full).
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

    float _tolerance { 0.0f }; ///< Interior inclusion slack, in local units.

    float _spacing { 0.1f }; ///< Grid sampling step, in local units.

    DeviceBuffer<Float3> _cache; ///< Accepted local interior points; built once, host-only lifetime.
};

/**
 * @brief Fluent builder for `VolumeSource`; validates then constructs.
 *
 * Setters return `*this` for chaining. `build()` calls `validate()` (which throws
 * on bad input), constructs the source — triggering the one-time cache build — and
 * then resets the builder's own state so it can be reused.
 */
class VolumeSource::Builder final {
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
     * @brief Sets the interior inclusion slack.
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
     * @brief Validates the configuration and builds a `VolumeSource`.
     *
     * On success the builder's fields are reset (unit cleared, tolerance/spacing
     * back to defaults) so the same builder can be reused.
     *
     * @return The constructed source, with its interior-point cache already built.
     * @throws std::runtime_error if the unit is unset, or tolerance/spacing are
     *         non-finite or out of range.
     */
    ATLAS_NODISCARD ATLAS_HOST VolumeSource
    build();

    /**
     * @brief Builds the source and wraps it in a host `shared_ptr`.
     * @return Shared owner of a heap-allocated `VolumeSource`.
     * @throws std::runtime_error under the same conditions as `build()`.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<VolumeSource>
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

    float _tolerance { 0.0f }; ///< Pending interior inclusion slack.

    float _spacing { 0.1f }; ///< Pending grid sampling step.
};

/// Host-side shared-ownership pointer to a `VolumeSource`.
using VolumeSourceHostPtr = atlas::host_shared_ptr<VolumeSource>;

/// Device-side shared-ownership pointer alias for a `VolumeSource`.
using VolumeSourceDevicePtr = atlas::device_shared_ptr<VolumeSource>;

}