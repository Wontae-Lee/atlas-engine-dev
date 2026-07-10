#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/parallel/parallel_fill.h>

#include <cstddef>
#include <utility>

namespace atlas {

/**
 * @brief Abstract base for a single per-cell field array owned by a `Universe`.
 *
 * Each concrete subclass wraps one `DeviceBuffer` holding one value per grid
 * cell (a temperature, a velocity, a DSMC counter, ...). Subclasses are stored
 * one-per-type in the `UniverseStateStore` and reached through
 * `Universe::state<T>()`. The base fixes the shared contract — query the length
 * and zero the buffer — and supplies a device-side fill helper for `reset()`.
 *
 * Move-only, since it owns a device buffer whose copy is host-only; the base is
 * polymorphic (virtual destructor) so the store can hold and destroy subclasses
 * through a `std::unique_ptr<UniverseState>`.
 */
class UniverseState {
public:
    /// Default-construct an empty state (no buffer). Subclasses size it later.
    UniverseState() = default;

    /// Deleted: the owned device buffer's copy is host-only and unwanted here.
    UniverseState(const UniverseState&) = delete;

    /// Move construction transfers the device buffer; defaulted.
    UniverseState(UniverseState&&) noexcept = default;

    /// Virtual so subclasses are destroyed correctly through a base pointer.
    virtual ~UniverseState() = default;

    /// Deleted: see the copy constructor.
    UniverseState&
    operator=(const UniverseState&)
        = delete;

    /// Move assignment transfers the device buffer; defaulted.
    UniverseState&
    operator=(UniverseState&&) noexcept = default;

    /**
     * @brief Number of stored elements, i.e. the cell count the buffer was sized to.
     * @return Length of the underlying device buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST virtual std::size_t
    size() const noexcept = 0;

    /**
     * @brief Zero every element of the field for a fresh accumulation pass.
     *
     * Concrete implementations delegate to `reset_buffer`.
     */
    ATLAS_HOST virtual void
    reset()
        = 0;

protected:
    /**
     * @brief Fill a device buffer with the value-initialized element (all zeros).
     *
     * Runs a device-side `parallel_fill` over `[begin, end)`, so the clear
     * happens on the GPU without a host round-trip. Shared by every subclass's
     * `reset()`.
     *
     * @tparam Buffer A `DeviceBuffer`-like type exposing `value_type`/`begin`/`end`.
     * @param buffer Buffer to overwrite with `value_type{}`.
     */
    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_buffer(Buffer& buffer) {
        using value_type = typename Buffer::value_type;
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), value_type {});
    }
};

/**
 * @brief Per-cell gas temperature field, one `float` (kelvin) per grid cell.
 */
class UniverseTemperatureState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseTemperatureState() = default;

    /**
     * @brief Allocate a zero-initialized temperature buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseTemperatureState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled temperature buffer.
     * @param temperature Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseTemperatureState(DeviceBuffer<float> temperature) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the temperature buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the temperature field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the temperature device buffer.
     * @return Reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    /**
     * @brief Const access to the temperature device buffer.
     * @return Const reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _temperature; ///< Per-cell temperature, kelvin.
};

/**
 * @brief Per-cell bulk (mean flow) velocity, one `Float3` per grid cell.
 *
 * The mass-averaged drift velocity of the particles in each cell, in world
 * units per second.
 */
class UniverseBulkVelocityState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseBulkVelocityState() = default;

    /**
     * @brief Allocate a zero-initialized bulk-velocity buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseBulkVelocityState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled bulk-velocity buffer.
     * @param bulk_velocity Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseBulkVelocityState(DeviceBuffer<Float3> bulk_velocity) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the bulk-velocity buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the bulk-velocity field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the bulk-velocity device buffer.
     * @return Reference to the underlying `DeviceBuffer<Float3>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Float3>&
    data() noexcept;

    /**
     * @brief Const access to the bulk-velocity device buffer.
     * @return Const reference to the underlying `DeviceBuffer<Float3>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Float3>&
    data() const noexcept;

private:
    DeviceBuffer<Float3> _bulk_velocity; ///< Per-cell mean flow velocity, world units/s.
};

/**
 * @brief Per-cell external field force, one `Float3` per grid cell.
 *
 * An externally-imposed force (e.g. an electromagnetic body force) applied to
 * the particles occupying each cell, held separately from gravity.
 */
class UniverseFieldForceState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseFieldForceState() = default;

    /**
     * @brief Allocate a zero-initialized field-force buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseFieldForceState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled field-force buffer.
     * @param field_force Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseFieldForceState(DeviceBuffer<Float3> field_force) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the field-force buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the field-force field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the field-force device buffer.
     * @return Reference to the underlying `DeviceBuffer<Float3>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Float3>&
    data() noexcept;

    /**
     * @brief Const access to the field-force device buffer.
     * @return Const reference to the underlying `DeviceBuffer<Float3>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Float3>&
    data() const noexcept;

private:
    DeviceBuffer<Float3> _field_force; ///< Per-cell external field force.
};

/**
 * @brief Per-cell gravitational force/acceleration, one `Float3` per grid cell.
 *
 * Kept as a distinct field from the generic external field force so gravity can
 * be configured or visualized on its own.
 */
class UniverseGravityState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseGravityState() = default;

    /**
     * @brief Allocate a zero-initialized gravity buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseGravityState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled gravity buffer.
     * @param gravity Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseGravityState(DeviceBuffer<Float3> gravity) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the gravity buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the gravity field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the gravity device buffer.
     * @return Reference to the underlying `DeviceBuffer<Float3>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<Float3>&
    data() noexcept;

    /**
     * @brief Const access to the gravity device buffer.
     * @return Const reference to the underlying `DeviceBuffer<Float3>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Float3>&
    data() const noexcept;

private:
    DeviceBuffer<Float3> _gravity; ///< Per-cell gravitational force.
};

/**
 * @brief Per-cell maximum relative speed of colliding pairs, one `float` per cell.
 *
 * A DSMC diagnostic: the largest pair relative speed sampled in the cell during
 * the collision step (`sqrt` of the tracked squared maximum). Written by the
 * DSMC solver each step and exposed for observation.
 */
class UniverseMaxRelativeSpeedState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseMaxRelativeSpeedState() = default;

    /**
     * @brief Allocate a zero-initialized max-relative-speed buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseMaxRelativeSpeedState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled max-relative-speed buffer.
     * @param max_relative_speed Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseMaxRelativeSpeedState(DeviceBuffer<float> max_relative_speed) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the max-relative-speed buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the max-relative-speed field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the max-relative-speed device buffer.
     * @return Reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    /**
     * @brief Const access to the max-relative-speed device buffer.
     * @return Const reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _max_relative_speed; ///< Per-cell max pair relative speed.
};

/**
 * @brief Per-cell running maximum of (cross-section x relative speed), `float`/cell.
 *
 * `(sigma * g)_max` is the key quantity of the DSMC No-Time-Counter (NTC)
 * collision scheme: it bounds the number of candidate collision pairs to draw
 * and the accept/reject probability `sigma*g / (sigma*g)_max`. The solver seeds
 * it, raises it whenever a larger `sigma*g` is sampled, and reads it back when
 * scheduling the next step's candidates.
 */
class UniverseMaxSigmaGState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseMaxSigmaGState() = default;

    /**
     * @brief Allocate a zero-initialized (sigma*g)_max buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseMaxSigmaGState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled (sigma*g)_max buffer.
     * @param max_sigma_g Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseMaxSigmaGState(DeviceBuffer<float> max_sigma_g) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the (sigma*g)_max buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the (sigma*g)_max field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the (sigma*g)_max device buffer.
     * @return Reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    /**
     * @brief Const access to the (sigma*g)_max device buffer.
     * @return Const reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _max_sigma_g; ///< Per-cell NTC bound (sigma*g)_max.
};

/**
 * @brief Per-cell thermal energy, one `float` per grid cell.
 *
 * The internal (thermal) energy accumulated for the particles in each cell.
 */
class UniverseThermalEnergyState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseThermalEnergyState() = default;

    /**
     * @brief Allocate a zero-initialized thermal-energy buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseThermalEnergyState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled thermal-energy buffer.
     * @param thermal_energy Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseThermalEnergyState(DeviceBuffer<float> thermal_energy) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the thermal-energy buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the thermal-energy field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the thermal-energy device buffer.
     * @return Reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    /**
     * @brief Const access to the thermal-energy device buffer.
     * @return Const reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _thermal_energy; ///< Per-cell thermal (internal) energy.
};

/**
 * @brief Per-cell particle number, one `float` per grid cell.
 *
 * The number of real particles represented in each cell. Stored as a `float`
 * (not an integer) so it can carry a fractional statistical weight; the DSMC
 * collision-rate estimate uses it as the pair count `N` in the cell.
 */
class UniverseNumberParticleState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseNumberParticleState() = default;

    /**
     * @brief Allocate a zero-initialized particle-number buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseNumberParticleState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled particle-number buffer.
     * @param number_particle Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseNumberParticleState(DeviceBuffer<float> number_particle) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the particle-number buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the particle-number field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the particle-number device buffer.
     * @return Reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    /**
     * @brief Const access to the particle-number device buffer.
     * @return Const reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _number_particle; ///< Per-cell real-particle count (weight).
};

/**
 * @brief Per-cell collision-candidate count, one `int` per grid cell.
 *
 * Doubles as the DSMC scheduling slot: the solver writes the number of
 * candidate collision pairs to attempt in a cell (the NTC expected count, or 0
 * when the cell is idle), then reads it back to flatten the per-cell candidates
 * into a flat work list for the next pass.
 */
class UniverseCollisionCountState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseCollisionCountState() = default;

    /**
     * @brief Allocate a zero-initialized collision-count buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseCollisionCountState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled collision-count buffer.
     * @param collision_count Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseCollisionCountState(DeviceBuffer<int> collision_count) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the collision-count buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the collision-count field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the collision-count device buffer.
     * @return Reference to the underlying `DeviceBuffer<int>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    data() noexcept;

    /**
     * @brief Const access to the collision-count device buffer.
     * @return Const reference to the underlying `DeviceBuffer<int>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    data() const noexcept;

private:
    DeviceBuffer<int> _collision_count; ///< Per-cell candidate-collision count.
};

/**
 * @brief Per-cell Knudsen number, one `float` per grid cell.
 *
 * The ratio of mean free path to a representative cell length; a dimensionless
 * rarefaction diagnostic used to decide, per cell, whether a continuum or a
 * particle (DSMC) solver is appropriate.
 */
class UniverseKnudsenNumberState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseKnudsenNumberState() = default;

    /**
     * @brief Allocate a zero-initialized Knudsen-number buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseKnudsenNumberState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled Knudsen-number buffer.
     * @param knudsen_number Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseKnudsenNumberState(DeviceBuffer<float> knudsen_number) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the Knudsen-number buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the Knudsen-number field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the Knudsen-number device buffer.
     * @return Reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<float>&
    data() noexcept;

    /**
     * @brief Const access to the Knudsen-number device buffer.
     * @return Const reference to the underlying `DeviceBuffer<float>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<float>&
    data() const noexcept;

private:
    DeviceBuffer<float> _knudsen_number; ///< Per-cell Knudsen number (dimensionless).
};

/**
 * @brief Per-cell owning-solver index, one `int` per grid cell.
 *
 * Populated by the codec's per-cell solver selection: `allocated_solver[cell]`
 * is the index of the solver responsible for that cell. A solver processes a
 * cell only when this equals its own index, so a multi-solver setup partitions
 * the grid without overlap. A null buffer means "every solver owns every cell".
 */
class UniverseAllocatedSolverState final : public UniverseState {
public:
    /// Empty state; size it with the cell-count constructor before use.
    UniverseAllocatedSolverState() = default;

    /**
     * @brief Allocate a zero-initialized owning-solver buffer of `cell_count` entries.
     * @param cell_count Number of grid cells to size the buffer to.
     */
    ATLAS_HOST explicit UniverseAllocatedSolverState(std::size_t cell_count);

    /**
     * @brief Adopt an already-filled owning-solver buffer.
     * @param allocated_solver Device buffer to take ownership of (moved in).
     */
    ATLAS_HOST explicit UniverseAllocatedSolverState(DeviceBuffer<int> allocated_solver) noexcept;

    /**
     * @brief Number of cells the buffer holds.
     * @return Length of the owning-solver buffer.
     */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept override;

    /// Zero the owning-solver field on the device.
    ATLAS_HOST void
    reset() override;

    /**
     * @brief Mutable access to the owning-solver device buffer.
     * @return Reference to the underlying `DeviceBuffer<int>`.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    data() noexcept;

    /**
     * @brief Const access to the owning-solver device buffer.
     * @return Const reference to the underlying `DeviceBuffer<int>`.
     */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    data() const noexcept;

private:
    DeviceBuffer<int> _allocated_solver; ///< Per-cell index of the owning solver.
};

}